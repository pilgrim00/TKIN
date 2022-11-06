import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.transforms.functional as F
from util import pose_utils
from PIL import Image
import pandas as pd
import torch
import math
import numbers
class BaseDataset_dtd(data.Dataset):
    def __init__(self):
        super(BaseDataset_dtd, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--angle', type=float, default=False)
        parser.add_argument('--shift', type=float, default=False)
        parser.add_argument('--scale', type=float, default=False)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.name_pairs, self.par_dir,self.dtd_images_path = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size
        self.class_num = 8

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size


        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list) 

        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        par_paths = []
        dtd_images_path = []#共5640张图，存在不一致。
        assert False, "A subclass of MarkovAttnDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths, par_paths,dtd_images_path

    def __getitem__(self, index):
        #1.dtd_img.texture
        dtd_index=index%5640
        dtd_path=self.dtd_images_path[dtd_index]
        dtd_img=Image.open(dtd_path).convert('RGB')
        dtd_img=dtd_img.crop((0,0,256,256))#(256,256)
        dtd_img=self.trans(dtd_img)
        #2.the deepfashion dataset in the pose-transfer task
        P1_name, P2_name = self.name_pairs[index]
        P1_path = os.path.join(self.image_dir, P1_name) # person 1
        P2_path = os.path.join(self.image_dir, P2_name) # person 2

        SPL1_path = os.path.join(self.par_dir, P1_name[:-4]+'.png')
        SPL2_path = os.path.join(self.par_dir, P2_name[:-4]+'.png')

        regions = (40,0,216,256)
        P1_img = Image.open(P1_path).convert('RGB')#.crop(regions)
        P2_img = Image.open(P2_path).convert('RGB')#.crop(regions)
        SPL1_img = Image.open(SPL1_path)#.crop(regions)
        SPL2_img = Image.open(SPL2_path)#.crop(regions)(256,256)
        
        if np.array(P1_img).shape[1]==176:
            tmp = np.ones([256, 40, 3])*255
            P1_img = Image.fromarray(np.uint8(np.concatenate([tmp, np.array(P1_img), tmp],1)))
            P2_img = Image.fromarray(np.uint8(np.concatenate([tmp, np.array(P2_img), tmp],1)))
        
        s1np = np.expand_dims(np.array(SPL1_img),-1)
        s2np = np.expand_dims(np.array(SPL2_img), -1)#(256,256,1)
        s1np = np.concatenate([s1np,s1np,s1np], -1)
        s2np = np.concatenate([s2np,s2np,s2np], -1)#(256,256,3)
        SPL1_img = Image.fromarray(np.uint8(s1np))
        SPL2_img = Image.fromarray(np.uint8(s2np))#(256,256,3)

        angle, shift, scale = self.getRandomAffineParam()
        P1_img = F.affine(P1_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        SPL1_img = F.affine(SPL1_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        if self.opt.point_line==1:
            BP1,LP1=self.obtain_bone_line(P1_name)
        else:
            BP1 = self.obtain_bone(P1_name, affine_matrix)
        P1 = self.trans(P1_img)




        angle, shift, scale = self.getRandomAffineParam()
        angle, shift, scale = angle*0.2, (shift[0]*0.5,shift[1]*0.5), 1 # Reduce the deform parameters of the generated image
        P2_img = F.affine(P2_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        SPL2_img = F.affine(SPL2_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        if self.opt.point_line==1:
            BP2,LP2=self.obtain_bone_line(P2_name)
        else:
            BP2 = self.obtain_bone(P2_name, affine_matrix)
        P2 = self.trans(P2_img)

        SPL1_img = np.expand_dims(np.array(SPL1_img)[:,:,0],0)#[:,:,40:-40] # 1*256*176
        SPL2_img = np.expand_dims(np.array(SPL2_img)[:,:,0],0)#(1,256,256)

        _, h, w = SPL2_img.shape
       # print(SPL2_img.shape,SPL1_img.shape)
        num_class = self.class_num
        tmp = torch.from_numpy(SPL2_img).view( -1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL2_onehot = ones.view([h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL2_onehot = SPL2_onehot.permute(2,0,1)


        tmp = torch.from_numpy(SPL1_img).view( -1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL1_onehot = ones.view([h,w, num_class])
        SPL1_onehot = SPL1_onehot.permute(2,0,1)

        SPL2 = torch.from_numpy(SPL2_img).long()#(1,256,256)
        SPL1 = torch.from_numpy(SPL1_img).long()#(1,256,256)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,'SPL1': SPL1_onehot, 'SPL2':SPL2_onehot,'dtd_img':dtd_img,
                'P1_path': P1_name, 'P2_path': P2_name,'label_P2': SPL2,'label_P1':SPL1}


    def obtain_bone(self, name, affine_matrix):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose  = pose_utils.cords_to_map(array, self.load_size, self.opt.old_size, affine_matrix)
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose  
    def obtain_bone_line(self, name):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        mask,limbs=pose_=pose_utils.new_draw_pose_from_cords(array,(256,256))#[256,256,3],[19,256,256]
        #cv2.imwrite('/apdcephfs/private_jiaxianchen/PISE/result/color'+name+'.png',colors)
        #cv2.imwrite('/apdcephfs/private_jiaxianchen/PISE/result/mask'+name+'.png',mask)
        #data type and range 
        limbs=limbs.astype(np.float32)
        mask=mask.astype(np.float32)
        pose = np.transpose(mask,(2, 0, 1))
        pose = torch.Tensor(pose)
        limbs = torch.Tensor(limbs)
        return pose,limbs  
   

    def __len__(self):
        return self.dataset_size

    def name(self):
        assert False, "A subclass of BaseDataset must override self.name"

    def getRandomAffineParam(self):
        if self.opt.angle is not False:
            angle = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1])
        else:
            angle = 0
        if self.opt.scale is not False:
            scale   = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1])
        else:
            scale=1
        if self.opt.shift is not False:
            shift_x = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
            shift_y = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
        else:
            shift_x=0
            shift_y=0
        return angle, (shift_x,shift_y), scale

    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
        #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
        #                              [     0                  0          1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1


        angle = math.radians(angle)
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = [math.radians(s) for s in shear]
        elif isinstance(shear, numbers.Number):
            shear = math.radians(shear)
            shear = [shear, 0]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix

    def get_affine_matrix(self, center, angle, translate, scale, shear):
        matrix_inv = self.get_inverse_affine_matrix(center, angle, translate, scale, shear)

        matrix_inv = np.matrix(matrix_inv).reshape(2,3)
        pad = np.matrix([0,0,1])
        matrix_inv = np.concatenate((matrix_inv, pad), 0)
        matrix = np.linalg.inv(matrix_inv)
        return matrix
 

class DtdfashionDataset(BaseDataset_dtd):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset_dtd.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 256))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser



    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        pairLst = os.path.join(root, 'fasion-pairs-%s.csv' %phase)#'fasion-pairs-%s.csv' % phase)
#        pairLst = os.path.join(root, 'arbf_pres.csv')
        name_pairs = self.init_categories(pairLst)
        
        image_dir = os.path.join(root, '%s' % phase)
        bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' %phase)#'fasion-annotation-%s.csv' % phase)
        par_dir = os.path.join(root, '%sSPL8' %phase)
        dtd_path=opt.dtd_path
        dtd_images_path=[]#2354
        #for imgs_path in sorted(os.listdir(self.image_path)):
        for catg in sorted(os.listdir(dtd_path)):
            for img in sorted(os.listdir(dtd_path+'/'+catg)):
                if 'jpg' in img:
                    img_path=dtd_path+'/'+catg+'/'+img

                    dtd_images_path.append(img_path)
        return image_dir, bonesLst, name_pairs, par_dir,dtd_images_path


    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')  
        return pairs    

    def name(self):
        return "DTDfashionDataset"