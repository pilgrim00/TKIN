import pytorch_ssim
import argparse
import pickle
import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as F_t
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception3
import numpy as np
from tqdm import tqdm
from inception import InceptionV3
from PIL import Image
from scipy import linalg
import math
#写一个dataset,已知文件夹
class dataset_gt(Dataset):
    def __init__(self,test_transform=None, gt_transform=None,resolution=256,test_path=None,gt_path=None,crop=0,divide=0):
        self.test_path=test_path
        self.gt_path=gt_path
        self.res=resolution
        self.index=0
        self.exm_path='/remote-home/share/inshop/eval_results/latest'
        assert (crop*divide!=1), 'can not equals to one at the same time '
        self.test_transform=test_transform
        self.gt_transform=gt_transform
        self.get_image()
        self.crop=crop
        self.divide=divide
        print('Images: {},{}'.format(len(self.images_path),len(self.gt_images_path)))
    def get_image(self):
        self.images_path=[]
        self.gt_images_path=[]
        for img in sorted(os.listdir(self.exm_path)):
                img_path=self.exm_path+'/'+img
                gt_name='fashion'+img.split('fashion')[-1]
                gt_name=gt_name.split('_vis')[0]+gt_name.split('_vis')[1]
                gt_img_path=self.gt_path+'/'+gt_name
                self.gt_images_path.append(gt_img_path)
        for img in sorted(os.listdir(self.test_path)):
                img_path=self.test_path+'/'+img
                self.images_path.append(img_path)

    def __getitem__(self, index):
        regions = (40,0,216,256)
        regions_div=(176*4,0,880,256)
        test_img=self.images_path[index]
        gt_img=self.gt_images_path[index]
        test_image=Image.open(test_img)#(256,256,3)
        #test_image=F_t.resize(test_image,(256,176))
        gt_image=Image.open(gt_img)
        if self.divide==1:
            test_image=test_image.crop(regions_div)#(256,256,3)
        if self.crop==1:
            test_image=test_image.crop(regions)#(256,256,3)
            gt_image=gt_image.crop(regions)
        return {'test':self.test_transform(test_image),'gt':self.gt_transform(gt_image)}
    def __len__(self):
        #这里是个整数才行
        return len(self.images_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="ssim test")
    parser.add_argument("--path", type=str,default='/mnt/private_jiaxianchen/PISE_out/result_wgtpar/eval', help="path to load the dataset")
    parser.add_argument("--crop", type=int,default=0, help="whether to crop the images")
    parser.add_argument("--pad", type=int,default=0, help="whether to crop the images")
    parser.add_argument("--divide", type=int,default=0, help="whether to crop the images")
    args = parser.parse_args()
    transform1 = transforms.Compose(
            [   
                transforms.Pad(padding=(40, 0),padding_mode='edge'),
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
    transform2 = transforms.Compose(
            [   
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
    test_path='/mnt/share_1227775/BaseAI/workspace/jiaxianchen/inshop/PISE/eval_results/latest'
    test_path='/mnt/private_jiaxianchen/PISE_out/10_7_two_k7/eval'
    gt_path='/remote-home/share/inshop/test'
    test_path=args.path
    #gt_dataset中传入test_path是为了筛选一致的图片。
    if args.pad==1:
        gt_dataset=dataset_gt(test_transform=transform1,gt_transform=transform2,test_path=test_path,gt_path=gt_path,crop=args.crop,divide=args.divide)
    else:
        gt_dataset=dataset_gt(test_transform=transform2,gt_transform=transform2,test_path=test_path,gt_path=gt_path,crop=args.crop,divide=args.divide)
    batch=1
    loader = DataLoader(
        gt_dataset,
        batch_size=batch,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
        shuffle=True,
    )
    num=0
    out=0
    for i,data in enumerate(loader):
        #print(num)
        img1,img2=data['test'],data['gt']
        #print(img1.shape,img2.shape)
        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()
        out+=pytorch_ssim.ssim(img1, img2).item()
        num+=1
    #print(num)
    print(test_path,'ssim:',out/num)

        #ssim_loss = pytorch_ssim.SSIM(window_size = 10)

        #print(ssim_loss(img1, img2))
    