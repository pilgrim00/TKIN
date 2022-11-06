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
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from tqdm import tqdm
from inception import InceptionV3
from PIL import Image
from scipy import linalg
import math
from torchvision import models
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()
        path='/remote-home/pilgrim02/ft_local/inception_v3_google-1a9a5a14.pth'
        inception = models.inception_v3(pretrained=False)
        inception.load_state_dict(torch.load(path))

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in 
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output 
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                            size=(299, 299),
                            mode='bilinear',
                            align_corners=False)

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

#写一个dataset,已知文件夹
class dataset_gt(Dataset):
    def __init__(self,transform=None, resolution=256,test_path=None,gt_path=None,crop=0):
        self.test_path=test_path
        self.gt_path=gt_path
        self.res=resolution
        self.index=0
        self.transform=transform
        self.get_image()
        self.crop=crop
        print('Images: {},{}'.format(len(self.images_path),len(self.gt_images_path)))
    def get_image(self):
        self.images_path=[]
        self.gt_images_path=[]
        
        for img in sorted(os.listdir(self.test_path)):
                img_path=self.test_path+'/'+img
                gt_name='fashion'+img.split('fashion')[-1]
                gt_name=gt_name.split('_vis')[0]+gt_name.split('_vis')[1]
                #print(gt_name)
                gt_img_path=self.gt_path+'/'+gt_name
                self.images_path.append(img_path)
                self.gt_images_path.append(gt_img_path)
        '''
        for img in sorted(os.listdir(self.gt_path)):
                gt_img_path=self.gt_path+'/'+img
                self.gt_images_path.append(gt_img_path)
        '''
    def __getitem__(self, index):
        regions = (40,0,216,256)
        #test_img=self.images_path[index]
        gt_img=self.gt_images_path[index]
        if self.crop==0:
            #test_image=Image.open(test_img)#(256,256,3)
            gt_image=Image.open(gt_img)
        if self.crop==1:
            #test_image=Image.open(test_img).crop(regions)#(256,256,3)
            gt_image=Image.open(gt_img).crop(regions)
        return self.transform(gt_image)
    def __len__(self):
        #这里是个整数才行
        return len(self.images_path)
class dataset_test(Dataset):
    def __init__(self,transform=None, resolution=256,path='/remote-home/share/inshop/',crop=0,divide=0):
        self.image_path=path
        self.res=resolution
        self.index=0
        self.transform=transform
        self.get_image()
        assert (crop*divide!=1), 'can not equals to one at the same time '
        self.crop=crop
        self.divide=divide
        print('Images: {}'.format(len(self.images_path)))
    def get_image(self):
        self.images_path=[]
        for img in sorted(os.listdir(self.image_path)):
                img_path=self.image_path+'/'+img
                self.images_path.append(img_path)

    def __getitem__(self, index):
        img=self.images_path[index]
        regions = (40,0,216,256)
        regions_div=(176*4,0,880,256)
        image=Image.open(img)
        #image=F_t.resize(image,(256,176))
        if self.divide==1:
            image=image.crop(regions_div)
        if self.crop==1:
            image=image.crop(regions)
        return self.transform(image)
    def __len__(self):
        return len(self.images_path)
@torch.no_grad()
def extract_features(loader, inception):
    pbar = tqdm(loader)
    feature_list=[]
    for img in pbar:
        img = img.cuda()
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature)
    features = torch.cat(feature_list, dim=0)

    return features
def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="fid test")
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
    gt_test_path='/remote-home/share/inshop/eval_results/latest'
    #test_path='/mnt/private_jiaxianchen/PISE_out/result_wgtpar/eval'
    #test_path='/mnt/share_1227775/BaseAI/workspace/jiaxianchen/inshop/PISE/eval_results/latest'
    test_path=args.path
    gt_path='/remote-home/share/inshop/test'
    gt_dataset=dataset_gt(transform=transform2,test_path=gt_test_path,gt_path=gt_path,crop=args.crop)
    if args.pad==1:
        test_dataset=dataset_test(transform=transform1,path=test_path,crop=args.crop,divide=args.divide)
    else:
        test_dataset=dataset_test(transform=transform2,path=test_path,crop=args.crop,divide=args.divide)
    gt_loader = DataLoader(
        gt_dataset,
        batch_size=64,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx],resize_input=True,normalize_input=True).eval().cuda()
    gt_features = extract_features(gt_loader, inception).cpu().numpy()
    gt_features = gt_features[: 60000]

    #print(f"extracted {gt_features.shape[0]} features")

    gt_mean = np.mean(gt_features, 0)
    gt_cov = np.cov(gt_features, rowvar=False)

    out_features = extract_features(test_loader, inception).cpu().numpy()
    out_features = out_features[: 60000]

    #print(f"extracted {out_features.shape[0]} features")

    out_mean = np.mean(out_features, 0)
    out_cov = np.cov(out_features, rowvar=False)
    fid=calc_fid(out_mean, out_cov, gt_mean, gt_cov, eps=1e-6)
    print(test_path,'fid_GFLA:',fid)