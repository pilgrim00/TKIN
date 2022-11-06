import argparse
import pickle
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
import math
class dataset_test(Dataset):
    def __init__(self,transform=None, resolution=256,test_path=None,gt_path=None):
        self.test_path=test_path
        self.gt_path=gt_path
        self.res=resolution
        self.index=0
        self.transform=transform
        self.get_image()
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

    def __getitem__(self, index):
        test_img=self.images_path[index]
        test_image=Image.open(test_img)#(256,256,3)
        gt_img=self.gt_images_path[index]
        gt_image=Image.open(gt_img)
        return self.transform(gt_image),self.transform(test_image)
    def __len__(self):
        #这里是个整数才行
        return len(self.images_path)
def psnr(img1, img2): #这里输入的是（0,255）的灰度或彩色图像，如果是彩色图像，则numpy.mean相当于对三个通道计算的结果再求均值

    diff=(img1-img2)**2
    #print(diff.mean(dim=[1,2]))
    mse = diff.mean(dim=[1,2]).mean()
    if mse < 1.0e-10: # 如果两图片差距过小代表完美重合
        return 100
    PIXEL_MAX = 1.0
    out=20*torch.log10(PIXEL_MAX/torch.sqrt(mse))
    return out.item()
    #return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) # 将对数中pixel_max的平方放了下来
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="fid test")
    parser.add_argument("--path", type=str,default='/apdcephfs/private_jiaxianchen/PISE_out/result_wgtpar/eval', help="path to load the dataset")
    args = parser.parse_args()
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    #test_path='/apdcephfs/private_jiaxianchen/PISE_out/result_orig/eval'
    test_path=args.path
    #test_path='/apdcephfs/share_1227775/BaseAI/workspace/jiaxianchen/inshop/PISE/eval_results/latest'
    gt_path='/apdcephfs/share_1227775/BaseAI/workspace/jiaxianchen/inshop/PISE/simple_img'
    #输入为0-1之间的图像,共8750张图片
    dataset=dataset_test(transform=transform,test_path=test_path,gt_path=gt_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )
    psnr_=0
    for index,data in enumerate(dataloader):
        img1=data[0][0]
        img2=data[1][0]
        out=psnr(img1,img2)
        psnr_+=out
        print(index,out)
    print(psnr_/len(dataset))


