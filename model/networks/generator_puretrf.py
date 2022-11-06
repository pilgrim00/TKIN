import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from model.networks.base_function import *
#from base_network import BaseNetwork
#from base_function import *
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from model_GFLA.networks.generator  import *
#from util.util import feature_normalize
from model.networks.patn import PATNModel
import numpy as np
import cv2
import os
from model.networks.bank import bank,my_pass,bank_org
def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, p=2, dim=1, keepdim=True) + sys.float_info.epsilon
    feature_normed = torch.div(feature_in, feature_in_norm)
    return feature_normed
def visualize_features(input_feats,name):

    def norm(x):
        return (x-np.min(x))/(np.max(x)-np.min(x)+1e-8)
    input_feats=input_feats[0].cpu().numpy()
    sav_path='/apdcephfs/private_jiaxianchen/PISE/result/'+name+'/'
    if not os.path.exists(sav_path):
        os.mkdir(sav_path)
    for i in range(len(input_feats)):
        img=norm(input_feats[i])*255.0
        img=img.astype(np.uint8)
        #transfer to heatmap img
        img=cv2.applyColorMap(img,cv2.COLORMAP_JET)
        cv2.imwrite(sav_path+str(i)+'feature.png',img)



class PoseGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, norm='instance', 
                activation='LeakyReLU', use_spect=True, use_coord=False,use_sn=None,gt_par=None,
                use_rec=None,ker_size=5,patch_dec=None):
        super(PoseGenerator, self).__init__()
        self.gt_par=gt_par
        self.use_sn=use_sn
        self.use_coordconv = True
        self.match_kernel = 3
        self.use_rec=use_rec
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self.Zencoder = twoencoder(3, ngf,ker_size=ker_size)

        self.imgenc = VggEncoder()

        self.parenc = HardEncoder(8+18, ngf)
        self.dec = BasicDecoder(3)
        if patch_dec==1:
            self.efb = patchdecoder(ngf*4, 64,256,norm_layer=norm_layer,ker_size=ker_size)
        else:
            self.efb = twodecoder(ngf*4, 64,256,norm_layer=norm_layer,ker_size=ker_size)
        self.res = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
                
        self.res1 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)

   

    def forward(self, img1, img2, pose1, pose2, par1, par2,img3,par3,pose3):
        #visualize_features(pose1,'pose1')
        #visualize_features(par1,'par1')
        #Fp
        shape = self.parenc(torch.cat((pose2,par2), 1))
        #visualize_features(parcode,'parcode')
        style_kernels, exist_vector= self.Zencoder(img1, par1)
        parcode = self.efb(style_kernels,exist_vector,shape,par2)
        parcode = self.res(parcode)
        parcode = self.res1(parcode)
        img_gen = self.dec(parcode)
        #------------------------------------------
        #par1=self.change_par(par1)
        shape1 = self.parenc(torch.cat((pose1,par1), 1))
        style_kernels3, exist_vector3= self.Zencoder(img3, par3)
        style_kernels1, exist_vector1= self.Zencoder(img1, par1)
        style_kernels1[:,3,:,:,:]=style_kernels3[:,3,:,:,:]
        parcode = self.efb(style_kernels1,exist_vector1,shape1,par1)
        parcode = self.res(parcode)
        parcode = self.res1(parcode)
        img_trf = self.dec(parcode)
        return img_gen,img_trf
    def change_par(self,par1):
        #1,8,256,256
        par1[:,3,:,:]+=par1[:,5,:,:]
        par1[:,5,:,:]=0.0
        return par1
if __name__=='__main__':
    a1=torch.randn(1,18,256,256).cuda()
    a2=torch.randn(1,18,256,256).cuda()
    b1=torch.randn(1,8,256,256).cuda()
    #model=ParsingNet()
    model=refine_ParsingNet().cuda()
    out=model(a1,b1,a2)
    #out=model(torch.cat([a1,b1,a2],dim=1))

    print(out.shape)
    tar=torch.randn(1,8,256,256).cuda()
    loss=(tar-out)**2
    loss=loss.mean()
    #loss.backward()
    #相当于是通道的维度进行了消失，每个空间位置h,w上的点都是不同通道上的norm之和。

