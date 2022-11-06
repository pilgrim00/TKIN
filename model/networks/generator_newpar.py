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
#from model_GFLA.networks.generator  import *
#from model.networks.patn import PATNModel
import numpy as np
import cv2
import os
def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, p=2, dim=1, keepdim=True) + sys.float_info.epsilon
    feature_normed = torch.div(feature_in, feature_in_norm)
    return feature_normed
def visualize_features(input_feats,name):

    def norm(x):
        return (x-np.min(x))/(np.max(x)-np.min(x)+1e-8)
    input_feats=input_feats[0].cpu().detach().numpy()
    sav_path='/apdcephfs/private_jiaxianchen/PISE/result/'+name+'/'
    if not os.path.exists(sav_path):
        os.mkdir(sav_path)
    for i in range(len(input_feats)):
        img=input_feats[i]
        print(np.min(img),np.max(img))
        #img[img<0.2]=0.0
        #img[img>1.0]=1.0
        img=norm(img)*255.0
        img=img.astype(np.uint8)
        #transfer to heatmap img
        img=cv2.applyColorMap(img,cv2.COLORMAP_JET)
        cv2.imwrite(sav_path+str(i)+'feature.png',img)


class ParsingNet(nn.Module):
    """
    define a parsing net to generate target parsing
    """
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU(0.2), use_spect=False):
        super(ParsingNet, self).__init__()

        self.conv1 = BlockEncoder(input_nc, ngf*2, ngf, norm_layer, act, use_spect)
        self.conv2 = BlockEncoder(ngf*2, ngf*4, ngf*4, norm_layer, act, use_spect)

        self.conv3 = BlockEncoder(ngf*4, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.conv4 = BlockEncoder(ngf*8, ngf*16, ngf*16, norm_layer, act, use_spect)
        self.deform3 = Gated_conv(ngf*16, ngf*16, norm_layer=norm_layer)
        self.deform4 = Gated_conv(ngf*16, ngf*16, norm_layer=norm_layer)

        self.up1 = ResBlockDecoder(ngf*16, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.up2 = ResBlockDecoder(ngf*8, ngf*4, ngf*4, norm_layer, act, use_spect)


        self.up3 = ResBlockDecoder(ngf*4, ngf*2, ngf*2, norm_layer, act, use_spect)
        self.up4 = ResBlockDecoder(ngf*2, ngf, ngf, norm_layer, act, use_spect)

        self.parout = Output(ngf, 8, 3, norm_layer ,act, None)
        self.makout = Output(ngf, 1, 3, norm_layer, act, None)

    def forward(self, input):
        #print(input.shape)
        x = self.conv2(self.conv1(input))
        x = self.conv4(self.conv3(x))
        x = self.deform4(self.deform3(x))

        x = self.up2(self.up1(x))
        x = self.up4(self.up3(x))

        #print(x.shape)
        par = self.parout(x)
        #mak = self.makout(x)
        
        par = (par+1.)/2.
        

        return par
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
        self.parnet=ParsingNet(input_nc=18*2+8, output_nc=8)
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

   

    def forward(self, img1, img2, pose1, pose2, par1, par2):
        #generate parsemap
        par2_=self.parnet(torch.cat([pose1,pose2,par1],dim=1))#0-1之间
        #visualize_features(pose1,'pose1')
        #visualize_features(par2_,'par_fake_newpar')
        #visualize_features(par2,'par_true_newpar')
        #exit()
        #print(torch.min(par2),torch.max(par2))
        #print(torch.min(par2_),torch.max(par2_))
        #exit()
        #Fp
        parcode = self.parenc(torch.cat([pose2,par2_], 1))
        #visualize_features(parcode,'parcode')
        style_kernels, exist_vector= self.Zencoder(img1, par1)
        parcode = self.efb(style_kernels,exist_vector,parcode,par2_)
        #visualize_features(parcode,'parcode2')
        parcode = self.res(parcode)
        #visualize_features(parcode,'parcode3')
        img2code = self.imgenc(img2)
        #visualize_features(img2code,'img2code')
        loss_reg = F.mse_loss(img2code, parcode)

        parcode = self.res1(parcode)
        #visualize_features(parcode,'parcode4')
        #exit()
        img_gen = self.dec(parcode)
        
        #reconstruction
        if self.use_rec==1:
            parcode1 = self.parenc(torch.cat([pose1,par1_], 1))
            parcode1 = self.efb(style_kernels,exist_vector,parcode,par1_)
            parcode1 = self.res(parcode1)

            img1code = self.imgenc(img1)
            loss_reg += F.mse_loss(img1code, parcode1)

            
            parcode1 = self.res1(parcode1)
            img_orig = self.dec(parcode1)
            return img_gen, loss_reg,img_orig,par2_

        return img_gen, loss_reg,par2_
        
if __name__=='__main__':
    a1=torch.randn(1,1,256,256).cuda()
    a2=torch.randn(1,1,256,256).cuda()
    b1=torch.randn(1,8,256,256).cuda()
    model=ParUNet(input_nc=2+8, output_nc=8).cuda()
    out=model(torch.cat([a1,b1,a2],dim=1))
    tar=torch.randn(1,8,256,256).cuda()
    loss=(tar-out)**2
    loss=loss.mean()
    #loss.backward()
    #相当于是通道的维度进行了消失，每个空间位置h,w上的点都是不同通道上的norm之和。

