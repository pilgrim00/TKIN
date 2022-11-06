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
    input_feats=input_feats[0].cpu().numpy()
    sav_path='/apdcephfs/private_jiaxianchen/PISE/result/'+name+'/'
    if not os.path.exists(sav_path):
        os.mkdir(sav_path)
    for i in range(len(input_feats)):
        img=input_feats[i]
        print(np.min(img),np.max(img))
        #img[img<0.2]=0.0
        #img[img>1.0]=1.0
        img=img*255.0
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

    def forward(self, input):
        #print(input.shape)
        x = self.conv2(self.conv1(input))
        x = self.conv4(self.conv3(x))
        x = self.deform4(self.deform3(x))

        x = self.up2(self.up1(x))
        x = self.up4(self.up3(x))

        #print(x.shape)
        par = self.parout(x)

        
        par = (par+1.)/2.
        

        return par
class ParUNet(nn.Module):
    """
    define a parsing net to generate target parsing
    """
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU(0.2), use_spect=False):
        super(ParUNet, self).__init__()

        self.conv1 = BlockEncoder(input_nc, ngf*2, ngf, norm_layer, act, use_spect)
        self.conv2 = BlockEncoder(ngf*2, ngf*4, ngf*4, norm_layer, act, use_spect)

        self.conv3 = BlockEncoder(ngf*4, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.conv4 = BlockEncoder(ngf*8, ngf*16, ngf*16, norm_layer, act, use_spect)
        self.deform1 = Gated_conv(ngf*16, ngf*16, norm_layer=norm_layer)
        self.deform2 = Gated_conv(ngf*8, ngf*8, norm_layer=norm_layer)
        self.deform3 = Gated_conv(ngf*4, ngf*4, norm_layer=norm_layer)

        self.up1 = ResBlockDecoder(ngf*16, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.up2 = ResBlockDecoder(ngf*16, ngf*4, ngf*8, norm_layer, act, use_spect)


        self.up3 = ResBlockDecoder(ngf*8, ngf*2, ngf*4, norm_layer, act, use_spect)
        self.up4 = ResBlockDecoder(ngf*2, ngf, ngf, norm_layer, act, use_spect)

        self.parout = Output(ngf, 8, 3, norm_layer ,act, None)
        #self.softmax=nn.Softmax(dim=1)
    def forward(self, input):
        #print(input.shape)
        a1=self.conv1(input)
        a2=self.conv2(a1)
        a3=self.conv3(a2)
        a4=self.conv4(a3)
        a4_=self.deform1(a4)
        a3_=self.deform2(a3)
        a2_=self.deform3(a2)
        b1=self.up1(a4_)
        #print(a3_.dtype,b1.dtype)
        #print(a3_.shape,b1.shape)
        b2=self.up2(torch.cat([a3_,b1],dim=1))
        b3=self.up3(torch.cat([a2_,b2],dim=1))
        b4=self.up4(b3)
        par=self.parout(b4)
        par = (par+1.)/2.
        #par=self.softmax(par)
        return par
class PoseGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, norm='instance', 
                activation='LeakyReLU', use_spect=True, use_coord=False,use_sn=None,gt_par=None,
                use_rec=None,ker_size=5,patch_dec=None,use_bank=None,parnet_pise=None):
        super(PoseGenerator, self).__init__()
        self.gt_par=gt_par
        self.use_sn=use_sn
        self.use_coordconv = True
        self.match_kernel = 3
        self.use_rec=use_rec
        self.use_bank=use_bank
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        if parnet_pise==1:
            self.parnet=ParsingNet(input_nc=20, output_nc=8)
        else:
            self.parnet=ParUNet(input_nc=20, output_nc=8)
        self.Zencoder = twoencoder(3, ngf,ker_size=ker_size)

        self.imgenc = VggEncoder()

        self.parenc = HardEncoder(8+20, ngf)#BasicEncoder(28,ngf)
        self.dec = BasicDecoder(3)
        if patch_dec==1:
            self.efb = patchdecoder(ngf*4, 64,256,norm_layer=norm_layer,ker_size=ker_size)
        if use_bank==1:
            self.efb = bankdecoder(ngf*4, 64,256,norm_layer=norm_layer,ker_size=ker_size)
        else:
            self.efb = twodecoder(ngf*4, 64,256,norm_layer=norm_layer,ker_size=ker_size)
        self.res = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
                
        self.res1 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)

   

    def forward(self, img1, img2, pose1, pose2, par1, par2):
        #generate parsemap
        par2_=self.parnet(pose2)#0-1之间
        par1_=self.parnet(pose1)
        shape = self.parenc(torch.cat([pose2,par2_], 1))
        #visualize_features(parcode,'parcode')
        style_kernels, exist_vector= self.Zencoder(img1, par1)
        parcode = self.efb(style_kernels,exist_vector,shape,par2_)
        #visualize_features(parcode,'parcode2')
        parcode = self.res(parcode)
        #visualize_features(parcode,'parcode3')
        img2code = self.imgenc(img2)
        #visualize_features(img2code,'img2code')
        loss_reg = F.mse_loss(img2code, parcode)

        parcode = self.res1(parcode)
        img_gen = self.dec(parcode)
        
        return img_gen, loss_reg,par2_,par1_
        
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

