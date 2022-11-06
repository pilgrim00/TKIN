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
#from util.util import feature_normalize
#from model.networks.patn import PATNModel
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
        return (x-np.min(x))/(np.max(x)-np.min(x))
    input_feats=input_feats[0].cpu().numpy()
    sav_path='/apdcephfs/private_jiaxianchen/PISE/result/'+name+'/'
    if not os.path.exists(sav_path):
        os.mkdir(sav_path)
    for i in range(len(input_feats)):
        img=norm(input_feats[i])*255.0
        cv2.imwrite(sav_path+str(i)+'feature.png',img)

class PoseGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, norm='instance', 
                activation='LeakyReLU', use_spect=True, use_coord=False,use_sn=None,gt_par=None,
                use_rec=None,use_shape=None,use_bank=None,use_bank_org=None):
        super(PoseGenerator, self).__init__()
        self.use_shape=use_shape
        self.gt_par=gt_par
        self.use_sn=use_sn
        self.use_coordconv = True
        self.match_kernel = 3
        self.use_rec=use_rec
        self.use_bank=use_bank
        self.use_bank_org=use_bank_org
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        #self.parnet = ParsingNet(8+18*2, 8)
        #self.parnet=PATNModel([8,36],8)

        self.Zencoder = dtdencoder(3, ngf)

        self.imgenc = VggEncoder()

        self.parenc = HardEncoder(8+18, ngf)
        self.dec = BasicDecoder(3)
        #self.banks=[]
        self.efb = EFB_dtd(ngf*4, 256)
        self.res = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
                
        self.res1 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
        self.res_dtd = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
        self.res3 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
        self.dec_dtd = BasicDecoder(3)
    def code_reg(self,code1,exits1,code2,exists2):
        eps=1e-7
        common_exits=exits1*exists2
        common_exits=common_exits.reshape(*common_exits.shape,1)
        code1=code1[:,:8,:]*common_exits
        code2=code2[:,:8,:]*common_exits
        loss=F.mse_loss(code1,code2)
        return loss
    def dtd_efg(self,code,par):
        bs,c,h,w=par.shape

    def forward(self, img1, img2, pose1, pose2, par1, par2,dtd):
        bs,_,_,_=img1.shape
        parcode2 = self.parenc(torch.cat((pose2,par2), 1))#[1,256,64,64]
        codes_vector1, exist_vector1, z_dtd,kl_loss1 = self.Zencoder(img1, par1,dtd)
        codes_vector2, exist_vector2,_ ,kl_loss2= self.Zencoder(img2, par2,dtd)
        code_reg=self.code_reg(codes_vector1,exist_vector1,codes_vector2,exist_vector2)


        parcode2,dtd_feat = self.efb(parcode2, par2, codes_vector1, exist_vector1,z_dtd)
        dtd_feat=self.res3(dtd_feat)
        dtd_code = self.imgenc(dtd)
        loss_reg = F.mse_loss(dtd_feat, dtd_code)
        dtd_feat=self.res_dtd(dtd_feat)
        dtd_img=self.dec_dtd(dtd_feat)
        parcode2 = self.res(parcode2)#torch.Size([1, 256, 64, 64])
        img2code = self.imgenc(img2)
        loss_reg += F.mse_loss(img2code, parcode2)

        parcode2 = self.res1(parcode2)
        img_gen = self.dec(parcode2)
        
        #reconstruction
        parcode1 = self.parenc(torch.cat((pose1,par1), 1))
        parcode1,_ = self.efb(parcode1, par1, codes_vector1, exist_vector1,z_dtd)
        parcode1 = self.res(parcode1)

        img1code = self.imgenc(img1)
        loss_reg += F.mse_loss(img1code, parcode1)

        parcode1 = self.res1(parcode1)
        img_orig = self.dec(parcode1)
        return img_gen, loss_reg,img_orig,dtd_img,code_reg,kl_loss1+kl_loss2
        
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

