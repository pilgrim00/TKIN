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
        return (x-np.min(x))/(np.max(x)-np.min(x))
    input_feats=input_feats[0].cpu().numpy()
    sav_path='/apdcephfs/private_jiaxianchen/PISE/result/'+name+'/'
    if not os.path.exists(sav_path):
        os.mkdir(sav_path)
    for i in range(len(input_feats)):
        img=norm(input_feats[i])*255.0
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


class refine_ParsingNet(nn.Module):
    """
    define a parsing net to generate target parsing
    """
    def __init__(self,norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU(0.2), use_spect=False):
        super(refine_ParsingNet, self).__init__()
        self.conv_share_1=nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                norm_layer(1),
                act,
                nn.Upsample(scale_factor=0.5, mode='nearest')
        )
        self.conv_share_2=nn.Sequential(
                nn.Conv2d(1+1, 1, kernel_size=3, padding=1),
                norm_layer(1),
                act,
                nn.Upsample(scale_factor=0.5, mode='nearest')
        )
        self.conv_share_3=nn.Sequential(
                nn.Conv2d(1+2, 1, kernel_size=3, padding=1),
                norm_layer(1),
                act,
                nn.Upsample(scale_factor=0.5, mode='nearest')
        )
        self.conv_mask_1=nn.Sequential(
                nn.Conv2d(1,1,kernel_size=3, padding=1),
                norm_layer(1),
                act,
                nn.Upsample(scale_factor=0.5, mode='nearest')
        )
        self.conv_mask_2=nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=3, padding=1),
                norm_layer(1),
                act,
                nn.Upsample(scale_factor=0.5, mode='nearest')
        )
        self.conv_mask_3=nn.Sequential(
                nn.Conv2d(3, 1, kernel_size=3, padding=1),
                norm_layer(1),
                act,
                nn.Upsample(scale_factor=0.5, mode='nearest')
        )
        self.conv_b_1=nn.Sequential(
            ResBlock(8,32,norm_layer=norm_layer,nonlinearity=act,learnable_shortcut=True,use_coord=True),
            nn.Upsample(scale_factor=0.5, mode='nearest')
            )
        self.conv_b_2=nn.Sequential(
            ResBlock(32,32,norm_layer=norm_layer,nonlinearity=act,learnable_shortcut=True,use_coord=True),
            nn.Upsample(scale_factor=0.5, mode='nearest')
            )
        self.conv_b_3=nn.Sequential(
            ResBlock(32,32,norm_layer=norm_layer,nonlinearity=act,learnable_shortcut=True,use_coord=True),
            nn.Upsample(scale_factor=0.5, mode='nearest')
            )
        self.conv_b_cut_1=ResBlock(32,32,norm_layer=norm_layer,nonlinearity=act,learnable_shortcut=True,use_coord=True)
        self.conv_b_cut_2=ResBlock(32,32,norm_layer=norm_layer,nonlinearity=act,learnable_shortcut=True,use_coord=True)
        self.up=nn.Upsample(scale_factor=2, mode='nearest')
        self.parout = nn.Sequential(
                nn.Conv2d(32, 8, kernel_size=3, padding=1),
                norm_layer(8),
                act
        )
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self, a1,b1,a2):
        bs,_,h,w=a1.shape
        a1=torch.sum(a1,dim=1).reshape(bs,1,h,w)
        a2=torch.sum(a2,dim=1).reshape(bs,1,h,w)
        a1_0=self.conv_share_1(a1)
        a2_0=self.conv_share_1(a2)
        bs,c,h,w=a1_0.shape
        mask_0=self.conv_mask_1(a2)
        matrix_0=torch.bmm(a1_0.view(bs,h*w,1),a2_0.view(bs,1,h*w))
        #matrix_0=self.softmax(matrix_0)
        b1_0=self.conv_b_1(b1)
        out_0=torch.bmm(b1_0.view(bs,-1,h*w),matrix_0.view(bs,h*w,h*w))
        out_0=(out_0.view(bs,-1,h,w))*mask_0
        out_0=self.up(out_0)

        a1_1=self.conv_share_2(torch.cat([a1,self.up(a1_0)],dim=1))
        a2_1=self.conv_share_2(torch.cat([a2,self.up(a2_0)],dim=1))
        mask_1=self.conv_mask_2(torch.cat([a2,self.up(mask_0)],dim=1))
        matrix_1=torch.bmm(a1_1.view(bs,h*w,1),a2_1.view(bs,1,h*w))
        #matrix_1=self.softmax(matrix_1)
        b1_1=self.conv_b_2(out_0)
        bias_1=self.conv_b_cut_1(b1_0)
        out_1=torch.bmm(b1_1.view(bs,-1,h*w),matrix_1.view(bs,h*w,h*w))
        out_1=(out_1.view(bs,-1,h,w)+bias_1)*mask_1
        out_1=self.up(out_1)
        '''
        a1_2=self.conv_share_3(torch.cat([a1,self.up(a1_0),self.up(a1_1)],dim=1))
        a2_2=self.conv_share_3(torch.cat([a2,self.up(a2_0),self.up(a2_1)],dim=1))
        mask_2=self.conv_mask_3(torch.cat([a2,self.up(mask_0),self.up(mask_1)],dim=1))
        matrix_2=torch.bmm(a1_2.view(bs,h*w,1),a2_2.view(bs,1,h*w))
        matrix_2=self.softmax(matrix_2)
        b1_2=self.conv_b_3(out_1)
        bias_2=self.conv_b_cut_2(b1_1)
        out_2=torch.bmm(b1_2.view(bs,-1,h*w),matrix_2.view(bs,h*w,h*w))
        out_2=(out_2.view(bs,-1,h,w)+bias_2)*mask_2
        out_2=self.up(out_2)
        '''
        out_final=self.parout(out_0)+self.parout(out_1)

        return out_final*1.0/2.0

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
        #self.parnet=ParseGenerator()
        #self.parnet = refine_ParsingNet()
        self.parnet = ParsingNet(8+18*2, 8)
        #self.parnet=PATNModel([8,36],8)

        self.Zencoder = Zencoder(3, ngf)

        self.imgenc = VggEncoder()
        #self.getMatrix = GetMatrix(ngf*4, 1)
        
        #self.phi = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)
        #self.theta = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)

        if self.use_shape==1:
            self.parenc = HardEncoder(8+18, ngf)
        else:
            self.parenc = HardEncoder(8+18+8+3, ngf)
        self.dec = BasicDecoder(3)
        self.banks=[]
        self.efb = EFB(ngf*4, 256,use_bank=self.use_bank,banks=self.banks)
        if self.use_bank==1:
            self.create_banks()
            self.efb = EFB(ngf*4, 256,use_bank=self.use_bank,banks=self.banks)
        if self.use_bank_org==1:
            self.create_banks_org()
            self.efb = EFB_org(ngf*4, 256,use_bank=self.use_bank,banks=self.banks)
        self.res = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
                
        self.res1 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)

        self.loss_fn = torch.nn.MSELoss()

        #self.forward_pass=my_pass
    def create_banks(self):
        self.bank0=bank(name='0')
        self.bank1=bank(name='1')
        self.bank2=bank(name='2')
        self.bank3=bank(name='3')
        self.bank4=bank(name='4')
        self.bank5=bank(name='5')
        self.bank6=bank(name='6')
        self.bank7=bank(name='7')
        for i in range(8):
            self.banks.append(self.__getattr__('bank' + str(i)))
    def create_banks_org(self):
        self.bank0=bank_org(name='0')
        self.bank1=bank_org(name='1')
        self.bank2=bank_org(name='2')
        self.bank3=bank_org(name='3')
        self.bank4=bank_org(name='4')
        self.bank5=bank_org(name='5')
        self.bank6=bank_org(name='6')
        self.bank7=bank_org(name='7')
        for i in range(8):
            self.banks.append(self.__getattr__('bank' + str(i)))
    def bank_org(self):
        loss=0
        for i in range(8):
            loss+=self.__getattr__('bank' + str(i)).loss_org()
        return loss
    def forward(self, img1, img2, pose1, pose2, par1, par2):
        #par_out=self.parnet([par1,torch.cat([pose1,pose2],1)])
        #par_out= self.parnet(pose1,par1,pose2)
        par_out=self.parnet(torch.cat((par1, pose1, pose2),1))
        if self.use_bank_org==1:
            loss_bank_org=self.bank_org()
        #1.generate the target parsing map
        if self.gt_par==0:
            par2 = par_out

        #2....
        #Fp
        if self.use_shape==1:
            parcode = self.parenc(torch.cat((pose2,par2), 1))
        else:
            parcode = self.parenc(torch.cat((par1,par2,pose2, img1), 1))#torch.Size([1, 256, 64, 64])
        #visualize_features(parcode,'shape')
        #Joint global and Per-region Normalization
        codes_vector, exist_vector, _ = self.Zencoder(img1, par1)
        self.codes_vector=codes_vector
        #print(codes_vector.shape,exist_vector.shape)#torch.Size([1, 9, 256]) torch.Size([1, 8])
        parcode = self.efb(parcode, par2, codes_vector, exist_vector)
        #visualize_features(parcode,'efb')
        #print(parcode.shape)#torch.Size([1, 256, 64, 64])
        parcode = self.res(parcode)#torch.Size([1, 256, 64, 64])
        #visualize_features(parcode,'efb_res')

        #Fn
        ## regularization to let transformed code and target image code in the same feature space
        #简单来说，让Fn和img2的feature相似。
        img2code = self.imgenc(img2)
        #img1code=self.imgenc(img1)
        #img2code1 = feature_normalize(img2code)
        loss_reg = F.mse_loss(img2code, parcode)

        #Spatial-aware Normalization
        if self.use_sn==1:
            #print('using the spatial normalization')
            img1code = self.imgenc(img1)
                
            parcode1 = feature_normalize(parcode)
            img1code1 = feature_normalize(img1code)
                
            if self.use_coordconv:
                #feature增加坐标的channel
                parcode1 = self.addcoords(parcode1)
                img1code1 = self.addcoords(img1code1)

            gamma, beta = self.getMatrix(img1code)
            batch_size, channel_size, h,w = gamma.shape
            att = self.computecorrespondence(parcode1, img1code1)
            #print(att.shape)
            gamma = gamma.view(batch_size, 1, -1)
            beta = beta.view(batch_size, 1, -1)
            imgamma = torch.bmm(gamma, att)
            imbeta = torch.bmm(beta, att)

            imgamma = imgamma.view(batch_size,1,h,w).contiguous()
            imbeta = imbeta.view(batch_size,1,h,w).contiguous()
            #在Fn之上直接进行空间归一化。
            parcode = parcode*(1+imgamma)+imbeta
        
        
        #decoder模块，似乎dec模块好实现bank的想法。 
        parcode = self.res1(parcode)
        img_gen = self.dec(parcode)
        if self.use_rec==1:
            if self.use_shape==1:
                parcode1 = self.parenc(torch.cat((pose1,par1), 1))
            else:
                parcode1 = self.parenc(torch.cat((par2,par1,pose1, img2), 1))#torch.Size([1, 256, 64, 64])
            #codes_vector, exist_vector, _ = self.Zencoder(img1, par1)
            parcode1 = self.efb(parcode1, par1, codes_vector, exist_vector)
            #print(parcode.shape)#torch.Size([1, 256, 64, 64])
            parcode1 = self.res(parcode1)

            img1code = self.imgenc(img1)
            loss_reg += F.mse_loss(img1code, parcode1)

            #Spatial-aware Normalization
        
            parcode1 = self.res1(parcode1)
            #parcode_orig=self.res1(parcode_orig)
            img_orig = self.dec(parcode1)
            if self.use_bank_org==1:
                #loss_bank_org=self.bank_org()
                return img_gen, loss_reg, par_out,img_orig,loss_bank_org
            return img_gen, loss_reg, par_out,img_orig
        #img_orig=self.dec(parcode_orig)
        if self.use_bank_org==1:
            #loss_bank_org=self.bank_org()
            return img_gen, loss_reg, par_out,loss_bank_org
        return img_gen, loss_reg, par_out
        
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

