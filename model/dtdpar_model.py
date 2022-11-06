import torch
import torch.nn as nn
from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from model.networks.generator import *
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os

import torch.nn.functional as F
from model.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
class dtdpar(BaseModel):
    def name(self):
        return "parsing and inpaint network"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        #parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=200.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=30.0, help='weight for the affine regularization loss')

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.add_argument('--semantic_nc', type=int, default=8, help='')
        parser.add_argument('--style_length', type=int, default=256, help='')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        self.opt=opt
        device='cuda'
        if opt.dist==1:
            device = "cuda:"+str(opt.local_rank)
        self.device=device
        BaseModel.__init__(self, opt)
        self.loss_names = [ 'app_gen','content_gen', 'style_gen','reg_gen','kl','reg_code',
                           'ad_gen', 'dis_img_gen','dis_img_joint','ad_joint','ad_dtd',
                           'CX','CX_dtd']   
      
        self.visual_names = ['input_P1','input_P2', 'img_gen','out_dtd','input_dtd']
        self.model_names = ['G','D','D_dtd',"D_joint"]
        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        # define the generator
        self.net_G = network.define_g(opt,filename='generator_dtd_par',image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64,
                                use_spect=opt.use_spect_g, norm='instance', activation='LeakyReLU',use_sn=opt.sn,gt_par=opt.gt_par,
                                use_rec=opt.use_rec,use_shape=opt.use_shape,use_bank=opt.use_bank,use_bank_org=opt.use_bank_org).to(device)

        # define the discriminator 
        self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d).to(device)
        self.net_D_dtd = network.define_d(opt,name='patch',ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d).to(device)
        self.net_D_joint = network.define_d(opt,name='patch',input_nc=6,ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d).to(device)
        '''
        trained_list = ['parnet']
        for k,v in self.net_G.named_parameters():
            flag = False
            for i in trained_list:
                if i in k:
                    flag = True
            if flag:
                #v.requires_grad = False
                print(k)
        '''       
        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(device)
            #此处可以改为L2loss
            self.VGG19=external_function.VGG19().to(device)
            self.L2loss = torch.nn.MSELoss()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss(Vgg19=self.VGG19)
            #self.Vggloss_dtd=external_function.VGGLoss_dtd(Vgg19=self.VGG19)
            self.CX_loss=external_function.CX_loss(Vgg19=self.VGG19)

            # define the optimizer
            #这一点设置上，就隔离开了feature上的vgg，loss。
            if self.opt.optim=='Adam':
                self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.9, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                filter(lambda p: p.requires_grad, self.net_D_dtd.parameters()),
                                filter(lambda p: p.requires_grad, self.net_D_joint.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.9, 0.999))#d_lr=0.1*g_l ,d不能过早收敛，x=G(z)会趋向于D(x)值高的区域移动.
                '''
                self.optimizer_D_dtd = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D_dtd.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.9, 0.999))
                self.optimizer_D_joint = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D_joint.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.9, 0.999))
                '''
            if self.opt.optim=='SGD':
                self.optimizer_G = torch.optim.SGD(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr)
                self.optimizer_D = torch.optim.SGD(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                                lr=opt.lr*opt.ratio_g2d)#d_lr=0.1*g_l
                self.optimizer_D_dtd = torch.optim.SGD(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D_dtd.parameters())),
                                lr=opt.lr*opt.ratio_g2d)
                self.optimizer_D_joint = torch.optim.SGD(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D_dtd.parameters())),
                                lr=opt.lr*opt.ratio_g2d)       
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #self.optimizers.append(self.optimizer_D_dtd)
            #self.optimizers.append(self.optimizer_D_joint)
        if self.opt.dist==0 or self.opt.local_rank==0:
            self.setup(opt)
        if opt.dist==1:
            self.net_G = nn.parallel.DistributedDataParallel(
                self.net_G,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True
            )

            self.net_D = nn.parallel.DistributedDataParallel(
                self.net_D,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True
            )
            self.net_D_dtd = nn.parallel.DistributedDataParallel(
                self.net_D,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True
            )
            self.net_D_joint = nn.parallel.DistributedDataParallel(
                self.net_D_joint,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True
            )
        if opt.dp==1:
            gpus=int(os.environ['HOST_GPU_NUM'])
            device_ids=[i for i in range(gpus)]
            #output_device
            self.net_G=nn.DataParallel(self.net_G, device_ids)
            self.net_D=nn.DataParallel(self.net_D, device_ids)
            self.net_D_dtd=nn.DataParallel(self.net_D_dtd, device_ids)
            self.net_D_joint=nn.DataParallel(self.net_D_joint, device_ids)
        # load the pre-trained model and schedulers
        #self.setup(opt)
        


    def set_input(self, input):
        # move to GPU and change data types
        #self.input = input
        input_P1, input_BP1, input_SPL1 = input['P1'], input['BP1'], input['SPL1']
        input_P2, input_BP2, input_SPL2= input['P2'], input['BP2'], input['SPL2']
        input_dtd=input['dtd_img']
        if len(self.gpu_ids) > 0 and self.opt.dist==0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], async=True)#相比async,non_blocking should be used instead.
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], async=True)
            #print(self.input_BP1.shape,torch.min(self.input_BP1),torch.max(self.input_BP1))
            #(bs,18,256,256) (0.0-1.0)十八个关键点。
            self.input_SPL1 = input_SPL1.cuda(self.gpu_ids[0], async=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], async=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], async=True)  
            self.input_SPL2 = input_SPL2.cuda(self.gpu_ids[0], async=True)
            self.input_dtd = input_dtd.cuda(self.gpu_ids[0], async=True)
            #选用上衣的mask
            self.input_dtd=self.input_dtd*self.input_SPL2[:,3,:,:].reshape(self.opt.batchSize,1,256,256)
            self.input_dtd=self.input_dtd.contiguous()  
        if self.opt.dist==1:
            self.input_P1=input_P1.to(self.device)
            self.input_BP1=input_BP1.to(self.device)
            self.input_SPL1=input_SPL1.to(self.device)   
            self.input_P2=input_P2.to(self.device)
            self.input_BP2=input_BP2.to(self.device)
            self.input_SPL2=input_SPL2.to(self.device)
            self.input_dtd=input_dtd.to(self.device)
            self.input_dtd=self.input_dtd*self.input_SPL2[:,3,:,:].reshape(self.opt.batchSize,1,256,256)
         
            self.input_dtd=self.input_dtd.contiguous()    
        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_' + input['P2_path'][i])

    def test(self):
        """Forward function used in test time"""
        self.img_gen, self.loss_reg,self.img_orig,self.out_dtd,self.kl_loss,self.code_reg= self.net_G(self.input_P1, self.input_P2,
                                                            self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2,self.input_dtd)
            
        ## test flow ##

        self.save_results(self.img_gen, data_name='vis')
        if self.opt.save_input or self.opt.phase == 'val':
            self.save_results(self.input_P1, data_name='ref')
            self.save_results(self.input_P2, data_name='gt')
            result = torch.cat([self.input_P1, self.img_gen, self.input_P2], 3)
            self.save_results(result, data_name='all')
                       
                

    def forward(self):
        """Run forward processing to get the inputs"""
        self.img_gen, self.loss_reg,self.img_orig,self.out_dtd,self.code_reg,self.loss_kl= self.net_G(self.input_P1, self.input_P2,
                                                            self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2,self.input_dtd)
          

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            #目前默认是lsgan
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        #D_loss.backward()

        return D_loss
    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        base_function._unfreeze(self.net_D_dtd)
        base_function._unfreeze(self.net_D_joint)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)
        self.loss_dis_img_gen+= self.backward_D_basic(self.net_D, self.input_P1, self.img_orig)
        self.loss_dis_img_gen+=self.backward_D_basic(self.net_D_dtd, self.input_dtd, self.out_dtd)
        real_joint=torch.cat([self.input_P1,self.input_P2],dim=1)
        self.fake_joint=torch.cat([self.img_orig,self.img_gen],dim=1)
        self.loss_dis_img_joint=self.backward_D_basic(self.net_D_joint, real_joint,self.fake_joint)
        total_loss=0
        for name in self.loss_names:
            if 'dis'  in name:
                #print(getattr(self, "loss_" + name))
                #temp=getattr(self, "loss_" + name)
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()

    def backward_G(self):
        """Calculate training loss for the generator"""
        #regularazation
        self.loss_reg_gen = self.loss_reg*self.opt.lambda_regularization
        self.loss_kl=self.loss_kl*10
        self.loss_reg_code=self.code_reg*10000
        # Calculate l1 loss 
        loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec
        loss_app_orig=self.L1loss(self.img_orig, self.input_P1)
        self.loss_app_gen += loss_app_orig * self.opt.lambda_rec
        self.loss_app_gen+=self.L1loss(self.out_dtd,self.input_dtd)*0.2
        # Calculate GAN loss
        base_function._freeze(self.net_D)
        base_function._freeze(self.net_D_dtd)
        base_function._freeze(self.net_D_joint)
        #1.针对单张人体图片的D
        D_fake = self.net_D(self.img_gen)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        D_fake_1=self.net_D(self.img_orig)
        self.loss_ad_gen += self.GANloss(D_fake_1, True, False) * self.opt.lambda_g
        #2.针对整体两张人体图片的D
        D_joint_fake=self.net_D_joint(self.fake_joint)
        self.loss_ad_joint=self.GANloss(D_joint_fake, True, False) * self.opt.lambda_g
        #3.针对单张DTD图片的D
        D_dtd_fake=self.net_D_dtd(self.out_dtd)
        self.loss_ad_dtd=self.GANloss(D_dtd_fake, True, False) * self.opt.lambda_g
        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2) 
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content
        loss_content_orig, loss_style_orig = self.Vggloss(self.img_orig, self.input_P1) 
        self.loss_style_gen += loss_style_orig*self.opt.lambda_style
        self.loss_content_gen += loss_content_orig*self.opt.lambda_content    
        #the texture_loss for the DTD_img
        #self.loss_style_dtd=self.Vggloss_dtd(self.out_dtd,self.input_dtd)*self.opt.lambda_style
        loss_content_gen_dtd, loss_style_gen_dtd = self.Vggloss(self.out_dtd,self.input_dtd)
        self.loss_style_gen+=loss_style_gen_dtd*self.opt.lambda_style
        self.loss_content_gen+=loss_content_gen_dtd*self.opt.lambda_content
        #the CX_loss for the person_img and DTD_img
        self.loss_CX=self.CX_loss(self.img_orig, self.input_P1)*0.1
        self.loss_CX+=self.CX_loss(self.img_gen, self.input_P2)*0.1
        self.loss_CX_dtd=self.CX_loss(self.out_dtd,self.input_dtd)*0.1
        total_loss = 0
        for name in self.loss_names:
            if 'dis' not in name:
                #print(getattr(self, "loss_" + name))
                #temp=getattr(self, "loss_" + name)
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()
        
    def optimize_parameters(self):
        """update network weights"""
        self.forward()
        self.optimizer_D.zero_grad()
        #self.optimizer_D_dtd.zero_grad()
        #self.optimizer_D_joint.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        #self.optimizer_D_dtd.step()
        #self.optimizer_D_joint.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
