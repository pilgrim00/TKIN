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
from model.networks.parsing_metrics import IoU
import torch.nn.functional as F
from model.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
def visualize_features(input_feats,name):

    def norm(x):
        return (x-np.min(x))/(np.max(x)-np.min(x))
    input_feats=input_feats[0].cpu().numpy()
    sav_path='/apdcephfs/private_jiaxianchen/PISE/result/'+name+'/'
    if not os.path.exists(sav_path):
        os.mkdir(sav_path)
    for i in range(len(input_feats)):
        #灰度化图片
        img=(input_feats[i]+1.0)/2.0
        img[img>1]=1
        img[img<0]=0
        img=img*255.0
        img=img.astype(np.uint8)
        cv2.imwrite(sav_path+str(i)+'feature.png',img)
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        #self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
        self.nll_loss=F.nll_loss
        self.softmax = nn.LogSoftmax(dim=1)
        self.ignore_index=ignore_index
    def forward(self, inputs, targets,weight):
        out=self.softmax(inputs)
        out=self.nll_loss(out,targets,weight=weight,reduction='mean',ignore_index=self.ignore_index)
        return out
class limbs(BaseModel):
    def name(self):
        #11_23:
        #将整体的关节连接图分为不同通道的关节图，表示连接关系，而不是孤立的点。
        #生成parsingmap--1.pose1输入；2.pose1,pose2,par2输入
        #网络用自己设计的
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
        parser.add_argument('--lambda_style', type=float, default=200.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=30.0, help='weight for the affine regularization loss')

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        self.num=0
        self.opt=opt
        device = "cuda"
        self.device=device
        BaseModel.__init__(self, opt)
        self.loss_names = [ 'app_gen','content_gen', 'style_gen', 'reg_gen',
                           'ad_gen', 'dis_img_gen','par','par1']     
        #self.visual_names = ['input_P1','input_P2', 'img_gen','parsav','input_SPL1','input_SPL2']
        #self.visual_names = ['input_P1','input_BP1','input_SPL1','input_P2','input_BP2','input_SPL2']
        self.visual_names = ['input_P1','input_P2', 'img_gen','par2_','input_SPL2']
        self.model_names = ['G','D','D_par']
        self.par_metric=IoU(num_classes=6)
        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        # define the generator
        net_G_network='generator_limbs'
        if opt.limb_one==1:
            #not using the ways of concating...
            net_G_network='generator_limbs_one'
        self.net_G = network.define_g(opt,filename=net_G_network,image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64,
                                use_spect=opt.use_spect_g, norm='instance', activation='LeakyReLU',use_sn=opt.sn,gt_par=opt.gt_par,
                                use_rec=opt.use_rec,ker_size=opt.ker_size,patch_dec=opt.patch_dec,use_bank=opt.use_bank,
                                parnet_pise=opt.parnet_pise).to(device)

        # define the discriminator 
        self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d).to(device)
        self.net_D_par = network.define_d(opt,input_nc=8,name='patch',ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d).to(device)
        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(device)
            #此处可以改为L2loss
            self.L2loss = torch.nn.MSELoss()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss().to(device)
            self.parLoss = CrossEntropyLoss2d()
            self.weight=torch.tensor([1.0,17.8, 15.9, 10.7, 17.7, 15.5, 15.3, 17.2]).to(device)
            # define the optimizer
            #这一点设置上，就隔离开了feature上的vgg，loss。
            if self.opt.optim=='Adam':
                self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.9, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                filter(lambda p: p.requires_grad, self.net_D_par.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.9, 0.999))#d_lr=0.1*g_l ,d不能过早收敛，x=G(z)会趋向于D(x)值高的区域移动.
            if self.opt.optim=='SGD':
                self.optimizer_G = torch.optim.SGD(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr)
                self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                filter(lambda p: p.requires_grad, self.net_D_par.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.9, 0.999))#d_lr=0.1*g_l ,d不能过早收敛，x=G(z)会趋向于D(x)值高的区域移动.
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        if self.opt.dist==0 or self.opt.local_rank==0:
            self.setup(opt)
        #niter让网络自动去统计当前迭代次数
        self.opt.iter_count = util.get_iterfrom_dir(self.save_dir)
        self.schedulers = [base_function.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
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
            self.net_D_par = nn.parallel.DistributedDataParallel(
                self.net_D_par,
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
        


    def set_input(self, input):
        self.num+=1
        print(input['BP1'].shape)
        BP1=torch.sum(input['BP1'],dim=1).reshape(1,1,256,256)
        visualize_features(BP1,'2_26keypoints'+str(self.num))
        if self.num>30:
            exit()
        '''
        # move to GPU and change data types
        #self.input = input
        input_P1, input_BP1, input_SPL1 = input['P1'], input['LP1'], input['SPL1']
        input_P2, input_BP2, input_SPL2, label_P2,label_P1 = input['P2'], input['LP2'], input['SPL2'], input['label_P2'],input['label_P1']
        print( input['LP1'].shape, input['BP1'].shape)
        visualize_features(input['LP1'],'2_26LP'+str(self.num))
        visualize_features(input['BP1'],'2_26BP'+str(self.num))
        if len(self.gpu_ids) > 0 and self.opt.dist==0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], async=True)#相比async,non_blocking should be used instead.
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], async=True)
            #print(self.input_BP1.shape,torch.min(self.input_BP1),torch.max(self.input_BP1))
            #(bs,18,256,256) (0.0-1.0)十八个关键点。
            self.input_SPL1 = input_SPL1.cuda(self.gpu_ids[0], async=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], async=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], async=True)  
            self.input_SPL2 = input_SPL2.cuda(self.gpu_ids[0], async=True)  
            self.label_P2 = label_P2.cuda(self.gpu_ids[0], async=True)
            self.label_P1 = label_P1.cuda(self.gpu_ids[0], async=True)
        if self.opt.dist==1:
            self.input_P1=input_P1.to(self.device)
            self.input_BP1=input_BP1.to(self.device)
            self.input_SPL1=input_SPL1.to(self.device)   
            self.input_P2=input_P2.to(self.device)
            self.input_BP2=input_BP2.to(self.device)
            self.input_SPL2=input_SPL2.to(self.device)
            self.label_P2=label_P2.to(self.device)
            self.label_P1=label_P1.to(self.device)    
        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_' + input['P2_path'][i])
        '''

    def test(self):
        
        """Forward function used in test time"""
        #self.img_gen, self.loss_reg,self.par2_,self.par1_= self.net_G(self.input_P1, self.input_P2,
        #                                                        self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2)
        #pred=torch.cat([self.par2_[:,:1,:,:],self.par2_[:,2:4,:,:],self.par2_[:,5:,:,:]],1)
        #gt=torch.cat([self.input_SPL2[:,:1,:,:],self.input_SPL2[:,2:4,:,:],self.input_SPL2[:,5:,:,:]],1)
        #self.par_metric.add(pred,gt)
        #exit()
        #self.save_results(torch.cat([self.par2_,self.input_SPL2], 3), data_name='par_out')
        #class_iou,miou=self.par_metric.value()
        #print(class_iou,miou)
        #self.save_results(self.img_gen, data_name='vis')
        #if self.opt.save_input or self.opt.phase == 'val':
            #self.save_results(self.input_P1, data_name='ref')
            #self.save_results(self.input_P2, data_name='gt')
            #result = torch.cat([self.input_P1, self.img_gen, self.input_P2], 3)
            #self.save_results(result, data_name='all')
                       
    def val(self):
        """Forward function used in test time"""
        self.img_gen, self.loss_reg,self.par2_,self.par1_= self.net_G(self.input_P1, self.input_P2,
                                                                self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2)


        result = torch.cat([self.input_P1, self.input_BP2,self.img_gen], 3)
        self.save_results(result, data_name='all')   

    def forward(self):
        """Run forward processing to get the inputs"""
        self.img_gen, self.loss_reg,self.par2_,self.par1_= self.net_G(self.input_P1, self.input_P2,
                                                                self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2)
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

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        base_function._unfreeze(self.net_D_par)
        #print(self.input_P2.shape, self.img_gen.shape)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)
        self.loss_dis_img_gen+= self.backward_D_basic(self.net_D_par, self.input_SPL2,self.par2_)
        self.loss_dis_img_gen+= self.backward_D_basic(self.net_D_par, self.input_SPL1,self.par1_)
        if self.opt.use_rec:
            self.loss_dis_img_gen+= self.backward_D_basic(self.net_D, self.input_P1, self.img_orig)   

    def backward_G(self):
        """Calculate training loss for the generator"""
        # Calculate regularzation loss to make transformed feature and target image feature in the same latent space
        self.loss_reg_gen = self.loss_reg * self.opt.lambda_regularization

        # Calculate l1 loss 
        loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec
        if self.opt.use_rec==1:
            loss_app_orig=self.L1loss(self.img_orig, self.input_P1)
            self.loss_app_gen += loss_app_orig * self.opt.lambda_rec
        # parsing loss
        label_P2 = self.label_P2.squeeze(1).long()
        label_P1 = self.label_P1.squeeze(1).long()
        if self.opt.par_weight==1:
            self.loss_par = self.parLoss(self.par2_,label_P2,self.weight)# * 20,entropy loss
            self.loss_par += self.parLoss(self.par1_,label_P1,self.weight)
        else:
            self.loss_par = self.parLoss(self.par2_,label_P2,None)# * 20,entropy loss
            self.loss_par += self.parLoss(self.par1_,label_P1,None)

        self.loss_par1 = self.L1loss(self.par2_, self.input_SPL2)*50+self.L1loss(self.par1_, self.input_SPL1)*50
     
        # Calculate GAN loss
        base_function._freeze(self.net_D)
        D_fake = self.net_D(self.img_gen)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        if self.opt.use_rec==1:
            D_fake_1=self.net_D(self.img_orig)
            self.loss_ad_gen += self.GANloss(D_fake_1, True, False) * self.opt.lambda_g
        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2) 
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content
        if self.opt.use_rec:
            loss_content_orig, loss_style_orig = self.Vggloss(self.img_orig, self.input_P1) 
            self.loss_style_gen += loss_style_orig*self.opt.lambda_style
            self.loss_content_gen += loss_content_orig*self.opt.lambda_content       
        total_loss = 0
        for name in self.loss_names:
            if name != 'dis_img_gen':
                #print(getattr(self, "loss_" + name))
                #temp=getattr(self, "loss_" + name)
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()
        

    def optimize_parameters(self):
        """update network weights"""
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
