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
class Puretransfer(BaseModel):
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
        self.opt=opt
        device = "cuda"
        self.device=device
        BaseModel.__init__(self, opt)   
        #self.visual_names = ['input_P1','input_P2', 'img_gen','parsav','input_SPL1','input_SPL2']
        #self.visual_names = ['input_P1','input_BP1','input_SPL1','input_P2','input_BP2','input_SPL2', 'img_gen']
        self.visual_names = ['input_SPL2','input_SPL1']
        self.model_names = ['G','D']
        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        # define the generator
        self.net_G = network.define_g(opt,filename='generator_puretrf',image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64,
                                use_spect=opt.use_spect_g, norm='instance', activation='LeakyReLU',use_sn=opt.sn,gt_par=opt.gt_par,
                                use_rec=opt.use_rec,ker_size=opt.ker_size,patch_dec=opt.patch_dec).to(device)

        # define the discriminator 
        self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d).to(device)

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
    def save4trans(self,input):
        self.input_P3=input['P1'].cuda()
        self.input_SPL3=input['SPL1'].cuda()
        self.input_BP3=input['BP1'].cuda()
    def push_output(self):
        #print(img1.shape)
        self.img_gen,self.img_trf= self.net_G(self.input_P1, self.input_P2,
                                                    self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2,
                                                    self.input_P3,self.input_SPL3,self.input_BP3)
        #out=torch.cat([self.input_P1,self.input_P2,self.img_gen.detach(),self.input_P3,self.img_trf.detach()],dim=2)[0]
        out=torch.cat([self.input_P1,self.input_P3,self.img_trf.detach()],dim=2)[0]
        return out
    def set_input(self, input):
        # move to GPU and change data types
        #self.input = input
        input_P1, input_BP1, input_SPL1 = input['P1'], input['BP1'], input['SPL1']
        input_P2, input_BP2, input_SPL2, label_P2 = input['P2'], input['BP2'], input['SPL2'], input['label_P2']
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
        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_' + input['P2_path'][i])
    def test(self):
        """Forward function used in test time"""  
        self.img_gen,self.img_trf= self.net_G(self.input_P1, self.input_P2,
                                                    self.input_BP1, self.input_BP2, self.input_SPL1, self.input_SPL2,
                                                    self.input_P3,self.input_SPL3,self.input_BP3)

        self.save_results(self.img_gen, data_name='vis')
        if self.opt.save_input or self.opt.phase == 'val':
            self.save_results(self.input_P1, data_name='ref')
            self.save_results(self.input_P2, data_name='gt')
            result = torch.cat([self.input_P1, self.img_gen, self.input_P2], 3)
            self.save_results(result, data_name='all')
                       
                
