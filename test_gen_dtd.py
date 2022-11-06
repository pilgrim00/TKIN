from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import numpy as np
from model.networks.base_function import *
import torch.nn as nn
from model.networks import base_function, external_function
import itertools
class dataset_texture(Dataset):
    def __init__(self,transform, resolution=256,path='./DTD/images'):
        self.image_path=path
        self.res=resolution
        self.index=0
        self.transform=transform
        self.get_image()
        print('Images: {}'.format(len(self.images_path)))
    def get_image(self):
        self.images_path=[]#2354
        #for imgs_path in sorted(os.listdir(self.image_path)):
        for catg in sorted(os.listdir(self.image_path)):
            for img in sorted(os.listdir(self.image_path+'/'+catg)):
                if 'jpg' in img:
                    img_path=self.image_path+'/'+catg+'/'+img
                    self.images_path.append(img_path)
    def __getitem__(self, index):
        #建议将比较消耗的操作放在getitem操作，如I/O
        img=self.images_path[index]
        image=Image.open(img)#L = R * 299/1000 + G * 587/1000 + B * 114/1000,上次搞错了，打开了原图然后.convert('L')，打开灰度图不需要这样操作
        #h,w=image.size
        image=image.crop((0,0,256,256))
        return self.transform(image)
    def __len__(self):
        return len(self.images_path)
class Generator(nn.Module):
    def __init__(self, image_nc=3,output_nc=3, ngf=64, norm='instance', activation='LeakyReLU', use_spect=True, 
                use_coord=False,use_sn=None,gt_par=None,use_rec=None,use_shape=None,use_bank=None,use_bank_org=None):
        super(Generator, self).__init__()
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
        self.block0 = EncoderBlock(3, ngf, norm_layer, nonlinearity, use_spect)
        self.block1 = EncoderBlock(ngf, ngf*2, norm_layer, nonlinearity, use_spect)
        self.block2 = EncoderBlock(ngf*2, ngf*2, norm_layer, nonlinearity, use_spect)
        self.block3 = EncoderBlock(ngf*2, ngf*2, norm_layer, nonlinearity, use_spect)
        self.block4 = ResBlockDecoder(ngf*2, ngf*2,ngf*2, norm_layer, nonlinearity, use_spect)
        self.block5 = ResBlockDecoder(ngf*2, ngf*2,ngf*2, norm_layer, nonlinearity, use_spect)
        self.get_code = nn.Sequential(nn.Conv2d(ngf*2, 1, kernel_size=1, padding=0), nn.Tanh())
        self.fc_m=nn.Linear(4096,256)
        self.fc_s=nn.Linear(4096,256) 
        self.imgenc = VggEncoder()

        self.dec = BasicDecoder(3)
        self.res_dtd = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
        self.res3 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
        self.dec_dtd = BasicDecoder(3)
    def kl_div(self,mu, logvar):
        kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KL divergence
        return kl_div
    def reparametrize(self, mu, logvar):
        #std = logvar.exp()
        std=torch.exp(logvar/2)
        eps=torch.randn_like(std)
        out=eps*std+mu
        #eps = torch.cuda.FloatTensor(std.size()).normal_()
        return out
    def forward(self,input_dtd):
        bs,_,_,_=input_dtd.shape
        out_dtd = self.block0(input_dtd)
        out_dtd = self.block3(self.block2(self.block1(out_dtd)))
        out_dtd = self.block5(self.block4(out_dtd))
        codes_dtd = self.get_code(out_dtd)#torch.Size([1, 256, 64, 64])
        #print(codes_dtd.shape)
        dtd_mean=self.fc_m(codes_dtd.reshape(bs,64*64))
        dtd_std=self.fc_s(codes_dtd.reshape(bs,64*64))
        z_dtd=self.reparametrize(dtd_mean,dtd_std)
        #z_dtd=dtd_mean
        kl_loss=self.kl_div(dtd_mean,dtd_std)
        dtd_expand=z_dtd.reshape(bs,256,1).expand(bs,256,64*64).reshape(bs,256,64,64)
        dtd_feat=self.res3(dtd_expand)
        dtd_code = self.imgenc(input_dtd)
        loss_reg = F.mse_loss(dtd_feat, dtd_code)*0.0
        dtd_feat=self.res_dtd(dtd_feat)
        dtd_img=self.dec_dtd(dtd_feat)
        return loss_reg,dtd_img,kl_loss
class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    :param use_attn: use short+long attention or not
    """
    def __init__(self, input_nc=3, ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}
        sequence = [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]

        mult = 1
        for i in range(1, layers):
            mult_prev = mult
            mult = min(2 ** i, img_f // ndf)
            sequence +=[
                coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                nonlinearity,
            ]

        mult_prev = mult
        mult = min(2 ** i, img_f // ndf)
        kwargs = {'kernel_size': 4, 'stride': 1, 'padding': 1, 'bias': False}
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out            


        
if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils import data
    GANloss = external_function.AdversarialLoss('lsgan').cuda()
    L1loss = torch.nn.L1Loss()
    Vggloss = external_function.VGGLoss().cuda()
    def backward_D_basic(netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss)*0.5
        # gradient penalty for wgan-gp
        #D_loss.backward()
        return D_loss
    def backward_D(net_D,input_dtd,out_dtd):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(net_D)
        #print(self.input_P2.shape, self.img_gen.shape)
        loss_dis_img_gen =backward_D_basic(net_D,input_dtd,out_dtd)
        loss_dis_img_gen.backward()
    def backward_G(net_D,input_dtd,out_dtd,loss_reg,loss_kl):
        """Calculate training loss for the generator"""
        loss_reg=loss_reg
        loss_kl=loss_kl*100.0
        # Calculate l1 loss 
        loss_app_gen = L1loss(input_dtd,out_dtd)*0.0
        
        # Calculate GAN loss
        base_function._freeze(net_D)
        D_fake = net_D(out_dtd)
        loss_ad_gen =GANloss(D_fake, True, False)*2.0
        # Calculate perceptual loss
        loss_content_gen, loss_style_gen =Vggloss(input_dtd,out_dtd) 
        total_loss = loss_app_gen+loss_ad_gen+loss_content_gen*0+loss_style_gen*0+loss_reg+loss_kl
        total_loss.backward()
        return loss_app_gen,loss_ad_gen,loss_content_gen*0.5,loss_style_gen*100,loss_reg,loss_kl
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),#左右翻转，切忌不要使用上下翻转
            #transforms.Pad(padding=(100, 0),fill=(0,0,0), padding_mode='constant'),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
    dataset=dataset_texture(transform=transform)
    dtd_loader = DataLoader(
            dataset,
            batch_size=4,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
            shuffle=True,
        )
    net_G=Generator().cuda()
    net_D=PatchDiscriminator().cuda()
    lr=1e-3
    optimizer_G = torch.optim.Adam(itertools.chain(
                                   filter(lambda p: p.requires_grad,net_G.parameters())),
                                    lr=lr, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, net_D.parameters())),
                                lr=0.1*lr, betas=(0.9, 0.999))
    GANloss = external_function.AdversarialLoss('lsgan').cuda()
    if not os.path.exists('./dtd_imgs'):
        os.mkdir('./dtd_imgs')
    def save_img(img,epoch,i,gt=False):
        img=img[0].cpu().permute(1,2,0).detach().numpy()
        img=(img+1.0)/2.0
        img=img[:,:,[2,1,0]]*255.0
        if gt==True:
            cv2.imwrite('./dtd_imgs/'+str(epoch)+'_'+str(i)+'gt.jpg',img)
        else:
            cv2.imwrite('./dtd_imgs/'+str(epoch)+'_'+str(i)+'gen.jpg',img)
    epochs=10
    for epoch in range(epochs):
        for i,img in enumerate(dtd_loader):
            #print(i)
            img=img.cuda()
            loss_reg,out_img,loss_kl=net_G(img)
            optimizer_D.zero_grad()
            backward_D(net_D,img,out_img)
            optimizer_D.step()
            optimizer_G.zero_grad()
            loss_app_gen,loss_ad_gen,loss_content_gen,loss_style_gen,loss_reg,loss_kl=backward_G(net_D,img,out_img,loss_reg,loss_kl)
            optimizer_G.step()
            if i%100==0:
                print(epoch,':',i,':loss_app_gen,loss_ad_gen,loss_content_gen,loss_style_gen,loss_reg,loss_kl\n')
                print(loss_app_gen.item(),loss_ad_gen.item(),loss_content_gen.item(),loss_style_gen.item(),loss_reg.item(),loss_kl.item())
            if i%100==0:
                save_img(img,epoch,i,gt=True)
                save_img(out_img,epoch,i,gt=False)



    