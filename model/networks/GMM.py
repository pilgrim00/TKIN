#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os

import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
class FeatureExtractionA(nn.Module):
    def __init__(self, input_nc,norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(FeatureExtractionA, self).__init__()
        downconv = nn.Conv2d(input_nc, 256, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(256)]
        model += [nn.Conv2d(256,256, kernel_size=4, stride=2, padding=1), nn.ReLU(True)]
        model += [norm_layer(256)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        
        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)
class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 256 else 256
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 256 else 256
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(out_ngf, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(256)]
        model += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        
        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()
        #将多通道尺寸图视为h*w个长度为c的向量，则该操作等同于对每个向量单位化。

    def forward(self, feature):
        epsilon = 1e-6
        norm=torch.sum(torch.pow(feature,2),1)
        norm = torch.pow(norm+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)# (b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)#(b,h*w,c)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)#(b,h*w,h*w)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)#(b,h*w,h,w)
        return correlation_tensor
    
class FeatureRegression(nn.Module):
    def __init__(self, input_nc=256,output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(1024, output_dim)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x

class AffineGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)
        
class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=256, use_regular_grid=True, grid_size=5, reg_factor=0):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3).cuda()
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3).cuda()
        # initialize regular grid for control points P_i
        
        if use_regular_grid:
            #采样点?
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X).cuda()
            P_Y = torch.FloatTensor(P_Y).cuda()
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)


            
    def forward(self, theta):
        grid=torch.cat((self.grid_X,self.grid_Y),3)
        warped_grid = self.apply_transformation(theta,grid)
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1).cuda()
        Z = torch.FloatTensor(3,3).fill_(0).cuda()       
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        batch_size = theta.size()[0]

        # split theta into point coordinates
        #self.N = grid_size*grid_size
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        #print(Q_X)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        #print(Q_Y)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)

class GMM(nn.Module):
    """ Geometric Matching Module
    """
    def __init__(self, grid_size=16,fine_height=256):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtractionA(256,norm_layer=nn.InstanceNorm2d) 
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3, norm_layer=nn.InstanceNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=256, output_dim=2*grid_size**2)
        self.gridGen = TpsGridGen(fine_height,fine_height,grid_size=grid_size)
        
    def forward(self, inputA,inputB):
        #A[1,256,64,64]
        #B[1,3,256,256]
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)
        #print(correlation.shape)

        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid
if __name__ == '__main__':
    import torch.nn.functional as F
    input1=torch.randn(1,22,256,256)
    input2=torch.randn(1,3,256,256)
    model=GMM(fine_height=128)
    grid=model(input1,input2)
    print(grid.shape)
    #For each output location output[n, :, h, w], the size-2 vector grid[n, h, w] specifies input pixel locations x and y
    #grid[n,h,w]对应输出位置[h,w]，它的数值[x,y]（所以是两个通道）指明了输入的位置来源,input[:,:,x,y]
    warped_input = F.grid_sample(input2, grid, padding_mode='border',align_corners=False)
    print(warped_input.shape)
    #print(torch.min(grid),torch.max(grid))
    #print(theta)



