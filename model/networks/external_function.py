import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
#from util import util
import numpy as np

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, for_dis=None):
        if self.type == 'hinge':
            if for_dis:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class VGGLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0],Vgg19=None):
        super(VGGLoss, self).__init__()
        if Vgg19:
            self.add_module('vgg', Vgg19)
        else:
            self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        for param in self.vgg.parameters():
            param.requires_grad_(False)
    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
    #@torch.no_grad()
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))


        return content_loss, style_loss
class VGGLoss_dtd(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self,Vgg19=None):
        super(VGGLoss_dtd, self).__init__()
        if Vgg19:
            self.add_module('vgg', Vgg19)
        else:
            self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        #self.weights = weights
        for param in self.vgg.parameters():
            param.requires_grad_(False)
    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
    #@torch.no_grad()
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss,不用信号量loss，取深层feature计算
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))


        return style_loss
class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class PerceptualCorrectness(nn.Module):
    r"""

    """

    def __init__(self, layer=['rel1_1','relu2_1','relu3_1','relu4_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer  
        self.eps=1e-8 
        self.resample = Resample2d(4, 1, sigma=2)

    def __call__(self, target, source, flow_list, used_layers, mask=None, use_bilinear_sampling=True):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)



        return loss

    def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape

        # maps = F.interpolate(maps, [h,w]).view(b,-1)
        flow = F.interpolate(flow, [h,w])

        target_all = target_vgg.view(b, c, -1)                      #[b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]


        source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps)
        target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
        try:
            correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
        except:
            print("An exception occurred")
            print(source_norm.shape)
            print(target_norm.shape)
        (correction_max,max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
        loss_map = torch.exp(-correction_sample/(correction_max+self.eps))
        if mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            mask=F.interpolate(mask, size=(target_vgg.size(2), target_vgg.size(3)))
            mask=mask.view(-1, target_vgg.size(2)*target_vgg.size(3))
            loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
            loss = torch.sum(mask * loss_map)/(torch.sum(mask)+self.eps)

        # print(correction_sample[0,2076:2082])
        # print(correction_max[0,2076:2082])
        # coor_x = [32,32]
        # coor = max_indices[0,32+32*64]
        # coor_y = [int(coor%64), int(coor/64)]
        # source = F.interpolate(self.source, [64,64])
        # target = F.interpolate(self.target, [64,64])
        # source_i = source[0]
        # target_i = target[0]

        # source_i = source_i.view(3, -1)
        # source_i[:,coor]=-1
        # source_i[0,coor]=1
        # source_i = source_i.view(3,64,64)
        # target_i[:,32,32]=-1
        # target_i[0,32,32]=1
        # lists = str(int(torch.rand(1)*100))
        # img_numpy = util.tensor2im(source_i.data)
        # util.save_image(img_numpy, 'source'+lists+'.png')
        # img_numpy = util.tensor2im(target_i.data)
        # util.save_image(img_numpy, 'target'+lists+'.png')
        return loss

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample



class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        model_vgg = models.vgg19().cuda()
        dict_=torch.load('/mnt/private_jiaxianchen/ft_local/vgg19-dcbb9e9d.pth')
        model_vgg.load_state_dict(dict_)
        features=model_vgg.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
class CX_loss(nn.Module):
    def __init__(self,band_width= 0.5,loss_type= 'cosine',Vgg19=None):
        super(CX_loss, self).__init__()
        if Vgg19:
            self.add_module('vgg', Vgg19)
        else:
            self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.band_width=band_width
        self.loss_type=loss_type
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss,不用信号量loss，取深层feature计算
        cx_loss = 0.0
        #cx_loss += self.contextual_loss(x_vgg['relu2_2'],y_vgg['relu2_2'])
        cx_loss += self.contextual_loss(x_vgg['relu3_2'],y_vgg['relu3_2'])
        cx_loss += self.contextual_loss(x_vgg['relu4_2'],y_vgg['relu4_2'])

        return cx_loss

    def contextual_loss(self,x,y):
        """
        Computes contextual loss between x and y.
        The most of this code is copied from
            https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
        Parameters
        ---
        x : torch.Tensor
            features of shape (N, C, H, W).
        y : torch.Tensor
            features of shape (N, C, H, W).
        band_width : float, optional
            a band-width parameter used to convert distance to similarity.
            in the paper, this is described as :math:`h`.
        loss_type : str, optional
            a loss type to measure the distance between features.
            Note: `l1` and `l2` frequently raises OOM.
        Returns
        ---
        cx_loss : torch.Tensor
            contextual loss between x and y (Eq (1) in the paper)
        """

        assert x.size() == y.size(), 'input tensor must have the same size.'

        N, C, H, W = x.size()

        if self.loss_type == 'cosine':
            dist_raw = self.compute_cosine_distance(x, y)
        elif self.loss_type == 'l1':
            dist_raw = self.compute_l1_distance(x, y)
        elif self.loss_type == 'l2':
            dist_raw = self.compute_l2_distance(x, y)

        dist_tilde = self.compute_relative_distance(dist_raw)
        cx = self.compute_cx(dist_tilde, self.band_width)
        cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)
        return cx_loss
    def compute_cx(self,dist_tilde, band_width):
        w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
        cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
        return cx


    def compute_relative_distance(self,dist_raw):
        dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
        dist_tilde = dist_raw / (dist_min + 1e-5)
        return dist_tilde


    def compute_cosine_distance(self,x, y):
        # mean shifting by channel-wise mean of `y`.
        y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x - y_mu
        y_centered = y - y_mu

        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)

        # channel-wise vectorization
        N, C, *_ = x.size()
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

        # consine similarity
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                            y_normalized)  # (N, H*W, H*W)

        # convert to distance
        dist = 1 - cosine_sim

        return dist

    # TODO: Considering avoiding OOM.
    def compute_l1_distance(self,x, y):
        N, C, H, W = x.size()
        x_vec = x.view(N, C, -1)
        y_vec = y.view(N, C, -1)

        dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
        dist = dist.sum(dim=1).abs()
        dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
        dist = dist.clamp(min=0.)

        return dist
    # TODO: Considering avoiding OOM.
    def compute_l2_distance(self,x, y):
        N, C, H, W = x.size()
        x_vec = x.view(N, C, -1)
        y_vec = y.view(N, C, -1)
        x_s = torch.sum(x_vec ** 2, dim=1)
        y_s = torch.sum(y_vec ** 2, dim=1)

        A = y_vec.transpose(1, 2) @ x_vec
        dist = y_s - 2 * A + x_s.transpose(0, 1)
        dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
        dist = dist.clamp(min=0.)

        return dist
class SW_loss(nn.Module):
    """
    Sliced-Wasserstein-Loss
    <<A Sliced Wasserstein Loss for Neural Texture Synthesis>>
    """

    def __init__(self,repeat=16):
        super(SW_loss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.flatten=nn.Flatten()
        self.repeat=repeat
    def calculate_sw(self,x,y):
        """ 
        Implementation of figure 2 in the paper 
        x,y must have the same size
        we do not calculate them within per channel
        """
        b,c,h,w=x.shape
        proj_dims=16
        x_ = torch.reshape(x, (b,-1))#b,c*w*h
        y_ = torch.reshape(y, (b,-1))
        losses=0
        for i in range(self.repeat):
            self.direction=torch.randn(c*h*w,proj_dims).cuda()
            #normalize
            self.direction=self.direction/torch.std(self.direction, dim=0, keepdim=True)
            # Project each pixel feature onto different directions
            sliced_x = torch.matmul(x_,self.direction)
            # Sort projections for each direction
            sorted_x,_ = torch.sort(sliced_x,dim=0)
            sliced_y = torch.matmul(y_,self.direction)
            # Sort projections for each direction
            sorted_y,_ = torch.sort(sliced_y,dim=0)
            losses+=self.criterion(self.flatten(sorted_x),self.flatten(sorted_y))
        return losses

    def __call__(self, x, y):
        # extracted pretrained features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        sw_loss = 0.0
        #sw_loss += self.calculate_sw(x_vgg['relu1_2'], y_vgg['relu1_2'])
        #sw_loss += self.calculate_sw(x_vgg['relu2_2'], y_vgg['relu2_2'])
        sw_loss += self.calculate_sw(x_vgg['relu3_2'], y_vgg['relu3_2'])
        sw_loss += self.calculate_sw(x_vgg['relu4_2'], y_vgg['relu4_2'])

        return sw_loss
if __name__=='__main__':
    loss=SW_loss()
    #loss_pec=PerceptualLoss()
    a=torch.randn(1,3,256,256).cuda()
    b=torch.randn(1,3,256,256).cuda()
    out=loss(a,b)
    print(out,out.grad)