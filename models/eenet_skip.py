import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d
import torchvision.models as models

from copy import deepcopy as copy

from pdb import set_trace as st

class SoftmaxLogProbability2D(torch.nn.Module):
    """Implements Softmax2D Layer
    """

    def __init__(self):
        super(SoftmaxLogProbability2D, self).__init__()

    def forward(self, x):
        orig_shape = x.data.shape
        seq_x = []
        for channel_ix in range(orig_shape[1]):
            softmax_ = F.softmax(x[:, channel_ix, :, :].contiguous()
                                 .view((orig_shape[0], orig_shape[2] * orig_shape[3])), dim=1)\
                .view((orig_shape[0], orig_shape[2], orig_shape[3]))
            seq_x.append(softmax_.log())

        x = torch.stack(seq_x, dim=1)
        return x

class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, output_padding=0):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=output_padding)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class BatchNormDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormDeconv2d, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.deconv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(FCN, self).__init__()
        self.deconv2d = UpsampleConvLayer(in_channels, out_channels, **kwargs)
        # self.conv2d = Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.deconv2d(x)
        # x = self.conv2d(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x, inplace=True)
        return x    

class EmbeddingNet(nn.Module):
    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

class EENet(EmbeddingNet):
    """Predicts Heatmap of where left and right finger tip of end effector are located

    """

    def __init__(self, img_height, img_width):
        #Implementation a mix of SE3 Nets and SE3 Pose Nets
        super(EENet, self).__init__()
        self.transform_input = True
        self.img_height = img_height
        self.img_width = img_width
        self.k = 2 # number of finger tips in scene
        # Encoder
        self.Conv1 = Conv2d(3, 8, bias=False, kernel_size=2, stride=1, padding=1)
        self.Pool1 = MaxPool2d(kernel_size=2)
        self.Conv2 = BatchNormConv2d(8, 16, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool2 = MaxPool2d(kernel_size=2)
        self.Conv3 = BatchNormConv2d(16, 32, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool3 = MaxPool2d(kernel_size=2)
        self.Conv4 = BatchNormConv2d(32, 64, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool4 = MaxPool2d(kernel_size=2)
        self.Conv5 = BatchNormConv2d(64, 128, bias=False, kernel_size=3, stride=1, padding=1)
        self.Pool5 = MaxPool2d(kernel_size=2)
        # Mask Decoder
        self.Deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)#, output_padding=0)
        self.Deconv2 = BatchNormDeconv2d(64, 32, kernel_size=3, stride=1, upsample=2)#, output_padding=0)
        self.Deconv3 = BatchNormDeconv2d(32, 16, kernel_size=3, stride=1, upsample=2)#, output_padding=0)
        self.Deconv4 = BatchNormDeconv2d(16, 8, kernel_size=3, stride=1, upsample=2)#, output_padding=0)
        self.Deconv5 = FCN(8, self.k, kernel_size=3, stride=1, upsample=2)#, output_padding=0)

        self.Softmax2D = SoftmaxLogProbability2D()
        self.Deconv5_l = UpsampleConvLayer(8, 1, kernel_size=3, stride=1, upsample=2)#, output_padding=0)
        self.Deconv5_r = UpsampleConvLayer(8, 1, kernel_size=3, stride=1, upsample=2)#, output_padding=0)

        self.LogSoftmax = nn.LogSoftmax(dim=1)
        # self.SpatialSoftmax = nn.Softmax2d()

    def encode_state(self, x):
                                               # x: 3 x 224 ** 2
        self.z1 = self.Pool1(self.Conv1(x))       # 8 x 112 ** 2
        self.z2 = self.Pool2(self.Conv2(self.z1)) # 16 x 56 ** 2
        self.z3 = self.Pool3(self.Conv3(self.z2)) # 32 x 28 ** 2
        self.z4 = self.Pool4(self.Conv4(self.z3)) # 64 x 14 ** 2
        self.z5 = self.Pool5(self.Conv5(self.z4)) # 128 x 7 ** 2
        z = self.z5
        return z

    def decode_mask(self, z):
                                                 # z: 128 x 7 ** 2 
        self.m1 = self.Deconv1(z)                 # 64 x 14 ** 2
        self.m2 = self.Deconv2(self.m1 + self.z4) # 32 x 28 ** 2
        self.m3 = self.Deconv3(self.m2 + self.z3) # 16 x 56 ** 2
        self.m4 = self.Deconv4(self.m3 + self.z2) # 8 x 112 ** 2
        # self.m5 = self.Deconv5(self.m4 + self.z1) # k x 224 ** 2

        self.m5_l = self.Deconv5_l(self.m4 + self.z1)
        self.m5_r = self.Deconv5_r(self.m4 + self.z1)

        # m = self.m5
        # import ipdb; ipdb.set_trace()

        x_l = self.m5_l
        x_r = self.m5_r

        # import ipdb; ipdb.set_trace()

        x_l  = F.interpolate(x_l, mode='nearest', size=(self.img_height, self.img_width))
        x_r  = F.interpolate(x_r, mode='nearest', size=(self.img_height, self.img_width))

        x_l = x_l.view(x_l.size()[0], -1)
        x_l = self.LogSoftmax(x_l)
        x_l = x_l.view(x_l.size()[0], self.img_height, self.img_width)

        x_r = x_r.view(x_r.size()[0], -1) # 2 channels for each fingertip
        x_r = self.LogSoftmax(x_r)
        x_r = x_r.view(x_r.size()[0], self.img_height, self.img_width)
        x = torch.stack((x_l, x_r), -1)
        # import ipdb; ipdb.set_trace()
        # x.transpose_(1, 3)

        return x
        
    def forward(self, x):
        if self.transform_input:
            if x.shape[1] == 4:
                x = x[:, :-1].clone()
            else:
                x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        enc_state = self.encode_state(x)
        mask = self.decode_mask(enc_state)
        out = mask
        # out = self.Softmax2D(mask)
        # return out.transpose(1, 2).transpose(2, 3)
        return out

def define_model(img_height, img_width):
    """Model loading wrapper
    
    Parameters
    ----------
    img_height : int
        image height
    img_width : int
        image width
    pretrained : bool, optional
        Use conv layers with pretrained weights of ImageNet (the default is True, which loads pretrained weights)
    
    Returns
    -------
    torch.nn.Module
        loads model
    """
    assert img_height == img_width
    return EENet(img_height, img_width)
