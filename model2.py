## 导入本章所需要的模块
import numpy as np

#config InlineBackend.figure_format = 'retina'

#matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import glob
from time import time
import os
from skimage.io import imread
import copy
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision.models import resnet18
from torchvision import transforms
from torchsummary import summary
#from data_generator import
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#判断用cuda还是cpu
high=448
width=480
device ='cpu'#由于GPU显存不够这里用cpu
model_resnet = resnet18(pretrained=True)#pretrained=True是指可以使用预训练模型
## 不使用vgg19网络中的后面的AdaptiveAvgPool2d和Linear层
## vgg19的features网络通过5个MaxPool将图像尺寸缩小了32倍
## 图像尺寸缩小后分别在：MaxPool2d-5(缩小2倍) ,MaxPool2d-10 （缩小4倍）,MaxPool2d-19（缩小8倍）,
## MaxPool2d-28（缩小16倍）,MaxPool2d-37（缩小32倍）
# 定义FCN语义分割网络
colormap =[[0, 0, 0], [0, 0, 64], [0, 0, 128], [0, 0, 196], [0, 64, 0], [0, 64, 64], [0, 64, 128],
           [0, 64, 196], [0, 128, 0], [0, 128, 64], [0, 128, 128], [0, 128, 196], [0, 196, 0], [0, 196, 64],
           [0, 196, 128], [0, 196, 196], [64, 0, 0], [64, 0, 64], [64, 0, 128], [64, 0, 196], [64, 64, 0],
           [64, 64, 64], [64, 64, 128], [64, 64, 196], [64, 128, 0], [64, 128, 64], [64, 128, 128], [64, 128, 196],
           [64, 196, 0], [64, 196, 64], [64, 196, 128], [64, 196, 196], [128, 0, 0], [128, 0, 64], [128, 0, 128],
           [128, 0, 196], [128, 64, 0], [128, 64, 64], [128, 64, 128], [128, 64, 196]]
def label2image(prelabel,colormap):
    ## 预测的到的标签转化为图像,针对一个标签图
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)#返回的是坐标值，类别为ii的坐标
        image[index,:] = colormap[ii]
    return image.reshape(h,w,3)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=21, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
unet=UNet(3,41).to(device)
'''
c_x1= Image.open("data/VOC2012/JPEGImages/2007_000063.jpg")#读取图片
c_y= Image.open("data/VOC2012/SegmentationClass/2007_000063.png")#读取相对应的标签图片
#由于网络设置要求输入图片长320，宽400，所以对图片进行中心剪裁
c_x1 = transforms.CenterCrop((high, width))(c_x1)
c_y = transforms.CenterCrop((high, width))(c_y)
#将读取的图片转化为RGB格式
c_x= c_x1.convert("RGB")
c_x=np.array(c_x)#转化为narray格式
#由于网络要求图片输入维数为（（3，长，宽））所以对图片维度数进行调换
c_x=c_x.transpose(2,1,0)
c_x=c_x.transpose(0,2,1)
c_x = torch.from_numpy(c_x)#将narray转化为tensor格式，输入网络必须是tensor格式
c_x=c_x.unsqueeze(0)#增加一个维度数，网络要求有四个维度
c_x  =c_x.float().to(device)#网络要求输入图片是float格式的
#前向传播
out=unet(c_x)
out = F.log_softmax(out,dim=1)
pre_lab = torch.argmax(out,1)
pre_lab_numpy = pre_lab.cpu().data.numpy()
#打印结果 从左往右三张分别是原图片，目标图片，网络前向传播输出图片
plt.figure(figsize=(16,6))#figsize是指定宽和高
plt.subplot(1,3,1)#第一个数和第二个数是分成几行几列，第三个数是摆放位置的序号
plt.imshow(c_x1)
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(c_y)
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(label2image(pre_lab_numpy[0],colormap))
plt.axis("off")
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
'''