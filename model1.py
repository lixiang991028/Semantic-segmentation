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
from torchvision.models import vgg19
from torchvision import transforms
from torchsummary import summary
#from data_generator import
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#判断用cuda还是cpu
high=320
width=480
device ='cpu'
model_vgg19 = vgg19(pretrained=True)#pretrained=True是指可以使用预训练模型
## 不使用vgg19网络中的后面的AdaptiveAvgPool2d和Linear层
base_model = model_vgg19.features
summary(base_model.cuda(),input_size=(3, high,width))
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
class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # num_classes:训练数据的类别
        self.num_classes = num_classes
        model_vgg19 = vgg19(pretrained=True)
        ## 不使用vgg9网络中的后面的AdaptiveAvgPool2d和Linear层
        self.base_model = model_vgg19.features
        ## 定义几个需要的层操作，并且使用转置卷积将特征映射进行升维
        # relu在每一次涉及到有参数的模块运算后，自动添加
        self.relu = nn.ReLU(inplace=True)
        # 反卷积卷积核大小不变，步长不变，padding不变，output_padding是反卷积的补全原图形长度，dilation是卷积核的间隔
        # 第一个参数是输入通道数，第二个参数是输出通道数
        # 只是反卷积，卷积可以改变形状和通道数，在前半段用卷积改变通道，用池化改变大小，现在用反卷积一步到位改变两者
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                          padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        self.deconv6 = nn.ConvTranspose2d(4096, 512, 3, 2, 1, 1, 1)
        ## vgg19中MaxPool2d所在的层
        self.layers = {"4": "maxpool_1", "9": "maxpool_2",
                       "18": "maxpool_3", "27": "maxpool_4",
                       "36": "maxpool_5"}

    def forward(self, x):
        output = {}
        for name, layer in self.base_model._modules.items():
            ## 从第一层开始获取图像的特征
            x = layer(x)
            ## 如果是layers参数指定的特征，那就保存到output中
            if name in self.layers:
                output[self.layers[name]] = x
        x5 = output["maxpool_5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["maxpool_4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["maxpool_3"]  # size=(N, 256, x.H/8,  x.W/8)
        ## 对特征进行相关的转置卷积操作,逐渐将图像放大到原始图像大小
        # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv1(x5))
        # 对应的元素相加, size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)
        # size=(N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv2(score))
        # 对应的元素相加, size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        # size=(N, 32, x.H, x.W)
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)
## 注意输入图像的尺寸应该是32的整数倍
fcn8s = FCN8s(41).to(device)#定义模型，21是分类数量
#summary(fcn8s,input_size=(3, high,width))#打印网络结构和参数
#读取一张图片进行前向传播

c_x1= Image.open("nyu_images/5.jpg")#读取图片
c_y= Image.open("nyu_labels40/5.png")#读取相对应的标签图片
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
out=fcn8s(c_x)
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