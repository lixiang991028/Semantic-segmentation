# 导入所需要的模块
import numpy as np
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
from torchvision import transforms
from torchsummary import summary
# 定义一个读取图像的函数
# def read_image(root):
#     """读取指定路径下的所指定的图像文件"""
#     image = np.loadtxt(root,dtype=str)#root是文件夹所在路径，dtype是读取后强制转换为str类型
#     n = len(image)#n是图片个数
#     data, label = [None]*n , [None]*n#生成有n个none元素的数组
#     for i, fname in enumerate(image):#i是索引，从0开始，fname是图片名称
#         data[i] = imread("C:/Users/l/Desktop/NYU/nyu_images/%s.jpg" %(fname))#%s位置的值是fname,这里需要自己改一下路径
#         label[i] = imread("C:/Users/l/Desktop/NYU/nyu_labels40/%s.png" %(fname))
#     return data,label
# ## 读取训练数据
# traindata1,trainlabel1 = read_image(root = "C:/Users/l/Desktop/NYU/train_mini.txt")#用来做cpu训练展示的
# #print(trainlabel[1].shape)
# testdata,testlabel = read_image(root = "C:/Users/l/Desktop/NYU/test.txt")
'''
## 查看训练集和验证集的一些图像
plt.figure(figsize=(12,8))#figsize是指定宽和高
plt.subplot(2,2,1)#第一个数和第二个数是分成几行几列，第三个数是摆放位置的序号
plt.imshow(traindata[0])#traindata里面的序号是前面的索引
plt.subplot(2,2,2)
plt.imshow(trainlabel[0])
plt.subplot(2,2,3)
plt.imshow(traindata[9])
plt.subplot(2,2,4)
plt.imshow(trainlabel[9])
plt.show()
'''
# 每个类的RGB值
color=[0,64,128,196]
Color=[]
COLOR=[]
for i in range(4):
    Color.append(color[i])
    for j in range(4):
        Color.append(color[j])
        for k in range(4):
            Color.append(color[k])
            COLOR.append(Color)
            Color=Color[0:2]
            if (len(COLOR) == 41):
                break
        Color = Color[0:1]
        if (len(COLOR) == 41):
            break
    Color.clear()
    if (len(COLOR) == 41):
        break

colormap=COLOR
# 给定一个标号图片，将图像的每个像素打上标签
#输入label图片3*320*480和colormap，输出0到20标记的1*320*480
def image2label(image):
    #256是RGB 0-255强度范围
    #cm2lbl = np.zeros(256**3)
    #for i,cm in enumerate(colormap):
        #cm2lbl[(cm[0]*256*256+cm[1]*256+cm[2])] = i #将RGB像素值唯一对应为标签
    ## 对一张图像转换，将图像的每个像素打上标签
    image = np.array(image, dtype="int64")
    # ix = (image[:,:,0]*256*256+image[:,:,1]*256+image[:,:,2])
    # image2 = cm2lbl[ix]
    return image[:,:,0]

## 随机裁剪图像数据
def rand_crop(data,label,high,width):
    im_width,im_high = data.size
    ## 生成图像随机点的位置，将输入图像随机截取成480*320的新图像
    left = np.random.randint(0,im_width - width)#左边随机点
    top = np.random.randint(0,im_high - high)#顶
    right = left+width
    bottom = top+high
    data = data.crop((left, top, right, bottom))#新图像size 480*320
    label = label.crop((left, top, right, bottom))#新图像像素点标签
    return data,label
## 单个图像的转换操作，图像预处理操作
def img_transforms(data, label, high,width,colormap):
    data, label = rand_crop(data, label, high,width)#生成随机裁剪的图像数据
    data=np.array(data)#将图像数据转化为array格式
    #图像转换操作
    data_tfs = transforms.Compose([
        transforms.ToTensor(),#图像转化为tensor格式
        transforms.Normalize([0.485, 0.456, 0.406],#图像标准化
                            [0.229, 0.224, 0.225])])

    data = data_tfs(data)
    label = torch.from_numpy(image2label(label))
    return data, label
## 定义一列出需要读取的数据路径的函数
def read_image_path(root):
    """保存指定路径下的所有需要读取的图像文件路径"""
    image = np.loadtxt(root,dtype=str)
    n = len(image)#训练集图片数量
    data, label = [None]*n , [None]*n
    #加载训练集原始图片和标签图片
    for i, fname in enumerate(image):
        data[i] = "../nyu_images/%s.jpg" %(fname)
        label[i] = "../nyu_labels40/%s.png" %(fname)
    return data,label



#定义 MyDataset 继承于torch.utils.data.Dataset构成自定的训练集
class MyDataset(Data.Dataset):
    """用于读取图像，并进行相应的裁剪等"""

    def __init__(self, data_root, high, width, imtransform, colormap):
        # data_root:数据索引所对应的路径,high,width:图像裁剪后的尺寸,high,width = 320,480
        ## imtransform:预处理操作,colormap:颜色
        self.data_root = data_root
        self.high = high
        self.width = width
        self.imtransform = imtransform
        self.colormap = colormap
       
        data_list, label_list = read_image_path(root=data_root)#生成原始图片和标签图片的列表 label_list\data_list{list:1464}
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

    def _filter(self, images):
        # 过滤掉图片大小小于指定high,width的图片，要把大于该长宽的保存
        return [im for im in images if (Image.open(im).size[1] >= 448 and
                                        Image.open(im).size[0] > 480)]  # for in if 满足条件的保存

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.imtransform(img, label, self.high,
                                      self.width, self.colormap)
        return img, label

    def __len__(self):
        return len(self.data_list)
    
    
def Get_test(test_root,train_root,img_transforms,colormap,high,width):
    # 读取数据
    high,width = 448,480
    voc_train1 = MyDataset(train_root,high,width, img_transforms,colormap)#img_transforms传入的是一个函数

    voc_test = MyDataset(test_root,high,width, img_transforms,colormap)
    # 创建数据加载器每个batch使用4张图像，数据封装
    train_loader1 = Data.DataLoader(voc_train1, batch_size=4,shuffle=True,
                                   num_workers=0,pin_memory=True)#将数据集每四个分成一组，shuffle是随机分配，pin_memory=True是存数据在CPU中，否则在GPU中

    test_loader=Data.DataLoader(voc_test, batch_size=4,shuffle=True,
                             num_workers=0,pin_memory=True)
    return train_loader1,test_loader
    ##  检查训练数据集的一个batch的样本的维度是否正确

# for step, (b_x, b_y) in enumerate(train_loader1):#b_x是图片数据tensor（4,3,320,480），b_y是标签数据tensor（4,320,480）
#     if step > 0:
#         break
'''
## 输出训练图像的尺寸和标签的尺寸，和数据类型
print("b_x.shape:",b_x.shape)
print("b_y.shape:",b_y.shape)
print("b_x.dtype:",b_x.dtype)
print("b_y.dtype:",b_y.dtype)
'''
## 将标准化后的图像转化为0～1的区间，归一化操作
def inv_normalize_image(data):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * rgb_std + rgb_mean
    return data.clip(0,1)

## 从预测的标签转化为图像的操作
def label2image(prelabel,colormap):
    ## 预测的到的标签图像转化为有颜色的图像分割图像,针对一个将像素点标签值转化为颜色
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)#prelabel ndarray:(153600,1)
    image = np.zeros((h*w,3),dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)##返回的是坐标值，类别为ii的坐标
        image[index,:] = colormap[ii]#将类别为ii的像素点赋上相应的RGB值
    return image.reshape(h,w,3)
# ## 可视化一个batch的图像，检查数据预处理 是否正确
# b_x_numpy = b_x.data.numpy()#将数据从tensor（4,3,320,480）转化为ndarray格式
# #print(b_x_numpy.shape)
# b_x_numpy = b_x_numpy.transpose(0,2,3,1)#（4,3,320,480）转化为（4,320,480,3）方便取出一个banch
# #print(b_x_numpy.shape)
# b_y_numpy = b_y.data.numpy()
# #print(b_y_numpy.shape)
'''
plt.figure(figsize=(16,6))
#print(inv_normalize_image(b_x_numpy[0]).shape)
#输出第一批次前4张图片看一下效果
for ii in range(4):
    plt.subplot(2,4,ii+1)
    plt.imshow(inv_normalize_image(b_x_numpy[ii]))
    plt.axis("off")
    plt.subplot(2,4,ii+5)
    plt.imshow(label2image(b_y_numpy[ii],colormap))
    plt.axis("off")
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
'''