# 从mat文件提取labels
# 需要注意这个文件里面的格式和官方有所不同，长宽需要互换，也就是进行转置
import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

dataFile = './labels40.mat'
data = scio.loadmat(dataFile)
labels = np.array(data["labels40"])

path_converted = './nyu_labels40'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

labels_number = []
for i in range(1449):
    labels_number.append(labels[:, :, i].transpose((1, 0)))  # 转置
    labels_0 = np.array(labels_number[i])
    # print labels_0.shape
    print(type(labels_0))
    label_img = Image.fromarray(np.uint8(labels_number[i]))
    # label_img = label_img.rotate(270)
    label_img = label_img.transpose(Image.ROTATE_270)

    iconpath = './nyu_labels40/' + str('%d' % (i + 1)) + '.png'
    label_img.save(iconpath, optimize=True)

