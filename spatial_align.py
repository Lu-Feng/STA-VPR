# -*- coding: UTF-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
from numpy import *
import os
import sys
import cv2
import datetime
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from numba import cuda,float64
import math
import time
import gc
import copy
import numba as numba

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def writeData(data,fileName):
    fp1 = open(fileName, 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            fp1.write(str(data[i][j])+'\t')
        fp1.write('\n')
    fp1.close()

def densenet_cifar(net,input_data):
    x = net.features.conv0(input_data)
    x = net.features.norm0(x)
    x = net.features.relu0(x)
    x = net.features.pool0(x)
    x = net.features.denseblock1(x)
    x = net.features.transition1(x)
    x = net.features.denseblock2(x)
    x = net.features.transition2(x)
    x = net.features.denseblock3(x)
    x = net.features.transition3(x)
    x = net.features.denseblock4.denselayer1(x)
    x = net.features.denseblock4.denselayer2(x)
    x = net.features.denseblock4.denselayer3(x)
    x = net.features.denseblock4.denselayer4(x)
    x = net.features.denseblock4.denselayer5(x)
    x = net.features.denseblock4.denselayer6(x)
    x = net.features.denseblock4.denselayer7(x)
    x = net.features.denseblock4.denselayer8(x)
    x = net.features.denseblock4.denselayer9(x)
    x = net.features.denseblock4.denselayer10.norm1(x)
    x = net.features.denseblock4.denselayer10.relu1(x)
    x = x.permute(0,2,3,1)
    return x

def getFeature(img):
    #resnet = models.resnet50(pretrained=True)
    densenet = models.densenet161(pretrained=False)

    state_dict = torch.load("densenet161_places365.pth.tar")["state_dict"]
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("norm.1","norm1")
        k = k.replace("norm.2", "norm2")
        k = k.replace("conv.1", "conv1")
        k = k.replace("conv.2", "conv2")
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        if(name=="classifier.bias"):
            new_state_dict[name]=torch.ones(1000)
        if(name=="classifier.weight"):
            new_state_dict[name]=torch.ones(1000,2208)
    # load params
    densenet.load_state_dict(new_state_dict)

    densenet = densenet.cuda()
    densenet.eval()
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5])
    ])

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    x = img.unsqueeze(0)
    x = x.cuda()
    features = densenet_cifar(densenet, x)
    features = features.cpu()
    features = features[0].data.numpy()
    # featureData = np.ravel(features)
    featureData = []
    for i in range(7):
        featureData.append(np.ravel(features[:, i, :]))
    return featureData


def distance(x,y):
    l1=7
    l2=7
    M=np.zeros([l1, l2])
    D = np.zeros([l1, l2, 4])  
    for i in range(l1):
        for j in range(l2):
            M[i][j]=pdist([x[i],y[j]], 'cosine')
    writeData(M,"Mdata.txt")
        # ---------------------------------------------------------------
    I3 = np.argmin(M[3, :]) #the argmin of the third row
    print("I3:", I3)
    x = math.sqrt(1.0 + abs(I3 - 3))
    print(x)
    D[0][0] = [M[0][0], -1, -1, 1]
    for i in range(1, l1):
        D[i][0][0] = M[i][0] + D[i - 1][0][0]
        D[i][0][1] = i - 1
        D[i][0][2] = 0
        D[i][0][3] = 1 + D[i - 1][0][3]
    for j in range(1, l2):
        D[0][j][0] = M[0][j] + D[0][j - 1][0]
        D[0][j][1] = 0
        D[0][j][2] = j - 1
        D[0][j][3] = 1 + D[0][j - 1][3]
    for i in range(1, l1):
        for j in range(1, l2):
            cand1 = M[i][j] + D[i - 1][j][0]
            cand2 = M[i][j] + D[i][j - 1][0]
            cand3 = x * M[i][j] + D[i - 1][j - 1][0]
            minValue = min(cand1, cand2, cand3)
            if minValue == cand1:
                D[i][j][0] = cand1
                D[i][j][1] = i - 1
                D[i][j][2] = j
                D[i][j][3] = 1 + D[i - 1][j][3]
            elif minValue == cand2:
                D[i][j][0] = cand2
                D[i][j][1] = i
                D[i][j][2] = j - 1
                D[i][j][3] = 1 + D[i][j - 1][3]
            elif minValue == cand3:
                D[i][j][0] = cand3
                D[i][j][1] = i - 1
                D[i][j][2] = j - 1
                D[i][j][3] = x + D[i - 1][j - 1][3]
    lastCost = D[-1,-1, 0]/D[-1,-1, 3]
    sequenceIndex = []
    ii=l1-1
    jj=l2-1
    while(ii!=-1):
        sequenceIndex.insert(0,[ii,jj])
        ii,jj=(int(D[ii][jj][1]),int(D[ii][jj][2]))
    return "{0:7.6f}".format(lastCost),sequenceIndex


# img0=cv2.imread(r"/home/lf/Desktop/place/Garden/night_right/Image096.jpg")
# img1=cv2.imread(r"/home/lf/Desktop/place/Garden/day_left/Image096.jpg")

# path0=r"/home/lf/Desktop/place/Garden/day_left/Image"
# path1=r"/home/lf/Desktop/place/Garden/day_right/Image"
path0=r"/home/lf/Desktop/lufeng/nordland/Summer/"
path1=r"/home/lf/Desktop/lufeng/nordland/Winter/"
# path0 = r"/home/lf/Desktop/place/Berlin/BerlinA100/Reference/image"  # r"/home/lf/Desktop/place/Garden/day_left/Image"#
# path1 = r"/home/lf/Desktop/place/Berlin/BerlinA100/Live/image"  # r"/home/lf/Desktop/place/Garden/night_right/Image" #

i=40
# img0=cv2.imread(path0+"{:03d}".format(i)+".jpg")
# img1=cv2.imread(path1+"{:03d}".format(i)+".jpg")
img0 = cv2.imread(path0 + str(i) + ".jpg")
img1 = cv2.imread(path1 + str(i) + ".jpg")
img0 = img0[:, :int(0.7 * len(img0[0])), :]
#img1 = img1[:, :int(0.7 * len(img1[0])), :]
img1 = img1[:, int(0.1 * len(img1[0])):, :]
#
cv2.imwrite(str(i) + "l.jpg",img0)
cv2.imwrite(str(i) + "r.jpg",img1)

temp=getFeature(img0)
img0=getFeature(img1)
img1=temp

# print(shape(img0))
# print(pdist([np.ravel(img0), np.ravel(img1)], 'cosine'))

print("aligned distance:",distance(img0,img1))
img0=np.ravel(img0)
img1 = np.ravel(img1)
print("cosine distance:",pdist([img0,img1], 'cosine'))