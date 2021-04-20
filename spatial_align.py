# -*- coding: UTF-8 -*-
#-------------------------------------------
# the code for STA-VPR proposed in "STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition" IEEE RA-L 2021.
# please consider citing our paper if you use any part of the provided code
# Time: 2021-4-15
# Author: Feng Lu (lufengrv@gmail.com)
#-------------------------------------------

import torch
from torchvision import models, transforms
import numpy as np
import os
import cv2
from scipy.spatial.distance import pdist
import math
import argparse

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

def densenetFeature(img):
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

def alignDistance(x,y):
    l1, l2 = 7, 7
    D=np.zeros([l1, l2])

    # --------------compute distance matrix-------------------------------------
    for i in range(l1):
        for j in range(l2):
            D[i][j]=pdist([x[i],y[j]], 'cosine')
    writeData(D,"FeaDisMat.txt")       # distance matrix of local features

    # ----------------compute adaptive parameter a------------------------------------
    I3 = np.argmin(D[3, :]) #the argmin of the third row
    print("I3 =", I3)   # the same as I4 in our paper
    a = math.sqrt(1.0 + abs(I3 - 3))
    print("adaptive parameter a =",a)

    # ------------------------dynamic programming----------------------------------
    # S is cumulative distance matrix, pi and pj record the coordinate (i and j) of previous point. C record the total cost from (0,0) to (i,j)
    S, pi, pj, C = np.zeros([l1, l2]), np.zeros([l1, l2]), np.zeros([l1, l2]), np.zeros([l1, l2])
    S[0][0], pi[0][0], pj[0][0], C[0][0] = D[0][0], -1, -1, 1
    for i in range(1, l1):
        S[i][0] = D[i][0] + S[i - 1][0]
        pi[i][0] = i-1
        pj[i][0] = 0
        C[i][0] = 1 + C[i - 1][0]
    for j in range(1, l2):
        S[0][j] = D[0][j]+S[0][j-1]
        pi[0][j] = 0
        pj[0][j] = j-1
        C[0][j] = 1 + C[0][j - 1]
    for i in range(1, l1):
        for j in range(1, l2):
            cand1 = D[i][j] + S[i - 1][j]
            cand2 = D[i][j] + S[i][j - 1]
            cand3 = a * D[i][j] + S[i - 1][j - 1]
            minValue = min(cand1, cand2, cand3)
            if minValue == cand1:
                S[i][j] = cand1
                pi[i][j] =i - 1
                pj[i][j] =j
                C[i][j] = 1 + C[i-1][j]
            elif minValue == cand2:
                S[i][j] = cand2
                pi[i][j] = i
                pj[i][j] = j - 1
                C[i][j] = 1 + C[i][j - 1]
            elif minValue == cand3:
                S[i][j] = cand3
                pi[i][j] = i - 1
                pj[i][j] = j - 1
                C[i][j] = a + C[i - 1][j - 1]
    ImgDistan = S[-1,-1]/C[-1,-1]    # normalize endpoint to get image distance
    alignedFeaPairs = []
    ii=l1-1
    jj=l2-1
    while(ii!=-1):  #backtracking
        alignedFeaPairs.insert(0,[ii,jj])
        ii,jj=(int(pi[ii][jj]),int(pj[ii][jj]))
    return "{0:7.6f}".format(ImgDistan),alignedFeaPairs

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='python argparse cookbook')
    parser.add_argument('-x', '--imageX', help="get file+path of image1")
    parser.add_argument('-y', '--imageY', help="get file+path of image2")
    args = parser.parse_args()

    img0 = cv2.imread(args.imageX)
    # img0 = img0[:, int(0.1 * len(img0[0])):, :]
    img1 = cv2.imread(args.imageY)
    # img1 = img1[:, :int(0.7 * len(img1[0])), :]

    img0=densenetFeature(img0)
    img1=densenetFeature(img1)

    ImgDistan, alignedFeaPairs=alignDistance(img0,img1)
    print("spatial alignment result (local feature pairs):")
    for align in alignedFeaPairs:
        print(align[0], "<-->",align[1])
    print("aligned distance:",ImgDistan)

    print("cosine distance:","{0:7.6f}".format(pdist([np.ravel(img0),np.ravel(img1)], 'cosine')[0]))