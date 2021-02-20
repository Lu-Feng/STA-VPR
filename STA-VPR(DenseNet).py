# -*- coding: UTF-8 -*-
#-------------------------------------------
#任 务：利用VGG19提取任意中间层特征
#-------------------------------------------

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
import re
from sklearn import random_projection

TPB=16

def loadDataSet(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = re.split('\t|,',line.strip())
        curLine=[float(temp) for temp in curLine]
        data.append(curLine)
    return data

def writeData(data,fileName):
    fp1 = open(fileName, 'w')
    for i in range(len(data)):
        for j in range(len(data[0])):
            fp1.write(str(data[i][j])+'\t')
        fp1.write('\n')
    fp1.close()

@cuda.jit
def dtwProcess(MMCindex,tempList,out):
    lQ=MMCindex.shape[1]
    l1=20
    l2=40
    for i in range(lQ):
        tempList[i][0]=1.0
    k = cuda.grid(1)
    if(k < lQ):
        M = cuda.local.array((l1, l2), dtype=float64)
        D = cuda.local.array((l1, l2), dtype=float64)  # 损失矩阵每个成员有4个数，第一个是累积距离，第2个是走过的距离，第3和4指示前一个元素的x和y坐标

        for i in range(l1):
            for j in range(l2):
                if(k+j<lQ):
                    M[i][j] = MMCindex[i][k + j]
                else:
                    M[i][j] = 3
        D[0][0] = 1
        # D[0][0][2] = -1.0
        # D[0][0][3] = -1.0
        for i in range(1, l1):
            M[i][0] = M[i][0] + M[i - 1][0]
            D[i][0] = 1 + D[i - 1][0]
            # D[i][0][2] = i - 1
            # D[i][0][3] = 0
        for j in range(1, l2):
            M[0][j] = M[0][j] + M[0][j - 1]
            D[0][j] = 1 + D[0][j - 1]
            # D[0][j][2] = 0
            # D[0][j][3] = j - 1
        for i in range(1, l1):
            for j in range(1, l2):
                if(M[i - 1][j] <= M[i][j - 1] and M[i - 1][j] <=M[i - 1][j - 1]):
                    M[i][j] = M[i][j] + M[i - 1][j]
                    D[i][j] = 1 + D[i - 1][j]
                    # D[i][j][2] = i - 1
                    # D[i][j][3] = j
                elif (M[i][j - 1] < M[i - 1][j - 1] and M[i][j - 1] < M[i - 1][j]):
                    M[i][j] = M[i][j] + M[i][j - 1]
                    D[i][j] = 1 + D[i][j - 1]
                    # D[i][j][2] = i
                    # D[i][j][3] = j - 1
                else:
                    M[i][j] = M[i][j] + M[i - 1][j - 1]
                    D[i][j] = 1 + D[i - 1][j - 1]
                    # D[i][j][2] = i - 1
                    # D[i][j][3] = j - 1

        for j in range(l2):
            if(M[l - 1][j]/D[l - 1][j]<tempList[k][0]):
                tempList[k][0]=M[l - 1][j]/D[l - 1][j]
                tempList[k][1]=k
                tempList[k][2]=k+j
    minCostIndex = 0
    for i in range(1,lQ):
        if(tempList[i][0]<tempList[minCostIndex][0]):
            minCostIndex=i
    out[0]=tempList[minCostIndex][0]#minCost
    out[1]=tempList[minCostIndex][1]#sequenceIndex
    out[2]=tempList[minCostIndex][2]

def dtw(MMCindex):
    '''host code for calling naive kernal
    '''
    d_MMCindex=cuda.to_device(MMCindex)
    tempList=np.zeros([len(MMCindex[0]),3])
    out = np.zeros(3)
    d_tempList=cuda.device_array(tempList.shape, np.float32)
    d_out = cuda.device_array(out.shape, np.float32)
    threadsperblock =8
    dtwProcess[math.ceil(len(MMCindex[0]) / threadsperblock), threadsperblock](d_MMCindex,d_tempList,d_out)
    return d_out.copy_to_host()

@cuda.jit
def distanceProcess1(A,B,C):
    '''matrix multiplication on gpu, naive method using global device memory
    '''
    ii, jj = cuda.grid(2)
    if ii < C.shape[0] and jj < C.shape[1]:

        l1 = 7
        l2 = 7
        M = cuda.local.array((l1, l2), dtype=float64)
        D = cuda.local.array((l1, l2), dtype=float64)  # 损失矩阵每个成员有4个数，第一个是累积距离，第2和3指示前一个元素的x和y坐标，第4个是走过的距离
        for i in range(l1):
            for j in range(l2):
                # if(i-j>2 or i-j<-2):
                #     M[i][j] = 3.0
                # else:
                    xy = modx = mody = 0.0
                    for k in range(len(A[ii][i])):  # 100352 is the length of vector
                        xy += A[ii][i][k] * B[jj][j][k]
                        modx += A[ii][i][k] * A[ii][i][k]
                        mody += B[jj][j][k] * B[jj][j][k]
                    M[i][j] = 1 - xy / math.sqrt(modx * mody)
                    # M[i][j] =10*(M[i][j]-0.56)
                    # M[i][j] =1.0/(1.0+math.exp(-M[i][j]))
            # ---------------------------------------------------------------
        I3value = min(M[3][0], M[3][1], M[3][2], M[3][3], M[3][4], M[3][5], M[3][6])

        I3 = 0
        for k in range(7):
            if I3value == M[3][k]:
                I3 = k
                break
        x = math.sqrt(1.0 + abs(I3 - 3))  #DMLI:set x=100 ; originalDTW set x=1

        # I2value = min(M[2][0], M[2][1], M[2][2], M[2][3], M[2][4], M[2][5], M[2][6])
        # I4value = min(M[4][0], M[4][1], M[4][2], M[4][3], M[4][4], M[4][5], M[4][6])
        # I2 = 0
        # for k in range(7):
        #     if I2value == M[2][k]:
        #         I2 = k
        #         break
        # I4 = 0
        # for k in range(7):
        #     if I4value == M[4][k]:
        #         I4 = k
        #         break
        # x = math.sqrt(1.0 + (abs(I2 + I3 + I4 - 2 - 3 - 4)) / 3.0)
        # x = math.sqrt(1.0 + (abs(I2 - 2) + abs(I3 - 3) + abs(I4 - 4)) / 3.0)

        D[0][0] = 1
        for i in range(1, l1):
            M[i][0] = M[i][0] + M[i - 1][0]
            D[i][0] = 1 + D[i - 1][0]
        for j in range(1, l2):
            M[0][j] = M[0][j] + M[0][j - 1]
            D[0][j] = 1 + D[0][j - 1]
        for i in range(1, l1):
            for j in range(1, l2):
                cand1 = M[i][j] + M[i - 1][j]
                cand2 = M[i][j] + M[i][j - 1]
                cand3 = x * M[i][j] + M[i - 1][j - 1]
                tempMin=min(cand1,cand2,cand3)
                if(tempMin==cand1):
                    M[i][j] = cand1
                    D[i][j] = 1 + D[i - 1][j]

                elif(tempMin==cand2):
                    M[i][j] = cand2
                    D[i][j] = 1 + D[i][j - 1]

                elif(tempMin==cand3):
                    M[i][j] = cand3
                    D[i][j] = x + D[i - 1][j - 1]
        C[ii][jj]=M[-1][-1]/D[-1][-1]

def distance1(A, B):
    '''host code for calling naive kernal
    '''

    lC=len(A)
    lQ=len(B)
    print(lC,lQ)
    print(shape(A),shape(B))
    d_A = cuda.to_device(A)  # d_ --> device
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((lC, lQ), np.float32)

    threadsperblock = (8, 8)
    blockspergrid_x = math.ceil(len(A) / threadsperblock[0])
    blockspergrid_y = math.ceil(len(B) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    distanceProcess1[blockspergrid, threadsperblock](d_A, d_B, d_C)

    return d_C.copy_to_host()

@cuda.jit
def distanceProcessV(A,B,C):
    '''matrix multiplication on gpu, naive method using global device memory
    '''
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        xy = modx = mody = 0.0
        for k in range(len(A[i])):  #100352 is the length of vector
            xy+=A[i][k]*B[j][k]
            modx+=A[i][k]*A[i][k]
            mody+=B[j][k]*B[j][k]
        C[i][j]=1-xy/math.sqrt(modx*mody)

def distanceV(A, B):
    '''host code for calling naive kernal
    '''
    lC=len(A)
    lQ=len(B)
    C = np.zeros([lC, lQ])
    d_A = cuda.to_device(A)  # d_ --> device
    d_B = cuda.to_device(B)
    d_C = cuda.device_array(C.shape, np.float32)

    threadsperblock = (8, 8)
    blockspergrid_x = math.ceil(len(A) / threadsperblock[0])
    blockspergrid_y = math.ceil(len(B) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    distanceProcessV[blockspergrid, threadsperblock](d_A, d_B, d_C)

    return d_C.copy_to_host()

def drawline(minCostList,sequenceIndex):
    global l,Plist1,stepLen
    #print(minCostList)
    drawLen=1000
    P = np.zeros(drawLen)
    R = np.zeros(drawLen)
    maxR=0.
    F=0.
    for k in range(drawLen):
        TP=0
        FN=0
        FP=0
        for i in range(len(minCostList)):
            if(minCostList[i]<k*1.0/drawLen):
                if (i == 0):
                    TP = TP + l / 2
                elif (i == Plist2[1] - l):
                    TP = TP + l / 2
                else:
                    temp1 = int(i * stepLen + 0.5 * (l - 1))
                    temp2 = int((sequenceIndex[i][0] + sequenceIndex[i][1]) / 2.0)
                    if (abs(temp1 - temp2) <= 3):
                        TP+=1
                    else:
                        FP+=1
            else:
                if (i == 0 or i == Plist2[1] - l):
                    FN = FN + l / 2
                else:
                    FN+=1
        if(TP+FP!=0 and TP + FN!=0):
            P[k]=TP/(TP+FP)
            R[k] =TP/(TP + FN)
        elif(TP+FP==0):
            P[k] = 1
            R[k] = 0
        if(P[k]==1):
            if(R[k]>maxR):
                maxR=R[k]
        if (P[k] + R[k] != 0 and 2 * P[k] * R[k] / (P[k] + R[k]) > F):
            F = 2 * P[k] * R[k] / (P[k] + R[k])
    #     fp1 = open("./PR/pWSU71.txt", 'a')
    #     fp1.write(str(P[k]) + '\t')
    #     fp2 = open("./PR/rWSU71.txt", 'a')
    #     fp2.write(str(R[k]) + '\t')
    # fp1.write('\n')
    # fp1.close()
    # fp2.write('\n')
    # fp2.close()

    print("F-score:",F)
    print("maxR:",maxR)
    plt.plot(R,P,"r")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlim(xmin=0, xmax=1.)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color="b",linestyle="--")
    ax=plt.gca()
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    #plt.savefig("SPSU.png")
    plt.show()

def drawline1(minCostList,argminIndex):
    drawLen=1000
    P = np.zeros(drawLen)
    R = np.zeros(drawLen)
    maxR=0.
    F=0.
    for k in range(drawLen):
        TP = 0
        FN = 0
        FP = 0
        for i in range(len(minCostList)):
            if (minCostList[i] < k * 1.0 / drawLen):
                if (abs(i-argminIndex[i])<=3):
                    TP += 1
                else:
                    FP += 1
            else:
                FN += 1
        if (TP + FP != 0 and TP + FN != 0):
            P[k] = TP / (TP + FP)
            R[k] = TP / (TP + FN)
        elif (TP + FP == 0):
            P[k] = 1
            R[k] = 0
        if (P[k] == 1):
            if (R[k] > maxR):
                maxR = R[k]
        if (P[k] + R[k] != 0 and 2 * P[k] * R[k] / (P[k] + R[k]) > F):
            F = 2 * P[k] * R[k] / (P[k] + R[k])
    #     fp1 = open("./PR/pWSU71.txt", 'a')
    #     fp1.write(str(P[k]) + '\t')
    #     fp2 = open("./PR/rWSU71.txt", 'a')
    #     fp2.write(str(R[k]) + '\t')
    # fp1.write('\n')
    # fp1.close()
    # fp2.write('\n')
    # fp2.close()

    print("F-score:", F)
    print("maxR:", maxR)
    plt.plot(R,P,"r")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlim(xmin=0, xmax=1.)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(color="b",linestyle="--")
    ax=plt.gca()
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    #plt.savefig("SPSU.png")
    plt.show()

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
    # x = net.features.denseblock4.denselayer10.conv1(x)
    # maxp = nn.MaxPool2d([3,1],[2,1])
    # x = maxp(x)
    x = x.permute(0,2,3,1)
    return x

def dtwCpu(num):
    global minCost,MM,l,lQ,Cindex
    if(lQ-num<2*l):
        M=MM[Cindex*stepLen:Cindex*stepLen+l,num:lQ]
    else:
        M=MM[Cindex*stepLen:Cindex*stepLen+l,num:num+2*l]
    l1=len(M)
    l2=len(M[0])
#---------------------------------------------------------------
    D=np.zeros([l1,l2,4])#损失矩阵每个成员有4个数，第一个是累积距离，第2和3指示前一个元素的x和y坐标，第4个是走过的距离
    D[0][0]=[M[0][0],-1,-1,1]
    for i in range(1, l1):
        D[i][0][0] = M[i][0] + D[i - 1][0][0]
        D[i][0][1] = i-1
        D[i][0][2] = 0
        D[i][0][3] = 1 + D[i - 1][0][3]
    for j in range(1, l2):
        D[0][j][0] = M[0][j]+D[0][j-1][0]
        D[0][j][1] = 0
        D[0][j][2] = j-1
        D[0][j][3] = 1 + D[0][j - 1][3]
    for i in range(1, l1):
        for j in range(1, l2):
            #choice=[D[i - 1][j][0],D[i][j - 1][0], D[i - 1][j - 1][0],D[i][j - 2][0]*5, D[i - 1][j - 2][0]*5,D[i-2][j][0]*5, D[i - 2][j - 1][0]*5]#,D[i - 2][j - 2][0]*5]
            choice = [D[i - 1][j][0], D[i][j - 1][0], D[i - 1][j - 1][0]]
            minValue=min(choice)
            minList=choice.index(minValue)
            D[i][j][0] = M[i,j]+ minValue
            if minList==0:
                D[i][j][1] =i - 1
                D[i][j][2] =j
                D[i][j][3] = 1+D[i-1][j][3]
            elif minList==1:
                D[i][j][1] = i
                D[i][j][2] = j- 1
                D[i][j][3] = 1 + D[i][j - 1][3]
            elif minList==2:
                D[i][j][1] = i-1
                D[i][j][2] = j - 1
                D[i][j][3] = 1 + D[i-1][j - 1][3]
    lastLineCost = D[l1 - 1, :, 0]/D[l1 - 1, :, 3]
    lastLine=lastLineCost.tolist()
    lineMinCost=min(lastLine)
    jj=lastLine.index(lineMinCost)
    return [lineMinCost,[num,num+jj]]

def vgg16Feature(Plist1,Plist2,path1,path2):
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

    C=[]
    Q=[]
    for i in range(Plist1[0],Plist1[1]):
        # img=Image.open(path1+"{:03d}".format(i)+".jpg")
        # #img = Image.open(path1 + str(i) + ".jpg")
        # #img=img[:,int(0.2*len(img[0])):,:]
        # img = transform(img)
        # x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        img=cv2.imread(path1+"{:03d}".format(i)+".jpg")
        # img = cv2.imread(path1 + str(i) + ".jpg")
        # img = img[:, :int(0.7 * len(img[0])), :]
        img = cv2.resize(img, (224, 224))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=transform(img)
        x=img.unsqueeze(0)
        x = x.cuda()
        features = densenet_cifar(densenet, x)
        features = features.cpu()
        features = features[0].data.numpy()
        #featureData = np.ravel(features)
        featureData =[]
        for i in range(7):
            featureData.append(np.ravel(features[:,i, :]))
        Q.append(featureData)

    for i in range(Plist2[0],Plist2[1]):
        # img=Image.open(path2+"{:03d}".format(i)+".jpg")
        # #img = Image.open(path2 + str(i) + ".jpg")
        # #img=img[:,int(0.2*len(img[0])):,:]
        # img = transform(img)
        # x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        img=cv2.imread(path2+"{:03d}".format(i)+".jpg")
        # img = cv2.imread(path2 + str(i) + ".jpg")
        # img = img[:, int(0.1 * len(img[0])):, :]
        img = cv2.resize(img, (224, 224))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=transform(img)
        x=img.unsqueeze(0)
        x = x.cuda()
        features = densenet_cifar(densenet, x)
        features = features.cpu()
        features = features[0].data.numpy()
        #featureData =np.ravel(features)
        featureData =[]
        for i in range(7):
            featureData.append(np.ravel(features[:,i, :]))
        C.append(featureData)
    return Q,C

if __name__ == '__main__':
    Plist1=[0,200]
    Plist2=[0,200]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    path1=r"/home/lf/Desktop/place/Garden/day_left/Image"#r"/home/lf/Desktop/lufeng/nordland/Summer/"#r"/home/lf/Desktop/place/UofA_dataset/day/"#
    path2=r"/home/lf/Desktop/place/Garden/day_right/Image" #r"/home/lf/Desktop/lufeng/nordland/Winter/"#r"/home/lf/Desktop/place/UofA_dataset/evening/"#

    start = datetime.datetime.now()
    Q,C=vgg16Feature(Plist1,Plist2,path1,path2)
    cuda.close()
    end=datetime.datetime.now()
    print("extract feature running time:", end - start)

    # start = datetime.datetime.now()
    # SQ, SC = shape(Q), shape(C)
    # rpQ, rpC = np.array(Q), np.array(C)
    # rpQ, rpC = rpQ.reshape(SQ[0] * SQ[1], SQ[2]), rpC.reshape(SC[0] * SC[1], SC[2])
    # print(np.shape(rpC), np.shape(rpQ))
    # reducedD = 512
    # transformer = random_projection.GaussianRandomProjection(n_components=reducedD).fit(rpQ)
    # Q, C = [], []
    # for i in range(Plist1[1]):
    #     Q.append(transformer.transform(rpQ[7 * i:7 * i + 7]))
    # for i in range(Plist2[1]):
    #     C.append(transformer.transform(rpC[7 * i:7 * i + 7]))
    # # rpQ = transformer.transform(rpQ)
    # # rpC = transformer.transform(rpC)
    # # Q1 = rpQ.reshape(SQ[0], SQ[1], reducedD)
    # # C1 = rpC.reshape(SC[0], SC[1], reducedD)
    # # Q = Q1.tolist()
    # # C = C1.tolist()
    # end = datetime.datetime.now()
    # print("reduce dimension running time:", end - start)

    #for i in range(6):
    start = datetime.datetime.now()
    MM=distance1(C,Q)
    cuda.close()
    end=datetime.datetime.now()
    print("matrix running time:", end - start)
    # writeData(MM,'test.txt')
    # MM = loadDataSet('./adaptiveDTW/DenseNet_WSU.txt')
    MM = np.array(MM)
    print(np.shape(MM))

    # # using the following code when l=1 (i.e. single image matching)
    # argminsimList = np.argmin(MM, axis=1)
    # minsimList = np.min(MM, axis=1)
    # # for i in range(877,882):
    # #     print(i,argminsimList[i])
    # drawline1(minsimList, argminsimList)

    #using the following code when l!=1 (i.e. sequence matching)
    lQ=len(MM[0])
    stepLen = 1  # step length
    l = 20  # length of match sequence
    count = 181 # count of match times
    change=0
    minCostList= np.zeros(count)
    sequenceIndex = np.zeros([count,2])

    start = datetime.datetime.now()
    for Cindex in range(count):
        MMCindex = copy.deepcopy(MM[Cindex * stepLen:Cindex * stepLen + l, :])
        # start = datetime.datetime.now()
        minCostList[Cindex],sequenceIndex[Cindex][0],sequenceIndex[Cindex][1]=dtw(MMCindex)
        # end = datetime.datetime.now()
        # print("running time:", end - start)
        sequenceIndex[Cindex][0]=sequenceIndex[Cindex][0]+change
        sequenceIndex[Cindex][1]=sequenceIndex[Cindex][1]+change
        sequenceStart =int(sequenceIndex[Cindex][0])
        sequenceEnd =int(sequenceIndex[Cindex][1])
        print("原序列:","{:03d}".format(Cindex*stepLen),"~","{:03d}".format(Cindex*stepLen+l-1),"   minCost:",\
              "{0:7.4f}".format(minCostList[Cindex]),"   匹配序列结果:", "{:04d}".format(sequenceStart),"~","{:04d}".format(sequenceEnd))
    end = datetime.datetime.now()
    print("running time:", end - start)

    # # -----------------------one processing-----------------------
    # for Cindex in range(count):
    #     minCostListTemp = sys.maxsize
    #     sequenceIndexTemp = [1, 1]
    #     for k in range(lQ):
    #         out = dtwCpu(k)
    #         if (out[0] < minCostListTemp):
    #             minCostListTemp = out[0]
    #             sequenceIndexTemp=out[1]
    #     minCostList[Cindex]=minCostListTemp
    #     sequenceIndex[Cindex]=sequenceIndexTemp
    #     print("原序列(right):", "{:03d}".format(Cindex * stepLen), "~", "{:03d}".format(Cindex * stepLen + l - 1),
    #           "   minCost:", \
    #           "{0:7.4f}".format(minCostList[Cindex]), "   匹配序列结果(left):", "{:04d}".format(sequenceIndexTemp[0]), "~",
    #           "{:04d}".format(sequenceIndexTemp[1]))
    # # ------------

    drawline(minCostList,sequenceIndex)