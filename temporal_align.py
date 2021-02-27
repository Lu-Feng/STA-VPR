
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import multiprocessing
import numpy as np
from numpy import *
import os
import sys
import tensorflow as tf
import cv2
import datetime
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from numba import jit,cuda, float64
import math
import time
import gc

def loadDataSet(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
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

def dtwProcess(num,MM):
    global sequenceIndex,minCost,l,lQ,startID
    if(lQ-num<2*l):
        M=MM[:,num:lQ]
    else:
        M=MM[:,num:num+2*l]
    l1=len(M)
    l2=len(M[0])
#---------------------------------------------------------------
    D=np.zeros([l1,l2,4])
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
    if minCost>lineMinCost:
        minCost=lineMinCost
        sequenceIndex = []
        ii=l1-1
        jj=lastLine.index(min(lastLine))
        while(ii!=-1):
            sequenceIndex.insert(0,[ii+startID,num+jj])
            ii,jj=(int(D[ii][jj][1]),int(D[ii][jj][2]))

if __name__ == '__main__':

    stepLen=1 #step length
    l=20  #length of match sequence
    startID=180

    MM=loadDataSet('test.txt')
    MM = np.array(MM)
    MM=MM[startID:startID+l,:]

    lQ=len(MM[0])
    start = datetime.datetime.now()

# -----------------------one processing-----------------------
    minCost= sys.maxsize
    sequenceIndex=[]
    for k in range(lQ):
        dtwProcess(k,MM)
    print(minCost)
    print(sequenceIndex)
    for i in sequenceIndex:
        if(abs(i[0]-i[1])>2):
            print(i)

    end=datetime.datetime.now()
    print("running time:", end - start)