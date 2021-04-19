# -*- coding: UTF-8 -*-
#-------------------------------------------
# the code for STA-VPR proposed in "STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition" IEEE RA-L 2021.
# please consider citing our paper if you use any part of the provided code
# Time: 2021-4-15
# Author: Feng Lu (lufengrv@gmail.com)
#-------------------------------------------

import numpy as np
import sys
import argparse

def loadData(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        curLine=[float(temp) for temp in curLine]
        data.append(curLine)
    return data

def dtwProcess(num,D_CH):
    global alignedImgPairs,minSeqDistan,l,H_len,startID
    # get the distance matrix between query seq C and current candidate seq T' (It's twice as long as C)
    if(H_len-num<2*l):
        D=D_CH[:,num:H_len]
    else:
        D=D_CH[:,num:num+2*l]

    [Cl,T2l]=D.shape    #get the length of seq C (Cl) and seq T' (T2l)

#----------------------------dynamic programming----------------------------------------------
    # S is cumulative distance matrix, pi and pj record the coordinate (i and j) of previous point. C record the total cost from (0,0) to (i,j)
    S, pi, pj, C=np.zeros([Cl,T2l]), np.zeros([Cl,T2l]), np.zeros([Cl,T2l]), np.zeros([Cl,T2l])
    S[0][0], pi[0][0], pj[0][0], C[0][0]=D[0][0], -1, -1, 1
    for i in range(1, Cl):
        S[i][0] = D[i][0] + S[i - 1][0]
        pi[i][0] = i-1
        pj[i][0] = 0
        C[i][0] = 1 + C[i - 1][0]
    for j in range(1, T2l):
        S[0][j] = D[0][j]+S[0][j-1]
        pi[0][j] = 0
        pj[0][j] = j-1
        C[0][j] = 1 + C[0][j - 1]
    for i in range(1, Cl):
        for j in range(1, T2l):
            choice = [S[i - 1][j], S[i][j - 1], S[i - 1][j - 1]]
            minValue=min(choice)
            minList=choice.index(minValue)
            S[i][j] = D[i,j]+ minValue
            if minList==0:
                pi[i][j] =i - 1
                pj[i][j] =j
                C[i][j] = 1 + C[i-1][j]
            elif minList==1:
                pi[i][j] = i
                pj[i][j] = j - 1
                C[i][j] = 1 + C[i][j - 1]
            elif minList==2:
                pi[i][j] = i - 1
                pj[i][j] = j - 1
                C[i][j] = 1 + C[i-1][j - 1]

    lastLineDistans = S[Cl - 1, :]/C[Cl - 1, :]   # normalize last line
    lastLineDistans=lastLineDistans.tolist()
    curMinSeqDistan=min(lastLineDistans)  # find min seq distance in last line
    if curMinSeqDistan < minSeqDistan:  # if the distance between query seq and current candidate seq T < previous min distance
        minSeqDistan=curMinSeqDistan    # update min distance
        # update aligned image pairs, backtracking
        alignedImgPairs = []
        ii=Cl-1
        jj=np.argmin(lastLineDistans)
        while(ii!=-1):
            alignedImgPairs.insert(0,[ii+startID,num+jj])
            ii,jj=(int(pi[ii][jj]),int(pj[ii][jj]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='python argparse cookbook')
    parser.add_argument('-l', '--length', help="get length of query sequence")
    parser.add_argument('-s', '--startid', help="get start ID of query sequence")
    parser.add_argument('-n', '--name', help="get file name of distance matrix")
    args = parser.parse_args()

    l=int(args.length)  #length of query sequence
    startID=int(args.startid)  #start ID of query sequence
    fname = args.name  # file name of distance matrix between all query images and historical reference images

    D_aCH=loadData(fname)   #load the distance matrix
    D_aCH = np.array(D_aCH)
    D_CH=D_aCH[startID:startID+l,:]   #get the distance matrix between current query seq and historical reference seq

    H_len=D_CH.shape[1]

    minSeqDistan= sys.maxsize    # global variable for recording min sequence distance
    alignedImgPairs=[]    # global variable for recording aligned image pairs
    for k in range(H_len):
        dtwProcess(k,D_CH)
    print("matched result sequence (startID - endID):", "{:04d}".format(alignedImgPairs[0][1]), "-", "{:04d}".format(alignedImgPairs[-1][1]))
    print("sequence distance:", minSeqDistan)
    print("temporal alignment result:")
    for align in alignedImgPairs:
        print("{:04d}".format(align[0]), "<-->", "{:04d}".format(align[1]))
    for align in alignedImgPairs:
        if(abs(align[0]-align[1])>2):
            print("non-corresponding frame alignment (out of tolerance):", "{:04d}".format(align[0]), "<-->", "{:04d}".format(align[1]))