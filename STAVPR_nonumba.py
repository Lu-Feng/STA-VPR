# -*- coding: UTF-8 -*-
#-------------------------------------------
# the code for STA-VPR proposed in "STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition" IEEE RA-L 2021.
# Time: 2021-4-15
# Author: Feng Lu (lufengrv@gmail.com)
#-------------------------------------------
import numpy as np
import math
from sklearn import random_projection
from tqdm import tqdm
from scipy.spatial.distance import pdist

def LMDTW(D_CH):
    Hl=D_CH.shape[1]  # length of seq H
    Cl=20  #length of seq C
    Tl=40  #length of candidate seq T'
    tempList = np.zeros([Hl, 2])  # for save the information (distance with query seq, length) of H_len candidate seqs.
    for i in range(Hl):
        tempList[i][0]=1.0   #initialize seq distance to a value >=1
    for k in range(Hl):
        # note that D is distance matrix before dynamic programming, but is cumulative distance matrix after dynamic programming.
        D = np.zeros([Cl, Tl])  # record image distance between image i and iamge j, i.e. distance matrix
        C = np.zeros([Cl, Tl])  # record total cost C from point (0,0) to (i,j)

        # --------------copy distance matrix-------------------------------------
        for i in range(Cl):
            for j in range(Tl):
                if(k+j<Hl):
                    D[i][j] = D_CH[i][k + j]
                else:
                    D[i][j] = 3   #if out of range, padding with a large number

        # -------------------dynamic programming--------------------------------------
        # note that after the update, D is cumulative distance matrix(i.e. matrix S in our paper)
        C[0][0] = 1
        for i in range(1, Cl):
            D[i][0] = D[i][0] + D[i - 1][0]
            C[i][0] = 1 + C[i - 1][0]
        for j in range(1, Tl):
            D[0][j] = D[0][j] + D[0][j - 1]
            C[0][j] = 1 + C[0][j - 1]
        for i in range(1, Cl):
            for j in range(1, Tl):
                if(D[i - 1][j] <= D[i][j - 1] and D[i - 1][j] <=D[i - 1][j - 1]):
                    D[i][j] = D[i][j] + D[i - 1][j]
                    C[i][j] = 1 + C[i - 1][j]
                elif (D[i][j - 1] < D[i - 1][j - 1] and D[i][j - 1] < D[i - 1][j]):
                    D[i][j] = D[i][j] + D[i][j - 1]
                    C[i][j] = 1 + C[i][j - 1]
                else:
                    D[i][j] = D[i][j] + D[i - 1][j - 1]
                    C[i][j] = 1 + C[i - 1][j - 1]

        #-----------find the best local sebsequence of k-th candidate seq T for matching query seq---------------
        for j in range(Tl):
            if(D[Cl - 1][j]/C[Cl - 1][j]<tempList[k][0]):
                tempList[k][0]=D[Cl - 1][j]/C[Cl - 1][j]   #for save seq distance between k-th candidate seq T and query seq
                tempList[k][1]=j   #for save length of k-th candidate seq T

    #----------find the start ID of the matched result seq T (that have min seq distance)--------------
    minDistanIndex = 0   #initialize start ID
    for i in range(1,Hl):
        if(tempList[i][0]<tempList[minDistanIndex][0]):
            minDistanIndex=i

    # get all information of matched result seq T
    minSeqDis=tempList[minDistanIndex][0]  #min seq distance (between seq C and T)
    startID=minDistanIndex  #start ID of matched result seq T
    endID=minDistanIndex+tempList[minDistanIndex][1]  #end ID of matched result seq T
    return minSeqDis, startID, endID

def alignDistance(aC,H,flag=True):
    l_aC = len(aC)
    l_H = len(H)
    D_aCH = np.zeros([l_aC, l_H])
    sta = "start to compute distance matrix D_aCH"
    for ii in tqdm(range(D_aCH.shape[0]), sta):
        for jj in range(D_aCH.shape[1]):
            l1 = 7
            l2 = 7
            #note that D is distance matrix before dynamic programming, but is cumulative distance matrix after dynamic programming.
            D = np.zeros([l1, l2])  # record image distance between image i and iamge j, i.e. distance matrix
            C = np.zeros([l1, l2])  # record total cost C from point (0,0) to (i,j)

            # --------------compute distance matrix-------------------------------------
            for i in range(l1):
                for j in range(l2):
                    if flag and (i-j>2 or i-j<-2):
                        D[i][j] = 3.0     #when apply restricted alignment, set some points to a large number directly
                    else:
                        D[i][j] = pdist([aC[ii][i],H[jj][j]], 'cosine')

            # --------------compute adaptive parameter a---------------------------------
            I3 = np.argmin(D[3, :]) #the argmin of the third row
            a = math.sqrt(1.0 + abs(I3 - 3))

            #-------------------dynamic programming--------------------------------------
            #note that after the update, D is cumulative distance matrix(i.e. matrix S in our paper)
            C[0][0] = 1
            for i in range(1, l1):
                D[i][0] = D[i][0] + D[i - 1][0]
                C[i][0] = 1 + C[i - 1][0]
            for j in range(1, l2):
                D[0][j] = D[0][j] + D[0][j - 1]
                C[0][j] = 1 + C[0][j - 1]
            for i in range(1, l1):
                for j in range(1, l2):
                    cand1 = D[i][j] + D[i - 1][j]
                    cand2 = D[i][j] + D[i][j - 1]
                    cand3 = a * D[i][j] + D[i - 1][j - 1]
                    tempMin=min(cand1,cand2,cand3)
                    if(tempMin==cand1):
                        D[i][j] = cand1
                        C[i][j] = 1 + C[i - 1][j]

                    elif(tempMin==cand2):
                        D[i][j] = cand2
                        C[i][j] = 1 + C[i][j - 1]

                    elif(tempMin==cand3):
                        D[i][j] = cand3
                        C[i][j] = a + C[i - 1][j - 1]
            D_aCH[ii][jj]=D[-1][-1]/C[-1][-1]
    return D_aCH

def cosineDistance(aC,H):
    l_aC = len(aC)
    l_H = len(H)
    D_aCH = np.zeros([l_aC, l_H])
    sta = "start to compute distance matrix D_aCH"
    for i in tqdm(range(D_aCH.shape[0]), sta):
        for j in range(D_aCH.shape[1]):
            D_aCH[i][j]=pdist([aC[i],H[j]], 'cosine')
    return D_aCH

def RandomProject(H,aC):
    print("start to reduce dimension!")
    HS, aCS = np.shape(H), np.shape(aC)
    rp_H, rp_aC = np.array(H), np.array(aC)
    rp_H, rp_aC = rp_H.reshape(HS[0] * HS[1], HS[2]), rp_aC.reshape(aCS[0] * aCS[1], aCS[2])
    reducedD = 512
    transformer = random_projection.GaussianRandomProjection(n_components=reducedD).fit(rp_H)
    H, aC = [], []
    for i in range(HS[0]):
        H.append(transformer.transform(rp_H[7 * i: 7 * i + 7]))
    for i in range(aCS[0]):
        aC.append(transformer.transform(rp_aC[7 * i: 7 * i + 7]))
    # rp_H = transformer.transform(rp_H)
    # rp_aC = transformer.transform(rp_aC)
    # H = rp_H.reshape(HS[0], HS[1], reducedD)
    # aC = rp_aC.reshape(aCS[0], aCS[1], reducedD)
    # H = H.tolist()
    # aC = aC.tolist()
    print("datasize before GRP (reference):",HS)
    print("datasize after GRP (reference):", np.shape(H))
    return H,aC