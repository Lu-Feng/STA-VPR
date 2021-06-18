# -*- coding: UTF-8 -*-
#-------------------------------------------
# the code for STA-VPR proposed in "STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition" IEEE RA-L 2021.
# Time: 2021-4-15
# Author: Feng Lu (lufengrv@gmail.com)
#-------------------------------------------
import numpy as np
from numba import cuda,float64
import numba as numba
import math
from sklearn import random_projection

@cuda.jit
def LMDTW_Process(D_CH,tempList,out):
    '''sequence matching on gpu (LM-DTW process)
    '''
    Hl=D_CH.shape[1]  # length of seq H
    Cl=20  #length of seq C
    Tl=40  #length of candidate seq T' (Notice it isn't the length of T)
    for i in range(Hl):
        tempList[i][0]=1.0   #initialize seq distance to a value >=1
    k = cuda.grid(1)
    if(k < Hl):
        # note that D is distance matrix before dynamic programming, but is cumulative distance matrix after dynamic programming.
        D = cuda.local.array((Cl, Tl), dtype=float64)  # record image distance between image i and iamge j, i.e. distance matrix
        C = cuda.local.array((Cl, Tl), dtype=float64)  # record total cost C from point (0,0) to (i,j)

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

        #-----------find the best local sebsequence of k-th candidate seq T' for matching query seq---------------
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
    out[0]=tempList[minDistanIndex][0]  #min seq distance (between seq C and T)
    out[1]=minDistanIndex  #start ID of matched result seq T
    out[2]=minDistanIndex+tempList[minDistanIndex][1]  #end ID of matched result seq T

def LMDTW(D_CH):
    '''host code for calling naive kernal
    '''
    H_len = D_CH.shape[1]
    tempList=np.zeros([H_len,2])  #for save the information (distance with query seq, length) of H_len candidate seqs.
    out = np.zeros(3)   #for save the information (distance, startID, endID) of matched result seq.
    d_D_CH = cuda.to_device(D_CH)  # d_ --> devicec
    d_tempList=cuda.device_array(tempList.shape, np.float32)
    d_out = cuda.device_array(out.shape, np.float32)
    threadsperblock =8
    LMDTW_Process[math.ceil(H_len/ threadsperblock), threadsperblock](d_D_CH,d_tempList,d_out)
    return d_out.copy_to_host()

@cuda.jit
def alignDistance_Process(aC,H,D_aCH,flag):
    '''computing aligned image distance on gpu (adaptive DTW process)
    '''
    ii, jj = cuda.grid(2)
    if ii < D_aCH.shape[0] and jj < D_aCH.shape[1]:

        l1 = 7
        l2 = 7
        #note that D is distance matrix before dynamic programming, but is cumulative distance matrix after dynamic programming.
        D = cuda.local.array((l1, l2), dtype=float64)  # record image distance between image i and iamge j, i.e. distance matrix
        C = cuda.local.array((l1, l2), dtype=float64)  # record total cost C from point (0,0) to (i,j)

        # --------------compute distance matrix-------------------------------------
        for i in range(l1):
            for j in range(l2):
                if flag and (i-j>2 or i-j<-2):
                    D[i][j] = 3.0     #when apply restricted alignment, set some points to a large number directly
                else:
                    xy = modx = mody = 0.0
                    for k in range(len(aC[ii][i])):
                        xy += aC[ii][i][k] * H[jj][j][k]
                        modx += aC[ii][i][k] * aC[ii][i][k]
                        mody += H[jj][j][k] * H[jj][j][k]
                    D[i][j] = 1 - xy / math.sqrt(modx * mody)

        # --------------compute adaptive parameter a---------------------------------
        I3value = min(D[3][0], D[3][1], D[3][2], D[3][3], D[3][4], D[3][5], D[3][6])

        I3 = 0
        for k in range(7):
            if I3value == D[3][k]:
                I3 = k
                break
        a = math.sqrt(1.0 + abs(I3 - 3))  #originalDTW set x=1

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

def alignDistance(aC, H, flag):
    '''host code for calling naive kernal
    '''
    print("start to compute distance matrix D_aCH!")
    cuda.close()
    l_aC=len(aC)
    l_H=len(H)
    print("datasize(reference):",np.shape(H),"\tdatasize(qurey):",np.shape(aC))
    d_aC = cuda.to_device(aC)  # d_ --> device
    d_H = cuda.to_device(H)
    d_DisMatD_aCH = cuda.device_array((l_aC, l_H), np.float32)       # for save distance matrix D_aCH

    threadsperblock = (8, 8)
    blockspergrid_x = math.ceil(len(aC) / threadsperblock[0])
    blockspergrid_y = math.ceil(len(H) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    alignDistance_Process[blockspergrid, threadsperblock](d_aC, d_H, d_DisMatD_aCH, flag)
    DisMatD_aCH=d_DisMatD_aCH.copy_to_host()
    cuda.close()
    return DisMatD_aCH

@cuda.jit
def cosineDistance_Process(aC,H,D_aCH):
    i, j = cuda.grid(2)
    if i < D_aCH.shape[0] and j < D_aCH.shape[1]:
        xy = modx = mody = 0.0
        for k in range(len(aC[i])):
            xy+=aC[i][k]*H[j][k]
            modx+=aC[i][k]*aC[i][k]
            mody+=H[j][k]*H[j][k]
        D_aCH[i][j]=1-xy/math.sqrt(modx*mody)

def cosineDistance(aC, H):
    '''host code for calling naive kernal
    '''
    cuda.close()
    l_aC=len(aC)
    l_H=len(H)
    d_aC = cuda.to_device(aC)  # d_ --> device
    d_H = cuda.to_device(H)
    d_DisMatD_aCH = cuda.device_array((l_aC, l_H), np.float32)   # for save distance matrix D_aCH

    threadsperblock = (8, 8)
    blockspergrid_x = math.ceil(len(aC) / threadsperblock[0])
    blockspergrid_y = math.ceil(len(H) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cosineDistance_Process[blockspergrid, threadsperblock](d_aC, d_H, d_DisMatD_aCH)
    DisMatD_aCH=d_DisMatD_aCH.copy_to_host()
    cuda.close()
    return DisMatD_aCH

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
