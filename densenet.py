# -*- coding: UTF-8 -*-
#-------------------------------------------
# the code for STA-VPR proposed in "STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition" IEEE RA-L 2021.
# Time: 2021-4-15
# Author: Feng Lu (lufengrv@gmail.com)
#-------------------------------------------

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
import cv2
from tqdm import tqdm

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

def densenetFeature(H_range,aC_range,H_path,aC_path):
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

    aC=[]
    H=[]
    sta="start to extract feature (reference images)"
    for i in tqdm(range(H_range[0],H_range[1]),sta):
        img = cv2.imread(H_path+"Image"+"{:03d}".format(i)+".jpg")
        # img = cv2.imread(H_path + str(i) + ".jpg")
        # img = img[:, :int(0.7 * len(img[0])), :]
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = transform(img)
        x = img.unsqueeze(0)
        x = x.cuda()
        features = densenet_cifar(densenet, x)
        features = features.cpu()
        features = features[0].data.numpy()
        # featureData = np.ravel(features)
        featureData = []
        for i in range(7):
            featureData.append(np.ravel(features[:,i, :]))
        H.append(featureData)

    sta="start to extract feature (query images)"
    for i in tqdm(range(aC_range[0],aC_range[1]),sta):
        img = cv2.imread(aC_path+"Image"+"{:03d}".format(i)+".jpg")
        # img = cv2.imread(aC_path + str(i) + ".jpg")
        # img = img[:, int(0.1 * len(img[0])):, :]
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = transform(img)
        x = img.unsqueeze(0)
        x = x.cuda()
        features = densenet_cifar(densenet, x)
        features = features.cpu()
        features = features[0].data.numpy()
        # featureData = np.ravel(features)
        featureData = []
        for i in range(7):
            featureData.append(np.ravel(features[:,i, :]))
        aC.append(featureData)
    return H,aC