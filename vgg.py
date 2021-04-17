# -*- coding: UTF-8 -*-
#-------------------------------------------
# the code for STA-VPR proposed in "STA-VPR: Spatio-Temporal Alignment for Visual Place Recognition" IEEE RA-L 2021.
# Time: 2021-4-15
# Author: Feng Lu (lufengrv@gmail.com)
#-------------------------------------------

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import cv2
from tqdm import tqdm

def vggFeature(H_range,aC_range,H_path,aC_path):
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    H=[]
    aC=[]
    localN = 7  #set the number of local features (only 7 or 14 is valid)
    sta = "start to extract feature (reference images)"
    for i in tqdm(range(H_range[0], H_range[1]), sta):
        img=cv2.imread(H_path+"Image"+"{:03d}".format(i)+".jpg")
        # img = cv2.imread(H_path + str(i) + ".jpg")
        # img = img[:, :int(0.7 * len(img[0])), :]
        img = cv2.resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)[0]
        #featureData =np.ravel(features)
        featureData =[]
        m=int(features.shape[1]/localN)
        for i in range(localN):
            featureData.append(np.ravel(features[:, m*i:m*i + m, :]))
        H.append(featureData)

    sta = "start to extract feature (query images)"
    for i in tqdm(range(aC_range[0], aC_range[1]), sta):
        img = cv2.imread(aC_path + "Image" + "{:03d}".format(i) + ".jpg")
        # img = cv2.imread(aC_path + str(i) + ".jpg")
        # img = img[:, int(0.1 * len(img[0])):, :]
        img = cv2.resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)[0]
        #featureData =np.ravel(features)
        featureData =[]
        m = int(features.shape[1] / localN)
        for i in range(localN):
            featureData.append(np.ravel(features[:, m * i:m * i + m, :]))
        aC.append(featureData)
    return H,aC