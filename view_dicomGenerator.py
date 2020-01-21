# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:53:01 2019

@author: Rafael Fricks

Description: data set and augmentation visualization
"""
import numpy as np
import cv2
import os, sys
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sklearn.utils as sk

import albumentations as A

from MercPhantom_prepareData import *
from MercPhantom_generators import *

##### TESTING MODULE #####        
datPath = 'D:/Data/MercPhantom' if sys.platform == 'win32' else '/data/'
DL_train = getDataList(datPath,'Train','MP_V3.0','Instance_')
BS = 32

labels = set()

# extract the class label, and update the labels list
for d in DL_train:
#	# the total number of training images
	labels.add(d[0])

NUM_TRAIN_IMAGES = len(DL_train)

lb = LabelBinarizer()
lb.fit(list(labels))

aug = ImageDataGenerator(rotation_range=60, zoom_range=0.25,
	width_shift_range=0.2, height_shift_range=0.2, shear_range= 5,
	horizontal_flip=False, fill_mode="nearest")

aug2 = A.Compose([A.HorizontalFlip(p=0.5),
                  A.VerticalFlip(p=0.5),
                  A.ShiftScaleRotate(shift_limit=0.20, scale_limit=0.1, rotate_limit = 90, interpolation=cv2.INTER_CUBIC, 
                                     border_mode=cv2.BORDER_CONSTANT, value = [0,0,0], p=0.5),
                  A.RandomBrightness(limit=0.0005, p=0.5),
                  A.MotionBlur(blur_limit=(3, 7), p=0.5)
#                  A.GaussNoise(var_limit = (0, 10), p=0.5)
                  ])
                  
aug3 = A.Compose([A.HorizontalFlip(p=0.5),
                  A.VerticalFlip(p=0.5),
                  A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.005, rotate_limit = 90, interpolation=cv2.INTER_CUBIC, 
                                     border_mode=cv2.BORDER_CONSTANT, value = [0,0,0], p=0.5)
#                  A.RandomBrightness(limit=0.02, p=0.5),
#                  A.MotionBlur(blur_limit=3, p=0.5),
#                  A.GaussNoise(var_limit = (2, 10), p=0.5)
                  ])

# DL_train = sk.shuffle(DL_train)
#trainGen = phantom_generator_classic(DL_train, BS, lb,	mode="train", aug=None)
trainGen = phantom_generator_advanced(DL_train, BS, lb,	aug=None, mode="train", shuffle=False)

# DL_test = getDataList(datPath,'ExtremeTest','MP_V3.0','Instance_')
# NUM_TEST_IMAGES = len(DL_test)
# trainGen = phantom_generator_advanced(sk.shuffle(DL_test), NUM_TEST_IMAGES//10, lb, aug=None, mode="eval", shuffle=True)

fig = plt.figure(1)
plt.clf()
plt.ion()

inc = 0
for i in range (0,1):
    print('\n Printing batch ' + str(i) + ' \n')
    x,y = next(trainGen)
    inc = i*BS
    
    imgs = np.asarray(x)
    labels = lb.inverse_transform(y)


    plt.clf()
    axs = fig.subplots(int(BS/8),4)
    plt.rcParams.update({'axes.titlesize' : 10})
    i = 0
    #for i in range(0,BS):
    for ax in axs.flat:
    #    ax = axs[i]#plt.subplot(BS/4, 4, i+1, frameon = False)
        ax.axis('off')
    #    ax.imshow(np.squeeze(imgs[i,:,:,1])/255.0, cmap=plt.cm.gray)
        ax.imshow(np.squeeze(imgs[i][:,:,1])/255.0, cmap=plt.cm.gray)
        
        title = ax.set_title(str(inc+i) + ': ' + str(labels[i]))
    #    title.set_y(1.05)
        i = i + 1
        plt.pause(0.1) #this one for image-by-image pause
        
    plt.pause(3) #this one for batch-by-batch pause
    