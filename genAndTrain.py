# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:24:42 2020

@author: Rafael Fricks

Description: Top level demonstration script to train & evaluate the phantom 
classifier described in SPIE 2020 publication by R. Fricks et al. titled
"Automatic phantom test pattern classification through transfer 
learning with deep neural networks"
"""
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer

import pickle
import numpy as np
import cv2
import random
import os, sys
import datetime
import sklearn.utils as sk

import albumentations as A

from MercPhantom_prepareData import *
from MercPhantom_generators import *
from model_evaluations import *

from Models.trainVGG19 import *

t1 = datetime.datetime.now()

# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 1 if sys.platform == 'win32' else 1000
BS = 12 if sys.platform == 'win32' else 32
print('Batch size of ' + str(BS))
xdim = 256
ydim = 256
save = False if sys.platform == 'win32' else True
datPath = 'D:\Data\MercPhantom' if sys.platform == 'win32' else '/data/'

# Seed
SEED = 50
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Load the list of training images, then initialize the set of class
# labels in the dataset along with the testing labels
DL_train = getDataList(datPath,'Train','MP_V3.0','Instance_')
DL_train = sk.shuffle(DL_train)
NUM_TRAIN_IMAGES = len(DL_train)
print("Samples in training set: ", NUM_TRAIN_IMAGES)

DL_valid = getDataList(datPath,'Validate','MP_V3.0','Instance_')
DL_valid = sk.shuffle(DL_valid)
NUM_VAL_IMAGES = len(DL_valid)
print("Samples in validation set: ", NUM_VAL_IMAGES)

DL_test = getDataList(datPath,'Test','MP_V3.0','Instance_')
NUM_TEST_IMAGES = len(DL_test)
print("Samples in test set: ", NUM_TEST_IMAGES)

DL_Xtreme = getDataList(datPath,'ExtremeTest','MP_V3.0','Instance_')
NUM_XTEST_IMAGES = len(DL_Xtreme)
print("Samples in extreme set: ", NUM_XTEST_IMAGES)

# create the label binarizer for one-hot encoding labels, then encode
# the testing labels
labels = set(DL_train[:,0])
lb = LabelBinarizer()
lb.fit(list(labels))
n_classes = len(lb.classes_)

validLabels = lb.transform(DL_valid[:,0])
testLabels = lb.transform(DL_test[:,0])
xtestLabels = lb.transform(DL_Xtreme[:,0])

steps = 1 if sys.platform == 'win32' else (NUM_TRAIN_IMAGES//BS + 1)

#### construct the training image generator for data augmentation
### augmentation used in SPIE 2020 publication
aug2 = A.Compose([A.HorizontalFlip(p=0.5),
                  A.VerticalFlip(p=0.5),
                  A.ShiftScaleRotate(shift_limit=0.20, scale_limit=0.1, rotate_limit = 90, interpolation=cv2.INTER_CUBIC, 
                                     border_mode=cv2.BORDER_CONSTANT, value = [0,0,0], p=0.5),
                  A.RandomBrightness(limit=0.005, p=0.5),
                  A.MotionBlur(blur_limit=(3, 7), p=0.5)
#                  A.GaussNoise(var_limit = (0, 10), p=0.5)
                  ])

### alternative, milder augmentation
aug3 = A.Compose([A.HorizontalFlip(p=0.5),
                  A.VerticalFlip(p=0.5),
                  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit = 90, interpolation=cv2.INTER_CUBIC, 
                                     border_mode=cv2.BORDER_CONSTANT, value = [0,0,0], p=0.5)
#                  A.RandomBrightness(limit=0.02, p=0.5),s
#                  A.MotionBlur(blur_limit=3, p=0.5),
#                  A.GaussNoise(var_limit = (2, 10), p=0.5)
                  ])

# initialize both the training and testing image generators
trainGen = phantom_generator_advanced(DL_train, BS, lb,	aug=aug2, mode="train", shuffle=True)
validGen = phantom_generator_advanced(DL_valid, BS, lb,	aug=None, mode="train", shuffle=True)

#### Preloading a model and then modifying it ###
t2 = datetime.datetime.now()
name = 'VGG19'
[H1, model1] = train_VGG19(xdim, ydim, n_classes, trainGen, validGen, steps, NUM_EPOCHS, BS, save, name)
# model1.load_weights(name+'_modelWeight.h5')
# model1 = tf.keras.models.load_model(name+'_fullModel.h5')
t3 = datetime.datetime.now()

####### Validate Model
valGen = phantom_generator_advanced(DL_valid, NUM_VAL_IMAGES//10, lb, aug=None, mode="eval", shuffle=False) ###batch size is adjusted for evaluation mode
evaluateModel(model1,valGen,validLabels,H1,NUM_EPOCHS,lb,name="Val_"+name,make_plots=True)
t4 = datetime.datetime.now()

########## Test Model
testGen = phantom_generator_advanced(DL_test, NUM_TEST_IMAGES//10, lb, aug=None, mode="eval", shuffle=False)
evaluateModel(model1,testGen,testLabels,H1,NUM_EPOCHS,lb,name="Tst_"+name,make_plots=False)
t5 = datetime.datetime.now()

########## Extreme Samples testing
XtestGen = phantom_generator_advanced(DL_Xtreme, NUM_XTEST_IMAGES//10, lb, aug=None, mode="eval", shuffle=False)
evaluateModel(model1,XtestGen,xtestLabels,H1,NUM_EPOCHS,lb,name="Ext_"+name,make_plots=False)
t6 = datetime.datetime.now()

print(" ")
print(">>> Runtime Summary <<<")
print("Begin program at:",t1)
print("End program at:",t6)
print("Total runtime:", t6-t1)
print("Training time:", t3-t2)
print("Validation evaluation time:", t4-t3)
print("Test evaluation time:", t5-t4)
print("Extreme test evaluation time:", t6-t5)
print(">>> End Summary <<<")
print(" ")