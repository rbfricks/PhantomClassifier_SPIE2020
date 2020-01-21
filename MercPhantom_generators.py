# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:19:54 2019

@author: Rafael Fricks

Description: constructs custom image generator for phantoms, implementing 
preprocessing described in the SPIE 2020 publication.
"""
import numpy as np
import pydicom as pd 
import cv2
import sklearn.utils as sk
     
def phantom_generator_advanced(dataList, bs, lb, aug=None, **kwargs):
    wmin = 0.0
    wmax = 2211.0
    maxImgs = len(dataList)
    itr = 0
    
	# loop indefinitely
    while True:
        # initialize our batches of images and labels
        images = []
        labels = []
#        itr = 0 #enable to keep printing the first batch
        
        # keep looping until we reach our batch size
        while len(images) < bs:
		# check to see if end of list reached
            if itr == maxImgs:
				# reset the array pointer
                itr = 0
				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
                if kwargs["mode"] == "eval":
                    break
                
            img0 = pd.dcmread(dataList[itr,1])
            img1 = pd.dcmread(dataList[itr,2])
            img2 = pd.dcmread(dataList[itr,3])
            
            imOut = np.dstack((img0.pixel_array, img1.pixel_array, img2.pixel_array))
            imOut = np.clip(cv2.resize(imOut, (256, 256)), wmin, wmax)
            imOut = ((imOut-wmin)*255.0/(wmax-wmin)).astype('uint8')
			
            # extract the label
            label = dataList[itr,0]
#            cat = lb.transform([label]).argmax(axis=1)[0]
            
            case = {'image': imOut, 'category_id': [label]}

        		# apply data augmentation if available
            if aug is not None:
                case = aug(**case)

			# append to batch lists
            images.append(case['image'])
            labels.append(case['category_id'][0])
            itr = itr + 1

		# one-hot encode the labels
        labels = lb.transform(np.array(labels)) 
        
        # shuffle the sample order within batch
        if(kwargs['shuffle']==True):
            images, labels = sk.shuffle(images, labels)
            
        	# yield the batch to the calling function
        yield (np.float32(np.array(images)), labels)