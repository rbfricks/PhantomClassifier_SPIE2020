# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:54:22 2019

@author: Rafael Fricks
"""
import numpy as np
import tensorflow as tf
import sys

import tensorflow.keras.optimizers as optm
from tensorflow.keras import applications
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model 
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import pickle

MP = False if sys.platform == 'win32' else True

def train_VGG19(xdim, ydim, classes, trainGen, valGen, steps, NUM_EPOCHS, bs, save=False, name = "Default"):
    print("[" + name + "] training w/ generator...")
    
    # Seed
    SEED = 50
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    opt = optm.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10E-8, decay=0.001, amsgrad=False)
    model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (xdim, ydim, 3))
        
    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(512, use_bias = False, kernel_initializer=initializers.he_normal(seed=SEED))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = Dense(512, use_bias = False, kernel_initializer=initializers.he_normal(seed=SEED))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    predictions = Dense(classes, activation="softmax")(x)
        
    # creating the final model 
    model_final = Model(inputs = model.input, outputs = predictions)
    print(model_final.summary())
     
    checkpointer = ModelCheckpoint(filepath=name+'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
    tbr = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    
    # compile the model 
    model_final.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

    H = model_final.fit_generator(
    	trainGen,
    	steps_per_epoch= steps,
    	validation_data=valGen,
    	validation_steps= steps,
    	epochs=NUM_EPOCHS,
    use_multiprocessing=MP,
    verbose = 1)#,
    #callbacks=[tbr])
    
    
    if(save==True):
        print("\nSaving model: " + name)
        model_final.save_weights(name+'_modelWeight.h5')
        
        model_final.save(name+'_fullModel.h5')

        with open(name+'_architecture.json', 'w') as f:
            f.write(model_final.to_json())

        with open(name+'_hist', 'wb') as file_pi:
            pickle.dump(H.history, file_pi)
            
        print('\nModel saved!\n')
    else:
        print('\nModel not saved!\n')   
        
    return H, model_final