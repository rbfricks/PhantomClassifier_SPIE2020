# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:14:49 2019

@author: Rafael Fricks

Description: model evaluation module
"""
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def evaluateModel(model,testGen,testLabels,H,NUM_EPOCHS,lb,name="Default",**kwargs):
    predIdxs = model.predict_generator(testGen,steps=11)
    predIdxs = np.argmax(predIdxs, axis=1)
    predIdxs = predIdxs[0:len(testLabels)]
    print(predIdxs)

    # show a nicely formatted classification report
    print("[" + name + "] evaluating network...")
    print(classification_report(testLabels.argmax(axis=1), predIdxs, 	target_names=lb.classes_))
    
    print("Confusion Matrix:")
    print(confusion_matrix(testLabels.argmax(axis=1), predIdxs))
    
    if(kwargs["make_plots"] == True):
        # plot the training loss and accuracy
        N = NUM_EPOCHS
        
        plt.style.use("ggplot")
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(0, N), H.history["loss"], label="train")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val")
        plt.title("Training Loss on Dataset [" + name + "]")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")        
        plt.savefig(name+"_lossplot.png")
        
        plt.style.use("ggplot")
        plt.figure(2)  
        plt.clf()
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val")
        plt.title("Training Accuracy on Dataset [" + name + "]")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")        
        plt.savefig(name+"_accplot.png")


def testModel_matrices(model,testGen,testLabels,lb,name="Default"):
    predIdxs = model.predict_generator(testGen,steps=11)
    predIdxs = np.argmax(predIdxs, axis=1)
    predIdxs = predIdxs[0:len(testLabels)]
    print(predIdxs)
    
    # show a nicely formatted classification report
    print("[" + name + "] evaluating network...")
    print(classification_report(testLabels.argmax(axis=1), predIdxs, target_names=lb.classes_))
#
    print("Confusion Matrix:")
    print(confusion_matrix(testLabels.argmax(axis=1), predIdxs))