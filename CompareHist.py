# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:57:42 2019

@author: Rafael Fricks

Description: Generates accuracy and loss plot figures from saved training history
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

title = "test"

names = ["4096", "2048", "1024", "512", "256"]
fmts = ['-', '--', '-.', ':', '-']

pairs = list()
for i in range(0,5):
    pairs.append( [names[i], fmts[i]])

plt.style.use("ggplot")
plt.clf()
for (n, f) in pairs:
    H = pickle.load(open("VGG19_" + n + "_hist","rb"))
    N = len(H["val_accuracy"])
    
    plt.subplot(121)
    plt.plot(np.arange(0, N), H["val_accuracy"], f, label=r'$n_p = $' + n)
    
    plt.subplot(122)
    plt.plot(np.arange(0, N), H["val_loss"], f, label=r'$n_p = $' + n)


plt.subplot(121)
plt.ylim((.5, 1))    
plt.title("Classification Accuracy on Validation Set")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")        

plt.subplot(122)
plt.title("Classification Loss on Validation Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")  

# plt.savefig(title+"_accplot.png") ### uncomment to save automatically