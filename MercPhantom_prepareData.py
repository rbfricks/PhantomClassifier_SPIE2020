# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:39:27 2019

@author: Rafael Fricks

Description: Reads file directory containing phantom images, returns a list
of associated inputs. Inputs are expected as 
<path-to-data>\PhantomModel\Set\Series########
"""

import os
import glob
import numpy as np

#whichSet = 'Train'
#whichPhnt = 'MP_V3.0'
#base = 'Instance_'
def getDataList(datPath, whichSet, whichPhnt, base):
    input_dir = os.path.join(datPath,whichPhnt,whichSet)
    outpt_dir = os.path.join(datPath,whichPhnt,whichSet)
    
    inpts = []
    outpts = []
    dataList = []
    
    for root, dirs, files in os.walk(input_dir):
        if dirs:
            for d in dirs:
                local = os.path.join(outpt_dir, d)
                #os.makedirs(local, exist_ok = True)
                outpts.append(local)
        else:
            inpts.append(root)
            print('\nAdded: ' + root)

    for inp in inpts:
        root, dirs, files in os.walk(inp)
        dcmCount = len(glob.glob1(inp,"*.dcm"))
        f = open(os.path.join(inp, 'labels.csv'), 'r')
        f.readline()
    
        for i in range(1,dcmCount+1):
            line = f.readline()
            line = line.strip().split(",")
            labl = line[1]
            
            if (i-1 < 1):
                prv = 1
            else:
                prv = i-1
                
            if (i+1 > dcmCount):
                nxt = dcmCount
            else:
                nxt = i+1   
                    
            
            dir0 = os.path.join(inp, (base + str(prv) + '.dcm'))
            dir1 = os.path.join(inp, (base + str(i) + '.dcm'))
            dir2 = os.path.join(inp, (base + str(nxt) + '.dcm'))
            
            dataList.append( [labl, dir0, dir1, dir2])
            
        f.close()
        
    return np.array(dataList)
