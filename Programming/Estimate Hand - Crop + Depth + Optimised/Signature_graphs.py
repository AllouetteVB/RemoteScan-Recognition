# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:23:39 2024

@author: victor
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#Plot the signatures in 2D graphs
scaling = [100,50,25,10,8,5,2,1]
conditions = ["cropping","masking","no_change"]



i=0


for scale in scaling:
    for condition in conditions:
        data = pd.read_csv('Signature csv/signature_{}_scale_{}.csv'.format(condition,scale), sep=';', index_col=0).dropna()
        
    
        X = -data.x
        Y = -data.y
        detection_confidence = np.log10(np.sqrt(scale))/2 + 0.2
    
        fig = plt.figure(num=i)
        i+=1
        plt.plot(X,Y,c='black')
        plt.xlabel("-X pixels")
        plt.ylabel("-Y pixels")
        
        
        title = "Signature " + condition + " cropping and DetectionConfidence of {%.2f}"%(detection_confidence)
        
        plt.title(title)
        plt.savefig("Signatures plots/" + title + ".png")
        plt.close(fig)
        

