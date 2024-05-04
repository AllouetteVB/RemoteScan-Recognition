# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:23:39 2024

@author: victor
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#Plot the signatures in 2D graphs
scaling = [100,10,5]
conditions = ["Full","Mask"]



i=0


for scale in scaling:
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    axes = [ax1,ax2]
    detection_confidence = np.log10(np.sqrt(scale))/2 + 0.2
    
    
    
    i=0
    for condition in conditions:
        
        data = pd.read_csv('Signature csv/{}_{}.csv'.format(condition,scale), sep=';', index_col=0).dropna()
        
        X = -data.x
        Y = -data.y
        
        axes[i].plot(X, Y, 'b-')
        axes[i].set_title(condition + ' frame processed')
        axes[i].set_xlabel('X axis')
        axes[i].set_ylabel('Y axis')
        i+= 1
        
    
    fig.suptitle('Signature scale : '+ str(scale)+ ' detection threshold: {:.2f} smoothed'.format(detection_confidence))
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    plt.savefig('Signatures plots/Plot_{}.png'.format(scale))
    plt.show()
    
    plt.close(fig)
    
    
