# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:47:35 2024

@author: victor
"""

import cv2
import numpy as np

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapT_x = cv_file.getNode('stereoMapT_x').mat()
stereoMapT_y = cv_file.getNode('stereoMapT_y').mat()
stereoMapB_x = cv_file.getNode('stereoMapB_x').mat()
stereoMapB_y = cv_file.getNode('stereoMapB_y').mat()

def undistortRectify(frameT, frameB):
    
    newT = cv2.remap(frameT, stereoMapT_x, stereoMapT_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    newB = cv2.remap(frameB, stereoMapB_x, stereoMapB_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    return newT, newB