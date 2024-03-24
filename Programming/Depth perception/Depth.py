# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:55:57 2024

@author: victor
"""

import cv2
import numpy as np


minDisparity = 0;
numDisparities = 64;
blockSize = 8;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;
 
# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )


cam = cv2.VideoCapture(0)
camB = cv2.VideoCapture(1) #1280.720p

while (cam.isOpened() and camB.isOpened()):
    success, imgT = cam.read()
    successb , imgB = camB.read()
    
    k = cv2.waitKey(5)
    
    if k == 27:
        print(imgT.shape, imgB.shape)
        break
    
    disp = stereo.compute(imgT, imgB).astype(np.float32)
    disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)
     
    # Displaying the disparity map
    cv2.imshow("disparity",disp)
    
    
cam.release()
camB.release()

cv2.destroyAllWindows()