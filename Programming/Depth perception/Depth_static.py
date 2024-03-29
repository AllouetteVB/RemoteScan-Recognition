# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:23:44 2024

@author: victor
"""
import cv2

import numpy as np

# Reading the left and right images.
  
imgL = cv2.imread("left_side.jpg",0)
imgR = cv2.imread("right_side.jpg",0)

imgL = cv2.resize(imgL.copy(), (1080, 720))
imgR = cv2.resize(imgR.copy(), (1080, 720))

"""
cv2.imshow("image",r_L)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# Setting parameters for StereoSGBM algorithm
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
 
# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(imgL, imgR).astype(np.float32)
disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)
 
# Displaying the disparity map
cv2.imshow("disparity",disp)
cv2.waitKey(0)
