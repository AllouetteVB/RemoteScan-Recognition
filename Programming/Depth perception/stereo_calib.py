# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:58:01 2024

@author: victor
"""

import numpy as np
import cv2
import glob


#
chessboardSize = (3,5)
frameSize = (1080,720)

#
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#
objp = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

#
objpoints = []
imgpointsT = []
imgpointsB = []

imagesTop = glob.glob('images/Calibration/Top/*.png')
imagesBottom = glob.glob('images/Calibration/Bottom/*.png')

for imgTop, imgBot in zip(imagesTop,imagesBottom):
    
    imgT = cv2.imread(imgTop)
    grayT = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
    
    imgB = cv2.imread(imgBot)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    
    retT, cornersT = cv2.findChessboardCorners(grayT, chessboardSize, None)
    retB, cornersB = cv2.findChessboardCorners(grayB, chessboardSize, None)
    
    if retT and retB == True:
        
        objpoints.append(objp)
        
        cornersT = cv2.cornerSubPix(grayT, cornersT, (11,11), (-1,-1), criteria)
        cornersB = cv2.cornerSubPix(grayB, cornersB, (11,11), (-1,-1), criteria)

        imgpointsT.append(cornersT)
        imgpointsB.append(cornersB)
        
        cv2.drawChessboardCorners(imgT, chessboardSize, cornersT, retT)
        cv2.imshow('Top', imgT)
        cv2.drawChessboardCorners(imgB, chessboardSize, cornersB, retB)
        cv2.imshow('Bottom', imgB)
        cv2.waitKey(1000)

cv2.destroyAllWindows()
                                        
retT, cameraMAtrixT, distT, rvecslT, tvecsT = cv2.calibrateCamera(objpoints, imgpointsT, frameSize, None, None)
heightT, widthT,channelsT = imgT.shape
newCamMatT, roi_T = cv2.getOptimalNewCameraMatrix(cameraMAtrixT, distT, (widthT, heightT), 1, (widthT, heightT))
    
retB, cameraMAtrixB, distB, rvecslB, tvecsB = cv2.calibrateCamera(objpoints, imgpointsB, frameSize, None, None)
heightB, widthB,channelsB = imgB.shape
newCamMatB, roi_B = cv2.getOptimalNewCameraMatrix(cameraMAtrixB, distB, (widthB, heightB), 1, (widthB, heightB))

#
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

#
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCamMatT, distT, newCamMatB, distB, rot, trans, essentialMat, fundamentalMat = cv2.stereoCalibrate(objpoints, imgpointsT, imgpointsB, newCamMatT, distT, newCamMatB, distB, grayT.shape[::-1], criteria_stereo, flags)
                                                                                                            
#
rectifyScale = 1
rectT, rectB, projMatT, projMatB, Q, roi_T, roi_B = cv2.stereoRectify(newCamMatT, distT, newCamMatB, distB, grayT.shape[::-1], rot, trans, rectifyScale, (0,0))

stereoMapT = cv2.initUndistortRectifyMap(newCamMatT, distT, rectT, projMatT, grayT.shape[::-1], cv2.CV_16SC2)
stereoMapB = cv2.initUndistortRectifyMap(newCamMatB, distB, rectB, projMatB, grayB.shape[::-1], cv2.CV_16SC2)



print('Saving parameters!')
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)

cv_file.write('stereoMapT_x', stereoMapT[0])
cv_file.write('stereoMapT_y', stereoMapT[1])


cv_file.write('stereoMapB_x', stereoMapB[0])
cv_file.write('stereoMapB_y', stereoMapB[1])

cv_file.release()

        
    
    
    

        
    