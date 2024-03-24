# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:46:25 2024

@author: victor
"""

import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

num = 0

while cap0.isOpened():
    
    success0, img0 = cap0.read()
    success1, img1 = cap1.read()
    
    k = cv2.waitKey(5)
    
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('images/Calibration/Top/img' + str(num) + '.png', img0)
        cv2.imwrite('images/Calibration/Bottom/img' + str(num) + '.png', img1)
        
        print('images saved!')
        num += 1 
        
    cv2.imshow('Top', img0)
    cv2.imshow('Bottom', img1)
    
cap0.release()
cap1.release()

cv2.destroyAllWindows()