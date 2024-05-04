# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 05:36:52 2024

@author: victor
"""
import cv2
import numpy as np
import time 


cam = cv2.VideoCapture('VideoTop.mp4')
camB = cv2.VideoCapture('VideoBot.mp4') 

numDisparities_options = [64]  # Needs to be multiple of 16
blockSize_options = [9]  # Typical values are odd numbers: 3, 5, 7, ...

minDisparity = 0;
numDisparities = 64;
blockSize = 8;

disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;

# Loop through each combination of parameters
for numDisparities in numDisparities_options:
    for blockSize in blockSize_options:
        # Reset the video capture
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        camB.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        Disparity_save = cv2.VideoWriter('Disparity_numD_{}_blocksize_{}.mp4'.format(numDisparities,blockSize),  
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              60, (640,480))
        
        # Create a StereoSGBM object with current parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity = minDisparity,
            numDisparities = numDisparities,
            blockSize = blockSize,
            disp12MaxDiff = disp12MaxDiff,
            uniquenessRatio = uniquenessRatio,
            speckleWindowSize = speckleWindowSize,
            speckleRange = speckleRange
        )
        
        num=0
        while True:
            
            b_loop_t = time.time()
            success, right_frame = cam.read()
            successb , left_frame = camB.read()
            
            if not success:
                break
            
            
            k = cv2.waitKey(1)
            
            if k == 27:
                # print(imgT.shape, imgB.shape)
                break
            
            
            # Convert images to grayscale as required by the algorithm
            gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # Calculate disparity
            disparity = stereo.compute(gray_left, gray_right).astype(np.float32)

            # Normalize the disparity for visualization
            disp_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_display = cv2.applyColorMap(disp_display, cv2.COLORMAP_BONE )

            
            if time.time() - b_loop_t > 0:
                fps = int(1/ (time.time() - b_loop_t))
            else:
                fps = float('inf') 
                
            disp_display = cv2.putText(disp_display,
                        str(fps),
                        (10,20), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(255,255, 0),2,
                        cv2.LINE_AA)
            
            
            if k == ord('s'):
                cv2.imwrite('images/Calibration/Disp_' + str(num) + '.png', disparity)
                
                
                print('images saved!')
                num += 1 
                
            # Display the disparity
            # cv2.imshow('Disparity Map', disp_display)
            Disparity_save.write(disp_display)
        print(f'Tested with numDisparities={numDisparities}, blockSize={blockSize}')
        

cam.release()
camB.release()
Disparity_save.release()

cv2.destroyAllWindows()
