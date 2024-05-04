# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:43:01 2024

@author: victor
"""

import threading
import time 

import Camera

import Hand
import Depth

import Point

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2


class TestThread(threading.Thread):
    def __init__(self, cam, scale = 10, cropping = False, masking = False, savingCSV=False, savingVideo=False, title='None'):
        """
        Initialisation of the Thread.
        
        The thread initialise the Hand detection model with the detection confidence.
        It also initialise the Depth estimation model if it exist.
        
        It initialise Three Points object to save the paths of the hand's bounding box and the finger path.

        Parameters
        ----------
        cam : Camera class
            The video feed. Either live feed or video feed
        scale : INT, optional
            Parameter related to the cropping or masking of the frame. 
            It also change the detection confidence of the hand. 
            The higher the scaling, the higher the detection threshold.
            The default is 10.
        cropping : BOOL, optional
            Define if the picture should be cropped before processing. The default is False.
        masking : BOOL, optional
            Define if the picture should have a mask before processing. The default is False.
        saving : BOOL, optional
            Define if the hand path should be saved into a csv file. The default is False.
        title : STRING, optional
            Define the csv file name. The csv will be saved in the "Signature csv" folder.
            The default is 'None'.

        Returns
        -------
        None.

        """
        #Parent Thread initialisation
        threading.Thread.__init__(self)
        
        #Initialisation of camera element and frame processing
        self.Cam = cam      
        self.scale = scale
        self.cropping = cropping
        self.masking = masking
        h,w,c = self.Cam.frame.shape #shape of frame
        
        #Initialisation of the differents paths
        box_TopLeft = Point.Point([0,0], [w,h], scale) 
        box_BottomRight = Point.Point([w,h],[w,h], scale)
        self.finger = Point.Point([None,None],[w,h],scale)
        
        self.boundingBox = [box_TopLeft, box_BottomRight]
        
        
        #Define if the hand path should be saved.        
        self.saving = savingCSV
        self.title  = title
        
        #Define if the hand video should be saved
        self.savingVideo = savingVideo
        self.Cam.launchSaving(self.title)
        
        #Initialise the Hand Model
        detection_confidence = np.log10(np.sqrt(scale))/2 + 0.2
        # detection_confidence = 0.7
        self.HandModel = Hand.Hand(detection_confidence)
        
        #Initialise the Depth Model
        # self.MiDaSModel = Depth.Depth() #MiDaS model
        self.depthModel = Depth.PersonalGeometricDepth()     #A personnaly computed depth. Cannot work for other hand. Should be modified.
        
        # Variable to compute metrics
        self.total_frames = 1 
        self.success = 0 
        self.time = []
        
        # Variable to have a return value to the thread
        self._return = None
    
    def join(self):
        """
            Method to wait for the thread to finish, and return a value

        Returns
        -------
        self._return : List
            Metrics computed in the main loop.

        """
        threading.Thread.join(self)
        return self._return
    
    def nextBoundingbox(self):
        """
        The method calculate the next bounding box. It is only an estimation.
        It is based on the normal distribution aroung where the next box should be.
        As shuch, it return an interval where the hand could be.

        Returns
        -------
        x_min : np.array, dtype = int
            Range where the top left x wise bounding box corner could be.
        y_min : np.array, dtype = int
            Range where the top left y wise bounding box corner could be.
        x_max : np.array, dtype = int
            Range where the bottom right x wise bounding box corner could be.
        y_max : np.array, dtype = int
            Range where the bottom right y wise bounding box corner could be.

        """
        x_min, y_min = self.boundingBox[0].approxNextPoint()
        x_max, y_max = self.boundingBox[1].approxNextPoint()
        return x_min, y_min, x_max, y_max
        
    def run(self):
        """
        Main loop of a Thread.
        
        The thread load the video feed and apply the different functions necessary to detect a signature.
        The thread save some metrics that can be return from the join method.

        Returns
        -------
        None.

        """
        

        finger_distance = 0 # initial distance at 0
        loopSinceHand = 2   # Saving number of frame since hand recognized
        picture_failed = 0  # Saving the number of frame where recognition should have been possible
        Area = 0
        
        while self.Cam.update():
            b_loop_t = time.time() 
            
            
            # Loop frame metrics
            frame = self.Cam.frame
            RGB_frame = self.Cam.RGB_frame
            h, w, c = frame.shape
            
            
            # # # Computation of the next bounding box
            X_min,Y_min,X_max,Y_max = self.nextBoundingbox() #range computed using density fonction
            
            #Consider which range is the highest
            X_range = np.sort([np.argmax(X_min > 0.05), w - np.argmax(X_max[::-1] > 0.05)]) 
            Y_range = np.sort([np.argmax(Y_min > 0.05), h - np.argmax(Y_max[::-1] > 0.05)])
            
            #The handModel return a value of x and y between 0 and 1 dependant of the frame, as such, when cropping, it wont give a normal resut
            start_x, start_y = X_range[0] if self.cropping else 0, Y_range[0] if self.cropping else 0 
            
            #Adding the two range as one
            X = np.add(X_min, X_max)
            Y = np.add(Y_min, Y_max)
            
            #Fill the range between the min and max points of the bounding box
            X[X_range[0]:X_range[1]] = 1
            Y[Y_range[0]:Y_range[1]] = 1
            
            #Matrix multiplication and creating a mask
            mask = np.einsum('i,j->ji', X, Y)
            mask[mask > 0.025] = 255 
            mask[mask != 255] = 0
            mask = mask.astype(np.uint8)
            
            
            #Define the frame type to use for computation (cropped, masked, full)
            if self.cropping and not self.masking:
                processedFrame = self.Cam.crop_frame(RGB_frame, Y_range, X_range)
                
                start_x, start_y = X_range[0], Y_range[0]
                h, w, c = processedFrame.shape
            elif self.masking and not self.cropping:
                processedFrame = self.Cam.mask_frame(RGB_frame, mask=mask)
                start_x, start_y = 0, 0
            else:
                processedFrame = RGB_frame
                start_x, start_y = 0, 0                
            
            # frame = self.Cam.mask_frame(frame, mask=mask)
            
            #Process the hand
            hasHand = self.HandModel.process(processedFrame)
            self.total_frames += 1
            
            #Compute the hand bounding box and finger points
            if hasHand:
                loopSinceHand = 0
                x_min, y_min, x_max, y_max, index_x, index_y = self.HandModel.computeHand(w,h)
                
                self.boundingBox[0].addPoint(start_x + x_min, start_y + y_min)
                self.boundingBox[1].addPoint(start_x + x_max, start_y + y_max)
                
                self.success +=1
                
                #Geometrical Depth
                #Distance computed using a model fitted for my hand... Only used for testing purpose waiting for a better depth model.
                box_w, box_h = x_max - x_min, y_max - y_min
                finger_distance =  self.depthModel.computeDepth(box_w, box_h)
                self.finger.addPoint(start_x + index_x, start_y + index_y, finger_distance)

                
                #Draw the bounding box
                frame = self.Cam.drawRectangle(frame, start_x + x_min,
                                               start_y + y_min,
                                               start_x + x_max,
                                               start_y + y_max)
                
                Area = box_w*box_h/1000
                #Draw the path
                frame = self.Cam.drawPoints(frame, self.finger.positions[-30:])
                
            #In case the hand was not found but should have been, save the frame. Used for further studying
            # elif False and not hasHand and loopSinceHand < 1:
            #     filepath = "PictureFailed/img_squared_{}.png".format(picture_failed)
            #     picture_failed += 1
            #     x_min, y_min = self.boundingBox[0].positions[-1]
            #     x_max, y_max = self.boundingBox[1].positions[-1]
                
                
            #     self.boundingBox[0].reset()
            #     self.boundingBox[1].reset()
                
            #     frame = self.Cam.drawRectangle(frame, start_x + x_min,
            #                         start_y + y_min,
            #                         start_x + x_max,
            #                         start_y + y_max)                
                
            #     self.Cam.imsave(filepath, frame)
            #     loopSinceHand += 1
            
            #Reset the finger points
            else:
                loopSinceHand += 1
                self.boundingBox[0].reset()
                self.boundingBox[1].reset()
                self.finger.reset()
                
            
            #Compute computation time
            self.time.append(time.time() - b_loop_t)
            
            if time.time() - b_loop_t > 0:
                fps = int(1/ (time.time() - b_loop_t))
            else:
                fps = float('inf')
            
            #Write Metrics on frame
            frame = self.Cam.flipFrame(frame)
            frame = self.Cam.write(frame, 'fps : {}'.format(str(fps)),
                            10,
                            20)
            frame = self.Cam.write(frame, 'Distance  {:.2f}'.format(finger_distance), 10, 45)
            # frame = self.Cam.write(frame, 'Area  {:.2f}'.format(Area), 10, 70)
            condition = 'cropping' if self.cropping else ('masking' if self.masking else 'no_change')
                                   
            if self.savingVideo:
                self.Cam.writeVideo(frame)
                
            self.Cam.show("Estimation_{}_scale_{}".format(condition,self.scale), frame)


        self.Cam.stop("Estimation_{}_scale_{}".format(condition,self.scale))
        
        # print('')
        # print('The percentage of detected hand in scale {} was: '.format(str(self.scale)), self.success * 100 / self.total_frames)
        # print('The average time of processing scale {} was: '.format(str(self.scale)), sum(self.time) / len(self.time))
        # print('The average framerate of scale {} was: '.format(str(self.scale)), len(self.time) / sum(self.time))
        
        
        
        #Return the metrics of the loop
        self._return = [condition, self.scale, "%.2f"%(self.success * 100 / self.total_frames), "%.2f"%(len(self.time) / sum(self.time))]
        
        
        if self.saving:
            self.finger.save('Signature csv/' + self.title)
        
        


handCircle = "Video/HandCircle.mp4" #A circle draw
signature = "Video/SignatureBenchmarck.mp4" #A signature
velocityTest = "Video/Test_vitesse.mp4" #A video to test the next position prediction
cameraId = 0


# Live test
CamTop = Camera.Camera('Video', cameraId)
CamTop.launch()
Test = TestThread(CamTop, 10, savingCSV=True, savingVideo=True, title="Signature_LiveDemo", cropping=False, masking=True)
Test.start()
# Test.join()


# # Check Estimation
# CamTop = Camera.Camera('Video', velocityTest)
# CamTop.launch()
# Test = TestThread(CamTop, 10, masking=True)
# Test.start()


# # # Saving video
# # Code to save a video
# CamTop = Camera.Camera('Video', cameraId)
# CamTop.launch()
# Test = TestThread(CamTop, 100, savingVideo=True, title = 'signature')
# Test.start()

# # Signature recognition Code
# Code to test the video signature with different parameters
scaling = [100,  10, 5]

# conditions = [[False, False], [False,True], [True, False]]
# conditions = [[False, False], [False,True]]

# Threads_results = []

# for scale in scaling:
#     for condition in conditions:
#         CamTop = Camera.Camera('Video', signature)
#         CamTop.launch()
        
#         condition_name = 'Crop' if condition[0] else ('Mask' if condition[1] else 'Full') 
#         title = condition_name + '_'+str(scale)+'_Smooth'
#         Test = TestThread(CamTop, scale, cropping=condition[0], masking=condition[1], 
#                           savingCSV=True, title=title, 
#                           )
#         Test.start()
#         Threads_results.append(Test.join())

# result = pd.DataFrame(data=Threads_results, columns=['Test','Scale','Detection','AvgFramerate'])

# # result.to_csv('Signature csv/Test_results_scaling_unchanged_estimate_complexe.csv',sep=';',encoding='utf-8')
# print("saved")
