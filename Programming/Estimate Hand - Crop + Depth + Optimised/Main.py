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
import cv2


class TestThread(threading.Thread):
    def __init__(self, cam, scale = 10, cropping = False, masking = False, saving=False, title='None'):
        """
            Initiate a a thread to work on multiple instance at the same time.
            
            It also has the main loop in it
        """
        threading.Thread.__init__(self)
        
        self.Cam = cam      #Camera or video instance
        self.scale = scale  #Scale of the parameters. The scale determine how much the frame will be cropped or masked, and the detection confidence
        self.cropping = cropping # Determine if the frame will be cropped before computing
        self.masking = masking   # Determine if the frame will be masked before computing
        
        #If the finger points should be saved        
        self.saving = saving
        self.title  = title
        
        detection_confidence = np.log10(np.sqrt(scale))/2 + 0.2
        self.HandModel = Hand.Hand(detection_confidence) #Initiate the Hand model
        # self.MiDaSModel = Depth.Depth()                #Initiate the MiDaS model
        self.depthModel = Depth.GeometricDepth()         #Initiate the experimentally computed Depth model
        
        h,w,c = self.Cam.frame.shape #shape of frame
        
        
        #Initiate the bounding boxes and the finger points with the Point class
        box_TopLeft = Point.Point([0,0], [w,h], scale) 
        box_BottomRight = Point.Point([w,h],[w,h], scale)
        
        self.finger = Point.Point([None,None],[w,h],scale)
        self.boundingBox = [box_TopLeft, box_BottomRight] #Save the bounding box in a list
        
        # Variable to compute metrics
        self.total_frames = 1 
        self.success = 0 
        self.time = []
        
        # Variable to have a return value to the thread
        self._return = None
    
    def join(self):
        """
            Method that is used to wait the end of the thread and get the return value
        """
        threading.Thread.join(self)
        return self._return
    
    def nextBoundingbox(self):
        """
            compute the next bounding box
        """
        x_min, y_min = self.boundingBox[0].approxNextPoint()
        x_max, y_max = self.boundingBox[1].approxNextPoint()
        return x_min, y_min, x_max, y_max
        
    def run(self):
        """
            The method to launch the thread
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
            
            frame = self.Cam.mask_frame(frame, mask=mask)
            
            #Process the hand
            hasHand = self.HandModel.process(processedFrame)
            self.total_frames += 1
            
            # #Conditions that retried to process the hand on the full picture if the frame was cropped. It wrong the result so it is not used anymore
            # if not hasHand and loopSinceHand < 1:
            #     hasHand = self.HandModel.process(RGB_frame)
            #     start_x, start_y = 0, 0
            #     h, w, c = frame.shape
            
            #Compute the hand bounding box and finger points
            if hasHand:
                loopSinceHand = 0
                x_min, y_min, x_max, y_max, index_x, index_y = self.HandModel.computeHand(w,h)
                
                self.boundingBox[0].addPoint(start_x + x_min, start_y + y_min)
                self.boundingBox[1].addPoint(start_x + x_max, start_y + y_max)
                
                self.success +=1
                
                #Midas Depth 
                # #Depth computed using MiDaS, it is too slow and unreliable to used
                # if depth_count%4 == 0:
                    # depth_map = self.MiDaSModel.depth_map(RGB_frame)
                    # finger_distance = self.MiDaSModel.prediction_distance(depth_map,
                                                                      # self.finger.positions[-1])
                    # self.finger.addPoint(index_x, index_y, finger_distance)
                    # depth_count = 0
                # else:
                    # self.finger.addPoint(index_x, index_y)
                
                # self.finger.addPoint(start_x + index_x, start_y + index_y)
                
                # depth_count+=1
                
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
                frame = self.Cam.drawPoints(frame, self.finger.positions[-10:])
                
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
            frame = self.Cam.write(frame, 'Area  {:.2f}'.format(Area), 10, 70)
            condition = 'cropping' if self.cropping else ('masking' if self.masking else 'no_change')
                                   
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
        
        


handCircle = "HandCircle.mp4" #A circle draw
signature = "signature.mp4" #A signature
velocityTest = "Test_vitesse.mp4" #A video to test the next position prediction
cameraId = 0


# Check 3D
CamTop = Camera.Camera('Video', cameraId)
CamTop.launch()
Test = TestThread(CamTop, 100, saving=False, title="3D Calibration")
Test.start()




# # Check Estimation
# CamTop = Camera.Camera('Video', velocityTest)
# CamTop.launch()
# Test = TestThread(CamTop, 10, masking=True)
# Test.start()


# # # Saving video
# # Code to save a video
# CamTop = Camera.Camera('Video', cameraId, writing=True, title = 'signature')
# CamTop.launch()
# Test = TestThread(CamTop, 100)
# Test.start()

# # # Signature recognition Code
# # Code to test the video signature with different parameters
# scaling = [100, 50, 25, 10, 8, 5, 2, 1]
# conditions = [[False, False], [False,True], [True, False]]

# Threads_results = []

# for scale in scaling:
#     for condition in conditions:
#         CamTop = Camera.Camera('Video', signature)
#         CamTop.launch()
#         Test = TestThread(CamTop, scale, cropping=condition[0], masking=condition[1])
#         Test.start()
#         Threads_results.append(Test.join())

# result = pd.DataFrame(data=Threads_results, columns=['Test','Scale','Detection','AvgFramerate'])

# result.to_csv('Signature csv/Test_results.csv',sep=';',encoding='utf-8')
# print("saved")
