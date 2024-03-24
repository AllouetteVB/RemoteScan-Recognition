# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:37:35 2024

@author: victor
"""
import numpy as np
import cv2

import mediapipe as mp
import Point

num_hands = 1
detection_confidence = 0.7

#The Hand detection Model
class Hand():
    def __init__(self, frame_shape, cropping = False):
        self.hand_model =  mp.solutions.hands
        self.draw = mp.solutions.drawing_utils

        self.Hand = self.hand_model.Hands(max_num_hands = num_hands, min_detection_confidence = detection_confidence)
        
        self.cropping = cropping
        self.total_loop = 1
        self.success = 0
        
        TopLeft  = Point.Point([0,0],frame_shape)
        BottomRight = Point.Point([frame_shape[0], frame_shape[1]],frame_shape)
        
        self.bounding_box_points = [TopLeft, BottomRight]
    
    def process(self, frame):
        
        self.total_loop += 1
        if self.cropping:
            return self.processCrop(frame)
        
        return self.processFull(frame)
    
    
    def processCrop(self, frame):
        bb_estimate = self.findNextBoundingBox()
        offset = 20
        hasLandmark = []
        
        while not hasLandmark and offset < 50:
            cropFrame, box = self.cropFrame(frame, bb_estimate, offset)
            
            try:
                BGR_cropFrame = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2RGB)
                hand_landmark = self.Hand.process(BGR_cropFrame)
                hasLandmark = hand_landmark.multi_hand_landmarks
            except:
                print('error in cropping')
            
            if not hasLandmark:
                offset += 20
            
        if not hasLandmark and offset > 50:
           cropFrame = frame.copy()
           BGR_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           hand_landmark = self.Hand.process(BGR_frame)

           self.bounding_box_points[0].reset()
           self.bounding_box_points[1].reset()
        
        if hand_landmark.multi_hand_landmarks:
            self.draw_landmarks(frame, cropFrame, box, hand_landmark)
            self.success += 1
            
        else:
            [x_min_e, y_min_e] = self.bounding_box_points[0].positions[-1]
            [x_max_e, y_max_e] = self.bounding_box_points[1].positions[-1]
            self.draw_box(frame, x_min_e, y_min_e, x_max_e, y_max_e)
        
        return frame
    
    
    def processFull(self, frame):
        
        BGR_fullFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_landmark = self.Hand.process(BGR_fullFrame)
        h, w, c = frame.shape
        
        if hand_landmark.multi_hand_landmarks:
            self.success += 1
            self.draw_landmarks(frame, frame, (0,0,w,h), hand_landmark)

        return frame
    
    
    def findNextBoundingBox(self):
        next_estimate = []
        for bb_point in  self.bounding_box_points:
            next_estimate.append(bb_point.estimateNextPoint())
 
        return next_estimate
    
    def cropFrame(self, frame, estimation, offset):
        #We want to take a frame slightly bigger than the estimated points
        correction = [[-1,-1],[1,1]]
        cropCoord = [[],[]]
        
        h, w, c = frame.shape
        
        for i in range(len(correction)):
            x_n = estimation[i][0] + correction[i][0]*offset    
            y_n = estimation[i][1] + correction[i][1]*offset        
        
            x_n = (x_n if x_n > 0 else 0) if x_n < w else w - 1
            y_n = (y_n if y_n > 0 else 0) if y_n < h else h - 1
            
            cropCoord[0].append(x_n)
            cropCoord[1].append(y_n)
        
        min_x, min_y, max_x, max_y = min(cropCoord[0]), min(cropCoord[1]), max(cropCoord[0]), max(cropCoord[1])
        
        cropFrame = frame.copy()
        cropFrame = cropFrame[min_y:max_y,min_x:max_x]
    
        return cropFrame, (min_x, min_y, max_x, max_y)
        
        
    def draw_landmarks(self, frame, cropFrame, box, handMLM):
        h, w = box[3] - box[1], box[2] - box[0]
        
        for handLM in handMLM.multi_hand_landmarks:
            
            x_max, y_max = 0, 0
            x_min, y_min = float('inf'), float('inf')
            
            for id, lm in enumerate(handLM.landmark):
                
                x, y = box[0] + int(lm.x * w), box[1] + int(lm.y * h)
  
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max
                y_min = y if y < y_min else y_min
            
            
            self.bounding_box_points[0].addPos(x_min, y_min)
            self.bounding_box_points[1].addPos(x_max, y_max)
            self.draw_box(frame, x_min, y_min, x_max, y_max)
                
        self.draw.draw_landmarks(cropFrame, handLM, self.hand_model.HAND_CONNECTIONS)
    
    def draw_box(self, frame, x_min,y_min,x_max,y_max):
        frame = cv2.rectangle(frame, (x_min,y_min),(x_max,y_max), (0,255,0),1)
           
            
            
            
            
            
            
            