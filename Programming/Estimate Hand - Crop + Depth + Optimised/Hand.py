# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:44:42 2024

@author: victor
"""


import mediapipe as mp

num_hands = 1

class Hand:
    def __init__(self, detection_confidence = 0.7):
        """
            Model to recognize a Hand in a frame
        """
        self.hand_model = mp.solutions.hands
        self.draw = mp.solutions.drawing_utils
        
        self.Hand = self.hand_model.Hands(max_num_hands = num_hands,
                                          min_detection_confidence = detection_confidence)
        
    def process(self, RGB_frame):
        """
            process a RGB frame and return the landmarks of the hand found, none is not found
        """
        self.hand_landmark = self.Hand.process(RGB_frame)
        return self.hand_landmark.multi_hand_landmarks
    
    
    def computeHand(self, w, h):
        """
            Determine the bounding box around the hand, and the index finger point
        """
        
        for handLM in self.hand_landmark.multi_hand_landmarks:
            
            x_max, y_max = 0, 0 
            x_min, y_min = float('inf'), float('inf')
        
            for point_id, lm in enumerate(handLM.landmark):
                x, y = int(lm.x*w), int(lm.y*h)
                
                x_max = x if x > x_max else x_max
                x_min = x if x < x_min else x_min
                y_max = y if y > y_max else y_max
                y_min = y if y < y_min else y_min
                
                if point_id == 8:
                    index_x, index_y = x, y
                    
        x_max, x_min = x_max, x_min
        y_max, y_min = y_max, y_min
        
        return x_min, y_min, x_max, y_max, index_x, index_y