# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:57:20 2024

@author: victor
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import Calibrate


mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# base_options = python.BaseOptions(model_asset_path='C:/Users/victo/Desktop/RemoteScan/Programming/Depth perception/gesture_recognizer.task')
# options = vision.GestureRecognizerOptions(base_options=base_options)
# recognizer = vision.GestureRecognizer.create_from_options(options)

cam = cv2.VideoCapture(0)
camB = cv2.VideoCapture(1)

with mp_hand.Hands(max_num_hands = 1, min_detection_confidence= 0.7) as Hand:
    #
    while (cam.isOpened() and camB.isOpened()):
        success, imgT = cam.read()
        successb , imgB = camB.read()
        
        k = cv2.waitKey(5)
        
        if k == 27:
            break        
        
        
        fT = cv2.cvtColor(imgT, cv2.COLOR_BGR2RGB)
        fB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        hand_T = Hand.process(fT)
        hand_B = Hand.process(fB)
                
        if hand_T.multi_hand_landmarks:
            for handLMs in hand_T.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = float('inf')
                y_min = float('inf')

                for id, lm in enumerate(handLMs.landmark):
                    # Get the pixel coordinates of the landmark
                    h, w, c = imgT.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                # compute the center and size of the bounding box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min
                angle = 0  # the angle is always 0 for an upright bounding box
                box = ((center_x, center_y), (width, height), angle)

                mp_draw.draw_landmarks(imgT, handLMs, mp_hand.HAND_CONNECTIONS)
                
        
        if hand_B.multi_hand_landmarks:
            for handLMs in hand_B.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = float('inf')
                y_min = float('inf')

                for id, lm in enumerate(handLMs.landmark):
                    h, w, c = imgB.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                # compute the center and size of the bounding box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min
                angle = 0  # the angle is always 0 for an upright bounding box
                box = ((center_x, center_y), (width, height), angle)

                mp_draw.draw_landmarks(imgB, handLMs, mp_hand.HAND_CONNECTIONS)
        
        
        #imgT, imgB = Calibrate.undistortRectify(imgT, imgB)
        # mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgT)
        # gesture = recognizer.recognize(mp_img)
        # print(gesture.gestures[0][0])
        cv2.imshow('Top', imgT)
        cv2.imshow('Bot', imgB)
        
cam.release()
camB.release()

cv2.destroyAllWindows()