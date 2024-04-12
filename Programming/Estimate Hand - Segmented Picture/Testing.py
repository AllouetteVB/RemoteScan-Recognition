# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:24:21 2024

@author: victor
"""

import Camera
import FindHand

import time
import cv2
import threading

font = cv2.FONT_HERSHEY_SIMPLEX


class TestThread(threading.Thread):
    def __init__(self, cam, cropping = False):
        threading.Thread.__init__(self)
        self.cropping = cropping
        
        self.Cam = cam
        
    def run(self):
        h, w, c = self.Cam.frame.shape

        HandModel = FindHand.Hand([w, h], self.cropping)

        while self.Cam.update():
            b_loop_t = time.time() 
            frame = HandModel.process(self.Cam.frame, b_loop_t)

            fps = int(1/ (time.time() - b_loop_t))
            frame = cv2.flip(frame, 1)
            cv2.putText(frame,'fps : {}'.format(str(fps)),(10,20), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('Top' + str(self.cropping),frame)

        
        cv2.destroyWindow('Top' + str(self.cropping))
        self.Cam.stop()
        print(str(self.cropping) + ' : ',  HandModel.success / HandModel.total_loop)
        
        
        
videoComparison = "HandComparison.mp4"
cameraId = 0


CamTop = Camera.Camera('Video',videoComparison)
CamTop.launch()

Test_crop = TestThread(CamTop, True)
Test_full = TestThread(CamTop, False)

Test_crop.start()
Test_full.start()

cv2.destroyAllWindows()

