# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:02:16 2024

@author: victor
"""

import cv2

class Camera():
    
    def __init__(self, cameraName, cameraId):
        self.name = cameraName
        self.id = cameraId
        
    def launch(self):
        self.cam = cv2.VideoCapture(self.id)
        rval, self.frame = self.cam.read()
        return self
            
    def update(self):
        key = cv2.waitKey(1)
        
        if(not self.cam.isOpened() or key == 27):
            return None
        
        rval, self.frame = self.cam.read()
        return rval
    
    def stop(self):
        self.cam.release()
        return self
        

        