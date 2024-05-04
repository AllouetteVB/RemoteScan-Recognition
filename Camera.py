# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:39:25 2024

@author: victor
"""

import cv2

class Camera():
    
    def __init__(self, cameraName, cameraId):
        """
            The camera class, help with using the camera or saving a video, or reading a video
        
        
        """
        self.name = cameraName
        self.id = cameraId
        
    def launch(self):
        """
            First method to use after initialising the camera
            It define the camera and the first frame to use
        """
        self.cam = cv2.VideoCapture(self.id)
            
        rval, self.frame = self.cam.read()
        return self
    
    def launchSaving(self, title='signature'):
        
        frame_width = int(self.cam.get(3)) 
        frame_height = int(self.cam.get(4))  
        size = (frame_width, frame_height) 
        self.saving = True
        self.result = cv2.VideoWriter('Video/' + title + '.mp4',  
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             60, size)
            
    def update(self):
        """
            Method to use in a loop
            
            It will create the next frame at the loop condition verification
            
            while Camera.update():
                ...
        """
        key = cv2.waitKey(2)
        
        if(not self.cam.isOpened() or key == 27):
            return None
        
        rval, self.frame = self.cam.read()
        if rval:
            self.RGB_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) 

        return rval
    
    def show(self, name = None, frame = None):
        """
            Create a window to show the current picture, or the entered picture
        """
        name = name if name is not None else self.name
        
        if frame is not None:
            cv2.imshow(name, frame)
        else:
            cv2.imshow(name, self.frame)
    
    def writeVideo(self, frame = None):
 
        frame = self.frame if frame is None else frame
        self.result.write(frame)
    
    
    def stop(self, name = None):
        """
            Stop the camera, and release it
        """
        name = name if name is not None else self.name
        self.cam.release()
        
        if self.saving:
            self.result.release()
        

        cv2.destroyWindow(name)
        
        return self
        
    def drawRectangle(self, frame, x_min, y_min, x_max, y_max):
        """
            Draw a rectangle in the entered frame
        """
        frame = cv2.rectangle(frame, (x_min,y_min),(x_max,y_max), (0,255,0),1)
        return frame
        
    def drawPoints(self, frame, points):
        """
            Draw points in the entered frame
        """

        for point in points:
            frame = cv2.circle(frame, point, 2, (0,0,255), -1)
        return frame
            
    def flipFrame(self, frame):
        """
            flip the frame vertically
        """
        return cv2.flip(frame, 1)
            
    def write(self, frame, text, x, y):
        """
            write on the frame at the coordinates (x,y)
        """
        
        return cv2.putText(frame,
                    text,
                    (x,y), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,255,255),2,
                    cv2.LINE_AA)

    def imsave(self, filepath, frame):
        """
            save a frame to the filepath
        """
        
        cv2.imwrite(filepath, frame)
        
    def mask_frame(self, frame, mask):
        """
            Aplly a mask to the frame
        """
        
        return cv2.bitwise_and(frame, frame ,mask=mask)
    
    def crop_frame(self, RGB_frame, Y_range, X_range):
        """
            crop the frame 
        """
        crop = RGB_frame[Y_range[0]:Y_range[1],X_range[0]:X_range[1]]
        return crop