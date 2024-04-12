# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 23:06:05 2024

@author: victor
"""

#A class to follow points.
#It is used to define the bounding box for cropping
#And it could be slightly refactored to work for the finger points 
class Point:
    def __init__(self, firstPos, frame_shape):
        self.firstPos = firstPos
        self.positions = [firstPos]
        
        self.shape = frame_shape
        self.isEstimate = False
        
        self.times = [0]
        
        
    def addPos(self,x,y, time, isEstimate = False):
        self.times.append(time)
        self.positions.append((x,y))
        self.isEstimate = isEstimate
        
    def velocity(self):
        # t, t_ = self.times[-1], self.times[-2]
        # vx = int((self.positions[-1][0] - self.positions[-2][0])/(t - t_))
        # vy = int((self.positions[-1][1] - self.positions[-2][1])/(t - t_))
        
        vx = self.positions[-1][0] - self.positions[-2][0]
        vy = self.positions[-1][1] - self.positions[-2][1]
        return vx, vy
    
    def acceleration(self):
        # t, t_, t__ = self.times[-1], self.times[-2], self.times[-3]
        # vx = int((self.positions[-1][0] - self.positions[-2][0])/(t - t_))
        # vy = int((self.positions[-1][1] - self.positions[-2][1])/(t - t_))
        # vx_ = int((self.positions[-2][0] - self.positions[-3][0])/(t_ - t__))
        # vy_ = int((self.positions[-2][1] - self.positions[-3][1])/(t_ - t__))
        
        vx = self.positions[-1][0] - self.positions[-2][0]
        vy = self.positions[-1][1] - self.positions[-2][1]
        vx_ = self.positions[-2][0] - self.positions[-3][0]
        vy_ = self.positions[-2][1] - self.positions[-3][1]
        return vx - vx_, vy - vy_
    
    def estimateNextPoint(self):
        if len(self.positions) < 4:
            return self.positions[0]
        
        
        vx, vy = self.velocity()
        ax, ay = self.acceleration()
        x_n, y_n = self.positions[-1][0] + vx, self.positions[-1][1] + vy
        
        
        #Be sure not to not overshoot the estimation
        w, h = self.shape[0], self.shape[1]
        x_n = (x_n if x_n > 0 else 0) if x_n < w else w - 1
        y_n = (y_n if y_n > 0 else 0) if y_n < h else h - 1
        
        return x_n, y_n
    
    def reset(self):
        self.positions = [self.firstPos]

    def getPath(self):
        return self.positions[-20:-1]