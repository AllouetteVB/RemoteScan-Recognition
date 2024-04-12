# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:21:21 2024

@author: victor
"""
import numpy as np
import pandas as pd
from BezierCurves import BezierCurve

import time

class Point:
    
    def __init__(self, firstPos, frame_shape, scale=10):
        """
            Create a point object that will help saving the previous position of the particular point.
            
            It is used to save the bounding box coordinate and the index finger coordinates.
            
            It should also be used to do computation on those points
        """
        self.firstPos = firstPos
        self.positions = [[None,None]]
        
        self.positions3D = [[None,None,None]]
        
        self.firstTime = time.time()
        self.times = [0]
        self.frame_shape = frame_shape
        self.scale = scale
        
    def addPoint(self, x, y, z= None):
        """
            Add a point to the list
        """
        self.positions.append([x,y])
        self.positions3D.append([x,y,z])
        
        self.times.append(time.time() - self.firstTime)
        
    def isComputable(self):
        """
            verify that the velocity and acceleration can be computed, 
            i.e that there are at least three previous frames with a hand on it 
        """
        isComputable = len(self.positions) > 3
        
        if not isComputable:
            return False
        
        for i in range(1,4):
            isComputable = isComputable and not (self.positions[-i][0] is None)
        
        return isComputable
        
    def velocity(self):
        """
            compute the velocity of the point
        """
        if not self.isComputable():
            return 0,0
        
        vx = self.positions[-1][0] - self.positions[-2][0]
        vy = self.positions[-1][1] - self.positions[-2][1]
        return vx, vy
    
    def acceleration(self):
        """
            compute the acceleration
        """
        if self.isComputable():
            return 0,0
        
        vx = self.positions[-1][0] - self.positions[-2][0]
        vy = self.positions[-1][1] - self.positions[-2][1]
        vx_ = self.positions[-2][0] - self.positions[-3][0]
        vy_ = self.positions[-2][1] - self.positions[-3][1]
        return vx - vx_, vy - vy_
    
    def gaussian(self,x, loc=0, scale=1):
        """
            compute a guassian
        """
        y = (x - loc)/scale
        return np.exp(-y*y/2)/(np.sqrt(2*np.pi)*scale)
        
    def density(self, x, vx, period, length):
        """
            Compute a density function.
            
            It is used to estimate the place where the next hand position might be using probability
            It is as reliable than computing the velocity. It need further experiment to use it well
        """
        distance = vx*period
        
        X = np.arange(0,length/self.scale,1/self.scale)        
        Y = self.gaussian(X,int(x + distance)/self.scale, 1+distance*distance/self.scale)
        return Y
        
    
    def approxNextPoint(self):
        """
            Approximate the next range of point on the x and y axis of the frame
            where the hand might be found using the density method
        """
        if not self.isComputable():
            return np.ones(self.frame_shape[0]),np.ones(self.frame_shape[1])
        
        x, y = self.positions[-1][0], self.positions[-1][1]
        vx, vy = self.velocity()
        period = time.time() - self.firstTime - self.times[-1]
        
        X,Y = self.density(x, vx, period,self.frame_shape[0]), self.density(y, vy, period, self.frame_shape[1])
        
        x_max, y_max = np.max(X), np.max(Y)
        
        if x_max !=0:
            X = X*1000/x_max    #Make sure the max is 1
        
        if y_max !=0:
            Y = Y*1000/y_max    #Make sure the max is 1

        return X, Y
    
    def reset(self):
        """
            In the case no point was found, had none to the list
        """
        self.positions.append([None,None])
        
    def save(self, filename, printSuccess=False):
        """
            Save the points coordinates in a csv file.
        """
        df = pd.DataFrame(data=self.positions3D, columns=['x','y','z'])
        time_point = pd.DataFrame(data=self.times, columns=['t'])
        
        df = pd.concat([df,time_point], axis=1)
        df.to_csv(filename + '.csv', sep=';')
        if printSuccess:
            print('File Saved')
            
    # def smoothCurve(self, dataframe):
        # """
        #   Method to compute beziers curves and smooths the points path
        # """
        # np.count_nonzero(np.isnan(a))
        # while 
            
            
            
            
        
        
        