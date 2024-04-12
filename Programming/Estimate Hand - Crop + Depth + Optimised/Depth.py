# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:25:57 2024

@author: victor
"""
import torch
from scipy.interpolate import RectBivariateSpline

import cv2
import numpy as np

class DepthMiDaS:
    
    def __init__(self):
        """
            A depth estimation model using MiDaS
            
            
            It is clear that this is copied code and we don't know how it works'
            
        """
        self.midas = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = self.transforms.small_transform
        
        self.alpha = 0.2
        self.previous_depth = 0.0
        self.depth_scale = 1.0       
        
    def depth_map(self, RGB_frame):
        """
            Create a depth map using a RGB frame
            and return a normalized depth map
        """
        frame_batch = self.transform(RGB_frame.copy()).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(frame_batch)
            prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=RGB_frame.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        depth_map_norm = cv2.normalize(depth_map,
                                         None,
                                         0, 1,
                                         norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)
        return depth_map_norm
    
    def prediction_distance(self, depth_map, point):
        """
            Use the depth map and the point we want to know the distance to to give an estimate
        """
        h, w = depth_map.shape
        
        x_grid = np.arange(w)
        y_grid = np.arange(h)
        
        spline = RectBivariateSpline(y_grid, x_grid, depth_map) #Extrapolate a depth from the depth map
        depth_filt = spline(point[1], point[0]) #Consider the depth at the pont
        depth_midas = self.depth_to_distance(depth_filt) #Compute a distance
        depth_mid_filt = (self.apply_ema_filter(depth_midas))[0][0] 
        return depth_mid_filt
        
    def depth_to_distance(self, depth_filt):
        """
            define a distance, the values used are arbitrary and needs refinement
        """
        return 1.0 / (depth_filt*self.depth_scale)
    
    def apply_ema_filter(self, depth):
        """
            Try to determine wheter the new depth is higher or not than the previous
        """
        filtered_depth = self.alpha*depth + (1-self.alpha)*self.previous_depth
        self.previous_depth = filtered_depth
        return filtered_depth


class GeometricDepth:
    
    def __init__(self):
        """
            Class that is derived from experimental testing
            
            It will work for me, but not other, it is a temporary step in creating a method that can compute realistic depth
        """
        self.previousWidth = None
        self.previousHeight = None
        self.previousArea = None
        self.distanceFromOrigin = 0
        
    def computeW(self, w):
        """
            f(w) = cte*w**(power)
            
            computed before hand
        """
        cte = 17970
        power = -(1/1.124)
        return ((1/cte)*w)**power
    
    def computeH(self, h):
        """
            f(h) = cte*h**(power)
            
            computed before hand
        """
        cte = 14633
        power = -(1/1.05)
        return ((1/cte)*h)**power
    
    def computeA(self, area):
        """
            f(h) = cte*h**(power)
            
            computed before hand
        """
        cte = 31026
        power = -(1/1.923)
        return ((1/cte)*area)**power
        
    def addPreviousLength(self, width, height):
        """
            Save previous distance
        """
        self.previousWidth = width
        self.previousHeight = height
        self.previousArea = width*height/1000
    
    def computeDepth(self, width, height):
        """
            compute the depth using the width and height
        """
       
        if self.previousArea is None:
            dA = 0
        else:
            dA = self.computeA(width*height/1000)
           
       
        # if self.previousWidth is None:
        #     dw = 0
        #     dh = 0
        # else:
        #     dw = self.computeW(width)
        #     dh = self.computeH(height)
        
        self.addPreviousLength(width, height)
        
        return dA      