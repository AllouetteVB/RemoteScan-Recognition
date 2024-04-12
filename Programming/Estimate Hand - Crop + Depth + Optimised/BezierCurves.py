# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 06:58:01 2024

@author: victor
"""
import bezier 

class BezierCurve:
    def __init__(self):
        return
    
    def fit(self, points):
        self.curve = bezier.Curve(points, degree=2)
    
    def evaluate(self, Y_point):
        return self.curve.evaluate(Y_point)
 