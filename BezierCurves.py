# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 06:58:01 2024

@author: victor
"""
import bezier
import simpy

class BezierCurve:
    def __init__(self):
        return
    
    def fit(self, points):
        self.curve = bezier.Curve(points, degree=points.shape[1]-1)
    
    def nodes(self):
        return self.curve.nodes
    
    def to_symbolic(self):
        return self.curve.to_symbolic()
    
    def evaluate(self, Y_point):
        return self.curve.evaluate(Y_point)
    
    def evaluate_multi(self, Y_points):
        return self.curve.evaluate_multi(Y_points)
    
    def plot(self, num_points):
        self.curve.plot(num_points)
 