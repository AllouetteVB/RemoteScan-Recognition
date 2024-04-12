# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:35:11 2024

@author: victor
"""

import cv2
import torch
import numpy as np


# "DPT_Large"      # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# "DPT_Hybrid"     # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# "MiDaS_small"      # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

cap = cv2.VideoCapture("FingerDepth.mp4")

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
   

result = cv2.VideoWriter('FingerDepth_depth_light.mp4',  
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          60, size) 

for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    if i%4 ==0:
        ret, frame = cap.read()
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        imgbatch = transform(img).to(device)
        
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = img.shape[:2],
                mode='bicubic',
                align_corners=False
                ).squeeze()
        
        output = prediction.cpu().numpy()

        
        output_show = None
        output_show = cv2.normalize(output, output_show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_show = cv2.applyColorMap(output_show, cv2.COLORMAP_BONE )
        result.write(output_show)
        # cv2.imshow('depth', output_show)
        print(i)
    
    # if i == 32: 
    #     break

    

cap.release()
result.release()


# resultBis = cv2.VideoCapture("HandComparison_depth.mp4")
# for i in range(int(resultBis.get(cv2.CAP_PROP_FRAME_COUNT))):
#     ret, frame = resultBis.read()
#     cv2.imshow('depth', frame)
    
# resultBis.release()
cv2.destroyAllWindows()