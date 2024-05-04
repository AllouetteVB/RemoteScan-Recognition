# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 03:03:14 2024

@author: victor
"""

# Full, Crop, Mask
# Scaling


# Mask Simple-Complex

# Signature csv/Crop_picture_scaling_2_confidence_changed_density.csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


results = pd.read_csv('Signature csv/Signature_LiveDemo.csv', sep=';')

data = results
X = -data.x
Y = -data.y

plt.plot(X, Y, 'b-')
plt.title('Live demo signature')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Type = []
# for i, result in results.iterrows():
#     Title = 'F' if result.Test == 'no_change' else ('M' if result.Test == 'masking' else 'C')
#     Title += '_' + ('DTC' if result.DetectionThresholdChange else 'DTF')
#     Title += '_' + ('DMS' if result.MaskingMethod == 'Simple' else 'DMD')

#     Type.append(Title)

# results['Type'] = Type

# results.to_csv('Signature csv/Test_results.csv', sep=';')