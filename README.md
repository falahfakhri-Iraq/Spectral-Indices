# Spectral-Indices
Introduction summary:
In order to investigate any part of the world, the first idea comes to mind is, the spectral indices, specially in low cloud cover study area, these indices give very good indicator about the land cover, and land use, as well as take a look at the true colour RGB image beside the indices is very important to consist the first idea about the area of interest. 

Instructions:
To use this code, please follow these statements, 
1-	Make sure to install all the following libraries, 
… 
    import sys
    from termcolor import colored
    import warnings
    from sklearn.exceptions import DataConversionWarning
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    import time 
    import os
    import glob

2-	Set the structure of your folder as below, 
…
   -Main folder
      --inputs 
         - image.tif

     -- outputs
    -- spectral_indices.py

3-	Set the line 257, DATA_PATH to your folder path. 
