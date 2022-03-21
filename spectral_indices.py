# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 07:43:26 2020

@author: FALAH FAKHRI

Module: spectral_indices.py
==================================================================================================================
This code is an investigator of any low clouds coverage study area, using optical data (Images), such as
sentinel – 2. It’s used in order to investigate the characteristics spectral behaviour of any study area,
to get a rapid glance, of the objects in the study area, the geographic distribution and also the relationship 
between them. 
The spectral indices are, NDVI, DBI, SAVI, MNDWI, NDBI, this piece of code plot and save all the results of indices
as *.tif, files in separate folder, additionally this code presents the true colour RGB image.
Please have a look to the README associated with this module for more information. 
==================================================================================================================
"""

# Import all required libraries, packages and test them
try:
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
except ModuleNotFoundError:
    print('Module improt error')
    sys.exit()
else:
    print(colored('\nAll libraries properly loaded. Ready to start!!!', 'green'), '\n')


# Disable all warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Warning used to notify implicit data conversions happening in the code.
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')


def load_raster(input_file):
    """Returns a raster array which consists of its bands and transformation matrix
    parameters
    ----------
    input_file: str
        path directory to the raster file
    sensor: str
        choose between landsat and senstinel2
    reproject: if true, reprojects the data into WGS 84
    """
    with rasterio.open(input_file) as src:
        band = src.read()
        transform = src.transform
        crs = src.crs
        shape = src.shape
        profile = src.profile
        raster_img = np.rollaxis(band, 0, 1)

        return {
            'band': band,
            'raster_img': raster_img,
            'transform': transform,
            'crs': crs,
            'shape': shape,
            'profile': profile
            }
    
    
def write_raster(raster, crs, transform, output_file):

    profile = {
        'driver': 'GTiff',
        'compress': 'lzw',
        'width': raster.shape[0],
        'height': raster.shape[1],
        'crs': crs,
        'transform': transform,
        'dtype': raster.dtype,
        'count': 1,
        'tiled': False,
        'interleave': 'band',
        'nodata': 0
    }

    profile.update(
        dtype=raster.dtype,
        height=raster.shape[0],
        width=raster.shape[1],
        nodata=0,
        compress='lzw')

    with rasterio.open(output_file, 'w', **profile) as out:
        out.write_band(1, raster) 

        
def NDVI(nir,red):
    """Calculates NDVI index

    parameters
    ----------
    nir: NIR band as input
    red: RED band as input
    """
    NDVI = (nir.astype('float') - red.astype('float')) / (nir.astype('float') + red.astype('float'))
    
    return NDVI

def DBI(swinr1, green):
    """Calculate Dry bareness Index
    
    parameters
    ----------
    swinr1: SWINR1 band as input
    green: green band as input
    """
    DBI = ((swinr1.astype('float') - green.astype('float')) / (swinr1.astype('float') + green.astype('float'))) - ndvi
            
    return DBI


def SAVI(red, nir, L=0.5):
    """Calculate Modified Soil-adjusted Vegetation Index
    
    parameters
    ----------
    nir: NIR band as input
    red: RED band as input
    """
    SAVI = (nir.astype('float') - red.astype('float')) / (nir.astype('float') + red.astype('float') + L) * (1 + L)
    
    return SAVI
 
   
def MNDWI(green, swinr1):
    """Calculate Modified Normalized Difference Water Index
    
    parameters
    ----------
    swinr1: MINR band as input
    green: GREEN band as input
    """
    MNDWI = (green.astype('float') - swinr1.astype('float')) / (green.astype('float') + swinr1.astype('float'))
    
    return MNDWI
  
 
def NDBI(swir, nir):
    """Calculate Normalized Difference Built-up Index
    
    parameter
    ---------
    swinr: SWINR band as input
    nir: NIR band as input
    """
    NDBI = (swinr.astype('float') - nir.astype('float')) / (swinr.astype('float') + nir.astype('float') - ndvi)
    
    return NDBI
   
 
def plot_index(ch_1, ch_2, ch_3, ch_4, ch_5):
    fig,  (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, nrows=1, figsize=(18, 6), sharex=True,
                                   sharey=True)
    nd = ax1.imshow(ch_1, cmap='RdYlGn')
    ax1.set_title('NDVI')
    ax1.axis('off')
    cbar1 = fig.colorbar(mappable=nd,
                         ax = ax1, shrink=0.6, 
                         orientation='vertical',
                         extend='both')
    cbar1.set_label('NDVI_RANGE')
    
    
    
    d = ax2.imshow(ch_2, cmap='cividis')
    ax2.set_title('DBI')
    ax2.axis('off')
    cbar2 = fig.colorbar(mappable = d,
                         ax = ax2, 
                         shrink=0.6, 
                         orientation='vertical', 
                         extend='both')
    cbar2.set_label('DBI RANGE')
    
    
    svi = ax3.imshow(ch_3, cmap='YlOrBr')
    ax3.set_title('SAVI')
    ax3.axis('off')
    cbar3 = fig.colorbar(mappable = svi, 
                         ax = ax3, 
                         shrink=0.6, 
                         orientation='vertical', 
                         extend='both')
    cbar3.set_label('SAVI RANGE')
    
    mw = ax4.imshow(ch_4, cmap='YlGnBu')
    ax4.set_title('MNDWI')
    ax4.axis('off')
    cbar4 = fig.colorbar(mappable = mw, 
                         ax = ax4, shrink=0.6, 
                         orientation='vertical', 
                         extend='both')
    cbar4.set_label('MDWI RANGE')
    
    
    bi = ax5.imshow(ch_5, cmap='bone')
    ax5.set_title('NDBI')
    ax5.axis('off')
    cbar5 = fig.colorbar(mappable = bi, 
                         ax = ax5, 
                         shrink=0.6,
                         orientation='vertical', 
                         extend='both')
    cbar5.set_label('NDBI RANGE')
    plt.tight_layout()
    

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0
    """
    array_min, array_max = array.min(), array.max()
    
    return ((array - array_min)/(array_max - array_min))


def color_composite(red, green, blue):
    """
    Visualize RBG composite image
    
    parameter
    ---------
    blue: Blue band as input
    green: Green band as input
    red: RED band as input
    """
    rgb = np.dstack((red, green, blue))
    plt.figure()
    plt.imshow(rgb, cmap='terrain')
    plt.title('RGB_True_Color_Image')
    plt.axis('off')
    
    return color_composite
    
  
if __name__ == "__main__":

    start = time.time()
    print('Indices calculation starts....', '\n')

    # set up the paths
    DATA_PATH = 'D:\TESTS'
    output_file = os.path.join(DATA_PATH,'outputs','NDVI_')
    output_file_1 = os.path.join(DATA_PATH,'outputs','DBI_')
    output_file_2 = os.path.join(DATA_PATH,'outputs','SAVI_')
    output_file_3 = os.path.join(DATA_PATH,'outputs','MNDWI_')
    output_file_4 = os.path.join(DATA_PATH,'outputs','NDBI_')
    input_raster = os.path.join(DATA_PATH,'inputs','*')
    
    for i, data in enumerate(sorted(glob.glob(input_raster))):
        if data.endswith('.tif'):
            print (f"scene number {i+1} is being processed", '\n')
            raster_path = data
            folder_path = os.path.split(data)
            folder_name = folder_path[1]
    
    # load the raster image        
    raster = load_raster(data)
    
    # slice the bands
    nir = raster['band'][7, :, :]
    blue = raster['band'][1, :, :]
    green = raster['band'][2, :, :]
    red = raster['band'][3, :, :]
    swinr1 = raster['band'][11, :, :]
    swinr = raster['band'][10, :, :]
    
    # calculate NDVI
    ndvi = NDVI(nir, red)
    
    # calculate DBI
    dbi = DBI(swinr1, green)
    
    # calculate SAVI
    savi = SAVI(red, nir, L= 0.5)
    
    # calculate MNDWI
    mndwi = MNDWI(green, swinr1)
    
    # calculate NDBI
    ndbi = NDBI(swinr, red)
    
    # visualize all indices
    plot_index(ndvi, dbi,savi, mndwi, ndbi)
    
    # Normalize the bands
    redn = normalize(red)
    greenn = normalize(green)
    bluen = normalize(blue)
    
    # Create RGB natural color composite
    color_composite(redn, greenn, bluen)
    
    # save the results in destination folder
    write_raster(ndvi, raster['crs'], raster['transform'], output_file + folder_name)
    write_raster(dbi, raster['crs'], raster['transform'], output_file_1 + folder_name)
    write_raster(savi, raster['crs'], raster['transform'], output_file_2 + folder_name)
    write_raster(mndwi, raster['crs'], raster['transform'], output_file_3 + folder_name)
    write_raster(ndbi, raster['crs'], raster['transform'], output_file_4 + folder_name)
    end = time.time()
    
    print(f"the processing time is:, {round((end - start)/60, 2)} minutes", '\n')
    print('ALL DONE!')
