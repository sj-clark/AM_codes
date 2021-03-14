
"""
Created on Wed Feb 17 16:20:16 2021

@author: samueljclark
"""

import matplotlib.pyplot as plt

import pylab

import os

import sys

import argparse

import glob

import cv2

import numpy as np

import skimage.morphology as skim
import skimage.filters as skif
import skimage.segmentation as skis

import scipy.ndimage as ndi
    
def interface_mask(im):
    skymask = np.zeros_like(im)
    im = skif.gaussian(im, sigma = 5)
    im = skif.sobel_h(im)
    
    for i in range(im.shape[1]):
        skymask[0:np.argmin(im[:,i]), i] = 1      
    
    skymask = skymask.astype(bool)
    return  skymask

def delta_stack(ims,im):
    im = np.log(im+1)
    ims = np.log(ims+1)
    #n = 20
    #n_shift = int(np.ceil((n/2)))
    #background = ndi.median_filter(ims, size = [1,1,n])
    #background = ndi.shift(background,[0,0,n_shift])
    #background[:,:,0:n_shift] = np.repeat(im.reshape((ims.shape[0], ims.shape[1], 1)),n_shift, axis = 2)
    #ims = ims-background
    ims = ims - (np.repeat(im.reshape((ims.shape[0], ims.shape[1], 1)), ims.shape[2], axis = 2)) 
    return ims

def loaddir(indir):
    imsequence = sorted(glob.glob(indir + '/*.tif'))
    dataset = cv2.imread(imsequence[0],-1)
    h,w = np.shape(dataset)
    ims = np.zeros((h,w,len(imsequence)))

    for i in range(len(imsequence)):
        dataset = cv2.imread(imsequence[i],-1)
        ims[:,:,i] = np.array(dataset)
        
    return ims

def loadsynchronous(indir):

    if isinstance(indir, list):
        imsample = sorted(glob.glob(indir[0] + '/*.tif'))
        dataset = cv2.imread(imsample[0], -1)
        h,w = np.shape(dataset)
        
        tempims = np.zeros((h, w, len(indir)))
        ims = np.zeros((h, w, len(imsample)))
        imsequence = np.empty((len(imsample), len(indir)), dtype = 'object')
        
        for i in range(len(indir)):
            dataset = sorted(glob.glob(indir[i] + '/*.tif'))
            
            if len(dataset) == len(imsample):
                imsequence[:,i] = sorted(glob.glob(indir[i] + '/*.tif'))
            
            else:
                print('Error - scans are of different length')
        
        for i in range(len(imsample)):
            for j in range(len(indir)):
                dataset = cv2.imread(imsequence[i,j],-1)
                ims[:,:,j] = np.array(dataset)
            
            ims[:,:,i] = np.nanmean(tempims, axis = 2)
            
    else:

        ims = loaddir(indir)
        
    return ims 

def flicker(ims):
    benchmark = np.copy(ims[:,:,0])+1
    
    for i in range(ims.shape[2]):
        ratio = np.nanmedian(((ims[:,:,i]+1)/benchmark))
        ims[:,:,i] = ims[:,:,i]/ratio
        
    return ims

def scale_stack(ims):
    upper_bound = np.nanmax(ims)
    lower_bound = np.nanmin(ims)
    ims = (ims-lower_bound)/(upper_bound-lower_bound)
    return ims

def normalize_stack(ims):
    
    for i in range(ims.shape[2]):
        upper_bound = np.nanquantile(ims[:,:,i], 0.9) 
        lower_bound = np.nanquantile(ims[:,:,i], 0.1)
        ims[:,:,i] = (ims[:,:,i]-lower_bound)/(upper_bound-lower_bound)
        
    return ims

def region_segment(ims, n):
    segmented = np.zeros_like(ims)
    markers = np.zeros_like(ims)
    elevation = np.zeros_like(ims)
    selem = skim.disk(1)
    r = abs(np.nanquantile(ims[:,:,0:n],0.0001)-np.nanquantile(ims[:,:,0:n], 0.9999))
    m = np.nanmedian(ims[:,:,0:n])
    thresholds = [m-(1*r), m-(0.25*r), m+(0.25*r), m+(1*r)]

    for i in range(ims.shape[2]):
        elevation[:,:,i] = skif.sobel(ims[:,:,i])
        markers[ndi.binary_erosion(np.logical_and(ims[:,:,i] > thresholds[1], ims[:,:,i] < thresholds[2])), i] = 1
        markers[ndi.binary_erosion(ims[:,:,i] < thresholds[0],structure=selem),i] = 2
        markers[ndi.binary_erosion(ims[:,:,i] > thresholds[3],structure=selem),i] = 3
        segmented[:,:,i] = skis.watershed(elevation[:,:,i], markers[:,:,i])
    
    return segmented, markers, elevation, thresholds


def saveims(ims,indir,outdir):
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    imsample = sorted(glob.glob(indir + '/*.tif'))
    
    for i in range(ims.shape[2]):
        base = os.path.basename(imsample[i])
        filename = (outdir + '/'+ base)
        cv2.imwrite(filename,ims[:,:,i])

def main(arg):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("filepathin", help = "top directory where the tif images are located")
    parser.add_argument("filepathout", help = "top directory where the labelled images are to be saved")
    parser.add_argument("n", help = "First n images for background detection")
    parser.add_argument("rad", help = "Radius of median filter")
    
    args = parser.parse_args()
    
    filepathout = args.filepathout
    filepathin = args.filepathin
    n = int(args.n)
    rad = args.rad
    
    """
    filepathout = '/Users/samueljclark/Desktop/output'
    filepathin = '/Users/samueljclark/Desktop/APS_Ti_Weld/UCL_S0433'
    n = 50 # first n clear images
    rad = 1.5 # Radius of median filter for initial filtering
    """
    
    A = loadsynchronous(filepathin)
    selem = skim.disk(float(rad))
    selem = selem.reshape((selem.shape[0],selem.shape[1],1))
    B = normalize_stack(skif.median(A,selem))
    C = flicker(B)
    D = np.nanmean(C[:,:,0:n], axis = 2)
    E = delta_stack(C, D)
    segmented, markers, elevation, thresholds = region_segment(E, n)
    saveims(np.uint8(segmented),filepathin,filepathout)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(E.ravel(), bins = 256, color = 'blue', label= 'Raw')
    ax.hist(E[:,:,0:n].ravel(), bins = 256, color = 'red', label= 'First n (background)')
    for thresh in thresholds:
        ax.axvline(thresh, color='black') 
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('Stack subtracted by average of first 50 frames')
    ax.set_ylabel('Counts')
    pylab.show()
    
if __name__ == "__main__":

    main(sys.argv[1:])
