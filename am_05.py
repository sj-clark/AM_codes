
"""
Created on Wed Feb 17 16:20:16 2021

@author: samueljclark
"""

import matplotlib.pyplot as plt
import matplotlib.widgets as wdg

import os

import sys

import argparse

import glob

import cv2

import numpy as np

import skimage.morphology as skim
import skimage.measure as skime
import skimage.filters as skif
import skimage.segmentation as skis

import scipy.ndimage as ndi

class slider():
    def __init__(self, data):
        self.data = data

        plt.subplot(111)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        self.frame = 0
        self.l = plt.imshow(self.data[:,:,self.frame], vmin = 0, vmax = 1) 
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe = wdg.Slider(axframe, 'Frame', 0, self.data.shape[2]-1,
                                 valfmt='%0.0f')
        self.sframe.on_changed(self.update)

    def update(self, val):
        self.frame = int(np.around(self.sframe.val))
        self.l.set_data(self.data[:,:,self.frame])
    
def interface_mask(im):
    skymask = np.zeros_like(im)
    im = skif.gaussian(im, sigma = 5)
    im = skif.sobel_h(im)
    
    for i in range(im.shape[1]):
        skymask[0:np.argmin(im[:,i]), i] = 1      
    skymask = skymask.astype(bool)
    return  skymask

def delta_stack(ims,im):
    delta_ims = ims / (np.repeat(im.reshape((ims.shape[0], ims.shape[1], 1)),
                                 ims.shape[2], axis = 2))
    noise = np.std(delta_ims) 
    return delta_ims, noise


def normalize_stack(ims, noise):
    for i in range(ims.shape[2]):
        clean = np.copy(ims[:,:,i])
        median = np.nanmedian(clean)
        clean[clean > (median+6*noise)] = np.nan
        clean[clean < (median-6*noise)] = np.nan
        noise2 = np.nanstd(clean)
        mean2 = np.nanmean(clean)
        ims[:,:,i] = (noise/noise2)*(ims[:,:,i]-mean2)+1
        
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
    
    benchmark = np.copy(ims[:,:,0])
    ratio = np.zeros(ims.shape[2])
    
    for i in range(ims.shape[2]):
        im = ims[:,:,i]
        imratio = im/benchmark
        ratio[i] = np.nanmedian(imratio)
        ims[:,:,i] = ims[:,:,i]/ratio[i]
        
    return ims

def FFC(ims, darkim, flatim):
        
    for i in range(ims.shape[2]):
        ims[:,:,i] = (ims[:,:,i]-darkim)/(flatim-darkim)
        
    return ims

def scale_stack(ims):
    upper_bound = np.nanmax(ims)
    lower_bound = np.nanmin(ims)
    ims = (ims-lower_bound)/(upper_bound-lower_bound)
    ims[ims > 1] = 1
    ims[ims < 0] = 0
    return ims

def denoise_segmentation(segmentation):
    additive = (segmentation == 0)
    subtractive = (segmentation == 2)
    additive = ndi.median_filter(additive, size = (6, 6, 1))
    subtractive = ndi.median_filter(subtractive, size = (6, 6, 1))
    segmentation = np.ones_like(segmentation)
    segmentation[additive] = 0
    segmentation[subtractive] = 2
    return segmentation

def interpret_segmentation(segmentation, skymask, keyhole_threshold):

    # Segmentation cleanup - remove segemnted negative build above initial 
    # level as this is impossible
    segmentation[((segmentation == 2) & np.repeat(skymask.reshape((
        segmentation.shape[0],segmentation.shape[1], 1)),segmentation.shape[2],
        axis = 2))] = 1
    # Segmentation cleanup - remove segemnted positive build below initial 
    # level as this would be unexpexted - review if using dissimilar powder
    # and substrate
    segmentation[((segmentation == 0) & ~np.repeat(skymask.reshape((
        segmentation.shape[0],segmentation.shape[1], 1)),segmentation.shape[2],
        axis = 2))] = 1
    
    # isolate and extract the keyhole location and label the keyhole as 3     
    n = 0
    keyhole_centroid = np.empty([1,3])
    for i in range(segmentation.shape[2]):
        clean = segmentation[:,:,i] == 2
        clean = skim.remove_small_objects(clean, min_size = keyhole_threshold)
        if (np.sum(clean) > 0):
            clean = skime.label(clean)
            clean = (clean==np.bincount(clean.ravel())[1:].argmax()+1).astype(
                int)
            clean2 = skis.clear_border(clean)
            regions = skime.regionprops(clean2)
            if (np.sum(clean2) > 0):
                if n == 0:
                    keyhole_centroid = [[i,regions[0].centroid[0],
                                         regions[0].centroid[1]]]
                else:
                    keyhole_centroid = np.append(keyhole_centroid, [[i,
                                   regions[0].centroid[0],regions[0].centroid[
                                       1]]], axis = 0)
                n = n+1    
            segmentation[(clean == 1),i] = 3
     
    # Frozen in pores are stationary use a median filter in the t direction
    clean = (segmentation == 2)
    clean = ndi.median_filter(clean, size=(1,1,10))
    segmentation[((segmentation == 2) & ~clean)] = 1       
            
    # Particles and spatters are positive and above the surface so find what is 
    # either added material or the inititial substrate spatters are labelled as
    # 4
    clean = ((segmentation == 0) | ~np.repeat(skymask.reshape((
        segmentation.shape[0],segmentation.shape[1], 1)),segmentation.shape[2],
        axis = 2) == 1)
    clean2 = np.copy(clean)
    # find all but the biggest object in each frame
    for i in range(segmentation.shape[2]):
        im = np.copy(clean[:,:,i])
        if (np.sum(clean[:,:,i]) > 0):
            im = skime.label(im)
            clean[:,:,i] = (im==np.bincount(im.ravel())[1:].argmax()+1).astype(
                int)
    segmentation[(clean2 & ~clean)] = 4
    
    # label the region of no change that is in the skymask as the sky labeled 
    # as 5
    segmentation[((segmentation == 1) & np.repeat(skymask.reshape((
        segmentation.shape[0], segmentation.shape[1], 1)), segmentation.shape[
            2], axis = 2))] = 5
    
    return segmentation, keyhole_centroid

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
    parser.add_argument("filepathin", help = "top directory where the tif images are located: /data/")
    parser.add_argument("filepathout", help = "top directory where the labelled images are to be saved: /data/")

    args = parser.parse_args()
    
    filepathout = args.filepathout
    filepathin = args.filepathin
    
    keyhole_threshold = 500
    
    rawims = loadsynchronous(filepathin)
    
    rawims = ndi.median_filter(rawims, size = (3, 3, 1))
    
    clearims = np.copy(rawims[:,:,0:50])
    clearims = flicker(clearims)
    clearim = np.nanmean(clearims, axis = 2)
    delta_clearims, noise = delta_stack(clearims, clearim)
    thresholds = [np.nanquantile(delta_clearims, 0.001),np.nanquantile(delta_clearims, 0.999)]
    
    im0 = np.copy(rawims[:,:,0]) 
    skymask = interface_mask(im0)
    
    # divide the stack by the mean of the first 50 images
    delta_rawims, _ = delta_stack(rawims, clearim)
    # Normalise the stack by matching the the mean and standard deviation 
    delta_rawims = normalize_stack(delta_rawims, noise)
    # Segment into 3 catagories (additive = 1, background = 2, subtractive = 3)
    segmentation = np.digitize(delta_rawims, bins=thresholds)
    
    # Apply a median filter to the additive and subtractive masks and 
    # re-combine 
    segmentation = denoise_segmentation(segmentation)
    # Interpret that segementation (additive = 0, substrate = 1,
    # frozen porosity = 2, keyhole = 3, spatter = 4, sky = 5)
    segmentation, keyhole_centroid = interpret_segmentation(segmentation, 
                                 skymask, keyhole_threshold)
    
    saveims(np.uint8(segmentation),filepathin,filepathout)

    plt.hist(delta_rawims.ravel(), bins = 256, color = 'blue')
    plt.hist(delta_clearims.ravel(), bins = 256, color = 'red')
    plt.yscale('symlog')
    plt.show()    
        
    slider_ims = scale_stack(segmentation)
    
    slider(slider_ims)
    
    plt.show()
    
    return segmentation
    
if __name__ == "__main__":
    
    segmentation = main(sys.argv[1:])