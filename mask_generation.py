#! /usr/bin/python

import cv2
import numpy as np
from scipy import ndimage as ndi

import matplotlib
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import convolve2d

import pywt
from scipy import stats
from tabulate import tabulate

import csv
import sys
import argparse, os

from analysis_library import *

if __name__ == '__main__':
    
    # Parsing arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to folder with images.")
    parser.add_argument("--oglib", "-o", help="Original images to read.")

    args=parser.parse_args()

    if args.path is None:
        path = "."
    else:
        path = args.path

    if args.oglib is None:
        oglib = "."
    else:
        oglib = args.oglib
    
    # Iterate through images in database
    if os.path.isfile(os.path.join(path, 'for_renoising.txt')):
        with open(os.path.join(path, 'for_renoising.txt')) as f:
            foldernames = f.read().splitlines()
    else:
        foldernames = [folder for folder in os.listdir(path)
                       if os.path.isdir(os.path.join(path, folder))]
        foldernames.sort()
    print(foldernames)

    if len(foldernames) == 0: 
        print("No files were segmented by hand.")
        exit(1)
    
    do_not_erode = []
    if os.path.isfile(os.path.join(path, 'unsegmented.txt')):
        with open(os.path.join(path, 'unsegmented.txt')) as f:
            do_not_erode = f.read().replace('\n','').split(' ')
    
    foldernames = [f for f in foldernames
                   if f not in do_not_erode]
    # Now analyse the images
    
    for foldername in foldernames:
        print("Working on image {0}...".format(foldername))

        local_outpath = os.path.join(path, foldername, 'segment_masks')
        if not os.path.exists(local_outpath):
            os.mkdir(local_outpath)

        img = cv2.imread(os.path.join(oglib, foldername + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        energy_map = calculate_energy_map(img)
        energy_map[energy_map < 2*np.median(energy_map)] = 0
        energy_map[energy_map > 0] = 1

        edge_map = clean_edge_map(energy_map)

        dilation_kernel = np.ones((15,15), dtype=np.uint8)
        final_mask = cv2.dilate(np.array(edge_map, dtype=np.uint8), dilation_kernel, iterations=1)
        final_mask = cv2.bitwise_not(final_mask)
        final_mask[final_mask < 255] = 0
        final_mask[final_mask != 0] = 1

        areas, grain_area_indices, num_features = segment_areas(img, final_mask)        

        for id in grain_area_indices:
            current_area = np.array(img, dtype=np.float32)
            current_area[areas!=id] = -1

            hist, edges = np.histogram(current_area[areas==id], bins=256, range=(0,255))
            hist = hist/np.sum(hist)

            maxima, _ = find_maxima(hist)
            if len(maxima)>0 and np.max(maxima) - np.min(maxima) > 40:
                ret, _ = cv2.threshold(np.array(current_area, dtype = np.uint8)[areas==id], 
                                       0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                extract_segment_mask(current_area, local_outpath, str(id), ret)
            else:
                extract_segment_mask(current_area, local_outpath, str(id))

