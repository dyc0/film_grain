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
    parser.add_argument("--out", "-o", help="Parent output path.")

    args=parser.parse_args()

    if args.path is None:
        path = "."
    else:
        path = args.path
    if args.out is None:
        outpath = "./analysis_output"
    else:
        outpath = args.out
    

    # Iterate through images in database

    all_images = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        if not os.path.isfile(img_path):
            continue
        all_images.append(filename)
    all_images.sort()

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    csv_header = ["name", "min", "max", "mean", "median", "std", "kurtosis", "skewness"]

    # Now analyse the images
    
    unsegmented = []
    for img_name in all_images:
        print("Working on image {0}...".format(img_name))

        local_outpath = os.path.join(outpath, img_name[:-4])
        if not os.path.exists(local_outpath):
            os.mkdir(local_outpath)

        current_noise_data = []

        img = cv2.imread(os.path.join(path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(local_outpath, "noisy.png"), img)

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
            plt.figure()
            plt.stairs(hist, edges)
            plt.savefig(os.path.join(local_outpath, str(id) + "_hist.png"))

            maxima, smoothed_hist = find_maxima(hist)

            if len(maxima)>0 and np.max(maxima) - np.min(maxima) > 40:
                ret, _ = cv2.threshold(np.array(current_area, dtype = np.uint8)[areas==id], 
                                       0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                lower = current_area[np.logical_and(
                    areas==id, current_area < ret)]
                upper = current_area[np.logical_and(
                    areas==id, current_area >= ret)]
                current_noise_data.append(statistic_analysis(lower,  str(id) + "_1")) 
                current_noise_data.append(statistic_analysis(upper, str(id) +  "_2"))
                extract_segment(current_area, local_outpath, str(id), ret)
            else:
                current_noise_data.append(statistic_analysis(current_area[areas==id], str(id)))
                extract_segment(current_area, local_outpath, str(id))

        if len(current_noise_data) > 0:
            with open(os.path.join(local_outpath, "noise_data.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
                writer.writerows(current_noise_data)
        else:
            unsegmented.append(img_name[:-4])
    
    print(unsegmented)
    np.savetxt(os.path.join(outpath, "unsegmented.txt"), unsegmented)



