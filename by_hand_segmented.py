#! /usr/bin/python

import analysis_library as al
import cv2

import csv
import argparse, os, sys
from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate

if __name__ == '__main__':
    
    # Parsing arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to results base.")

    args=parser.parse_args()

    if args.path is None:
        path = "."
    else:
        path = args.path
 
    filenames = []
    with open(os.path.join(path, "unsegmented.txt"), 'r') as f:
       names_as_txt = f.readline() 
       filenames = names_as_txt.strip('\n').split(' ')

    if len(filenames) == 0: 
        print("No files were segmented by hand.")
        exit(1)

    csv_header = ["name", "min", "max", "mean", "median", "std", "kurtosis", "skewness"]

    for name in filenames:
        print("Working on {0}...".format(name))

        current_noise_data = []
        local_path = os.path.join(path, name)

        segments = os.listdir(local_path)
        segments = [s for s in segments if s.startswith('noisy_segment')]
        
        for segment in segments:
            img = cv2.imread(os.path.join(local_path, segment))
            current_noise_data.append(al.statistic_analysis(img.flatten(), name=segment[-5:-4]))
            segment_name = segment.split('_')[2].split('.')[0]
            
            hist, edges = np.histogram(img, bins=256, range=(0,255))
            hist = hist/np.sum(hist)
            plt.figure()
            plt.stairs(hist, edges)
            plt.savefig(os.path.join(local_path, segment_name + "_hist.png"))
            plt.close()
        
        if len(current_noise_data) > 0:
            with open(os.path.join(local_path, "noise_data.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
                writer.writerows(current_noise_data)

