#! /usr/bin/python

import cv2

import argparse, os, sys
import csv
import numpy as np
from tabulate import tabulate

import analysis_library as al

if __name__ == '__main__':
    
    # Parsing arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to results base.")

    args=parser.parse_args()

    if args.path is None:
        path = "."
    else:
        path = args.path

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
    
    do_not_erode = [dne for dne in do_not_erode if dne in foldernames]
    print(do_not_erode)


    csv_header = ["name", "min", "max", "mean", 
                  "median", "std", "kurtosis", "skewness", "sr", "r"]

    for foldername in foldernames:
        print("Working on image {0}...".format(foldername))

        all_segments_path = os.path.join(path, foldername, 'renoised_segments')
        all_masks_path = os.path.join(path, foldername, 'segment_masks')
        segments = [f for f in os.listdir(all_segments_path)
                    if os.path.isdir(os.path.join(all_segments_path, f))]

        noise_stats = []
        for segment in segments:
            print(segment)
            segment_path = os.path.join(all_segments_path, segment)
            image_names = os.listdir(segment_path)
            
            for image_name in image_names:
                r = '0.' + image_name.split('_')[2]
                sr = '0.' + image_name.split('_')[4].split('.')[0]

                img = cv2.imread(os.path.join(segment_path, image_name), cv2.IMREAD_GRAYSCALE)

                noise = []
                if foldername in do_not_erode:
                    noise = al.statistic_analysis(img.flatten(), 
                                                  name=segment.replace('segment_', ''))
                else:
                    mask_path = os.path.join(all_masks_path, 'mask_' + segment.replace('segment_', '') + '.txt')
                    mask = np.loadtxt(mask_path, dtype=np.uint8)

                    noise = al.statistic_analysis(img[mask==1],
                                                  name=segment.replace('segment_', ''))

                noise.append(sr)
                noise.append(r)
                noise_stats.append(noise)

        if len(noise_stats) > 0:
            with open(os.path.join(all_segments_path, "generated_noise_data.csv"), "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
                writer.writerows(noise_stats)
