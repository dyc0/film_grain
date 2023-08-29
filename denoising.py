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
 
    foldernames = [folder for folder in os.listdir(path)
                   if os.path.isdir(os.path.join(path, folder))]
    foldernames.sort()
    print(foldernames)

    if len(foldernames) == 0: 
        print("No files were segmented by hand.")
        exit(1)

    segmented_by_hand = []
    with open(os.path.join(path, "unsegmented.txt"), 'r') as f:
       names_as_txt = f.readline() 
       segmented_by_hand = names_as_txt.strip('\n').split(' ')



    for folder in foldernames:
        print("Working on folder {0}...".format(folder))
        if folder in segmented_by_hand: print("This folder was segmented by hand.")

        current_path = os.path.join(path, folder)
        image_names = [file for file in os.listdir(current_path)
                  if file.lower().endswith('.png') and file.lower().startswith('noisy')]
        
        denoised_path = os.path.join(current_path, 'denoised')
        if not os.path.isdir(denoised_path): os.mkdir(denoised_path)
        
        for image_name in image_names:
            img = cv2.imread(os.path.join(current_path, image_name), cv2.IMREAD_GRAYSCALE)
            denoised = cv2.fastNlMeansDenoising(img, h=9)
            
            if image_name.startswith('noisy_segment') or folder in segmented_by_hand:
                final_denoised = denoised

            else:
                energy_map = al.calculate_energy_map(img)
                energy_map[energy_map < 2*np.median(energy_map)] = 0
                energy_map[energy_map != 0] = 1
                energy_map = cv2.dilate(energy_map, np.ones((3,3), dtype=np.uint8), iterations=1)

                final_denoised = denoised.copy()
                np.putmask(final_denoised, energy_map.astype(np.bool_), img)
                final_denoised[energy_map==1] = img[energy_map==1]
 
            cv2.imwrite(os.path.join(denoised_path, image_name.replace('noisy', 'denoised')), final_denoised)
