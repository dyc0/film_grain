#! /usr/bin/python

from re import I
import analysis_library as al
import cv2

import csv
import argparse, os, sys
from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate

import subprocess

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

    grain_sizes = [0.5, 1, 2, 3]
    sigma_relative = [0, 0.25, 0.5]

    log_file = open('subprocess_log.txt', 'w')

    for foldername in foldernames:
        
        print('------------------NEW IMAGE------------------')
        print('Working on image {0}...'.format(foldername))

        in_path = os.path.join(path, foldername, 'denoised')
        out_path = os.path.join(path, foldername, 'renoised_segments')

        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        segments = [f for f in os.listdir(in_path) if f.startswith('denoised_segment')]
        
        img = cv2.imread(os.path.join(in_path, 'denoised.png'))
        n_px = np.max(img.shape)
        d_px = 35000 / n_px

        rs = [gs/d_px for gs in grain_sizes]
        all_sigmas = [[sr*r for sr in sigma_relative] for r in rs]
        
        for segment,n in zip(segments, range(len(segments))):
            
            print('Working on segment {0} of {1}...'.format(n+1, len(segments)))
            
            segment_outpath = \
                os.path.join(out_path, segment.replace('denoised_', '').replace('.png', ''))

            if not os.path.isdir(segment_outpath):
                os.mkdir(segment_outpath)

            for r, sigma_rs in zip(rs, all_sigmas):
               for sigma_r in sigma_rs:
                    segment_outname = 'renoised_r_' + str(round(r,4)).split('.')[1] + \
                            '_sr_' + str(round(sigma_r,4)).split('.')[1] + '.png'

                    print('Generating renoised image for r={0} and sr={1}'.format(
                        str(round(r,4)), str(round(sigma_r,4))))

                    subprocess.run(['./film_grain_rendering_main',
                                    os.path.join(in_path, segment),
                                    os.path.join(segment_outpath, segment_outname),
                                    '-r', str(round(r,4)),
                                    '-sigmaR', str(round(sigma_r, 4)),
                                    '-color', '0'
                                    ], stdout=log_file)



