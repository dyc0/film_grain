#! /usr/bin/python

from re import I
import analysis_library as al

import pandas as pd
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

    datatypes = {'image': str, 'r': str, 'sigmaR': str}
    data = pd.read_csv(os.path.join(path, 'predictions.csv'), dtype=datatypes)
    log_file = open('finals_generation_log.txt', 'w')

    for name in data['image']:
        print('Working on image {0}...'.format(name))

        localpath = os.path.join(path, name)
        denoised_source_path = os.path.join(localpath, 'denoised', 'denoised.png')

        r = data.loc[data['image']==name]['r'].values[0]
        sr = data.loc[data['image']==name]['sigmaR'].values[0]

        subprocess.run(['./film_grain_rendering_main',
                        denoised_source_path,
                        os.path.join(localpath, name + '_renoised.png'),
                        '-r', r,
                        '-sigmaR', sr,
                        '-color', '0'
                        ], stdout=log_file)

 
