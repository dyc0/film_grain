#! /usr/bin/python

import cv2
import numpy as np
from scipy import ndimage

from matplotlib import pyplot as plt

import argparse, os

import warnings
warnings.filterwarnings("ignore")

def print_help():
    print("grain_analysis.py path_to_folder")

def ee(LH: np.array, HL: np.array, p = 2):
    return np.sqrt(LH**p + HL**p)**(1/p)

def generate_energy_map(img):
    h = 0.125 * np.array([-1, 2, 6, 2, -1],     dtype=np.float32)
    g1 = 0.5 * np.array([1, 0, -1],            dtype=np.float32)
    g2 = 0.5 * np.array([1, 0, 0, 0, -1],      dtype=np.float32)
    g3 = 0.5 * np.array([1, 0, 0, 0, 0, 0 -1], dtype=np.float32)

    f1 = np.matrix(np.convolve(h, g1))
    imgf1h = cv2.filter2D(img, -1, f1)
    imgf1v = cv2.filter2D(img, -1, f1.T)
    ee1 = ee(imgf1h, imgf1v)

    f2 = np.matrix(np.convolve(h, g2))
    imgf2h = cv2.filter2D(img, -1, f2)
    imgf2v = cv2.filter2D(img, -1, f2.T)
    ee2 = ee(imgf2h, imgf2v)

    f3 = np.matrix(np.convolve(h, g3))
    imgf3h = cv2.filter2D(img, -1, f3)
    imgf3v = cv2.filter2D(img, -1, f3.T)
    ee3 = ee(imgf3h, imgf3v)

    return np.maximum(np.maximum(ee1, ee2), ee3)

def get_thresholds(img, energy_map):
    img_thresh = np.floor(img*0.125)
    img_thresh = np.array(img_thresh, dtype=np.uint8)

    thresholds = np.full(32, 3, dtype=np.float32)
    w = 10E-4
    c = 2.5

    for _ in range(10):
        for value in range(32):
            mask = np.zeros(img.shape, dtype=np.uint8)
            mask[img_thresh==value] = 1
            thresholds[value] = (1-w) * thresholds[value] + w*c * np.mean(energy_map[mask==1])

    return np.nan_to_num(thresholds, nan=3), img_thresh

def clean_edges(energy_map):
    cleaned = cv2.medianBlur(np.array(energy_map, dtype=np.uint8), 5)

    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1

    min_size = 30

    cleaned = np.zeros_like(im_with_separated_blobs)

    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            cleaned[im_with_separated_blobs == blob + 1] = 255

    return cleaned

def analyze_grain(img, areas, index, imgname, outfolder):
    area = img.copy()
    area = np.array(img, dtype=np.float32)
    area[areas!=index] = -1

    noise_min = np.min(area[areas==index])
    noise_mean = np.mean(area[areas==index])
    noise_median = np.median(area[areas==index])
    noise_std  = np.std(area[areas==index])

    only_noise = img.copy()
    only_noise[areas!=index] = 0
    plt.imshow(only_noise, cmap="gray")
    plt.savefig(os.path.join(outfolder, "areas", imgname + "_" + str(index) + ".png"))
    plt.close()
    
    fft_img = np.fft.fftshift(np.fft.fft2(only_noise))
    plt.figure()
    plt.imshow(np.log(abs(fft_img)))
    plt.savefig(os.path.join(outfolder, "fft", imgname + "_" + str(index) + ".png"))
    plt.close()

    hist, edges = np.histogram(area[areas==index], bins=256, range=(0,255))
    hist = hist/np.sum(hist)
    edges = edges-noise_mean
    plt.figure()
    plt.stairs(hist, edges)
    plt.savefig(os.path.join(outfolder, "histogram", imgname + "_" + str(index) + ".png"))
    plt.close()

    return noise_min, noise_mean, noise_median, noise_std

    

if __name__ == '__main__':
    
    # Parsing arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to folder with images.")

    args=parser.parse_args()

    if args.path is None:
        path = "."
    else:
        path = args.path
    

    # Iterate through images in database

    all_files = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        if not os.path.isfile(img_path):
            continue
        all_files.append(img_path)
    all_files.sort()

    # Now analyse the image

    parameters = []
    for img_path in all_files:
        print("Working on image {0}".format(img_path))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Bluring helps with very noisy images:
        #img = cv2.medianBlur(img, 3)

        energy_map = generate_energy_map(img)
        thresholds, img_thresh = get_thresholds(img, energy_map)

        for val in range(32):
            energy_map[img_thresh==val] = (energy_map[img_thresh==val] > thresholds[val])

        cleaned = clean_edges(energy_map)

        dilation_kernel = np.ones((31,31), dtype=np.uint8)
        final_mask = cv2.dilate(np.array(cleaned, dtype=np.uint8), dilation_kernel, iterations=1)
        final_mask = cv2.bitwise_not(final_mask)
        final_mask[final_mask < 255] = 0

        connectivity_map = np.ones((3,3), dtype=np.uint8)
        areas, num_features = ndimage.label(final_mask, connectivity_map)

        vals, counts = np.unique(areas, return_counts=True)
        threshold_min = max(np.median(counts), 10000)
        threshold_max = img.size[0]*img.size[1]*0.5
        grain_area_indices = []
        for val, cnt in zip(vals[1:], counts[1:]):
            if cnt > threshold_min and cnt < threshold_max:
                grain_area_indices.append(val)

        print("Number of detected areas: {0}".format(len(grain_area_indices)))
        
        for index in grain_area_indices:
            parameters.append(analyze_grain(img, areas, index, img_path[-7:-4], "results"))

       
    with open("results/parameters.txt", "w") as of:
        of.write("MIN MEAN MEDIAN STD\n")
        for param in parameters:
            of.write(str(param) + "\n")


        #plt.figure()
        #plt.imshow(final_mask, cmap="gray")
        #plt.show()
    

