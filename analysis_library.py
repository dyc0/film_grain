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

import sys
import argparse, os

def ee(LH: np.array, HL: np.array, p = 2.0):
    return (np.absolute(LH)**p + np.absolute(HL)**p)**(1.0/p)


def statistic_analysis(data, name=None):
    noise_min = np.min(data)
    noise_max = np.max(data)
    noise_mean = np.mean(data)
    noise_median = np.median(data)
    noise_std  = np.std(data)
    noise_kurtosis = stats.kurtosis(data)
    noise_skewness = stats.skew(data)
    
    if name is None:
        return [noise_min, noise_max, noise_mean, noise_median, noise_std, noise_kurtosis, noise_skewness]
    else:
        return [name, noise_min, noise_max, noise_mean, noise_median, noise_std, noise_kurtosis, noise_skewness] 

def find_maxima(data):
    data_mean = np.mean(data)

    wavelet = "sym7"
    coeffs = pywt.wavedec(data, wavelet, level=3)
    coeffs[1:] = [pywt.threshold(c, value=0.5, mode='soft') for c in coeffs[1:]]
    smoothed_data = pywt.waverec(coeffs, wavelet)

    maxima = signal.argrelmax(smoothed_data, order=15)
    maxima = maxima[0]
    maxima = [m for m in maxima if  data[m] > data_mean/8]

    return maxima, smoothed_data


def calculate_energy_map(img):
    h = 0.125 * np.array([-1, 2, 6, 2, -1],     dtype=np.float32)
    g1 = 0.5 *  np.array([1, 0, -1],            dtype=np.float32)
    g2 = 0.5 *  np.array([1, 0, 0, 0, -1],      dtype=np.float32)
    g3 = 0.5 *  np.array([1, 0, 0, 0, 0, 0 -1], dtype=np.float32)

    gs = [g1, g2, g3]
    filters = []

    for g in gs:
        convolved_filter = np.convolve(h, g)
        padding = convolved_filter.shape[0]//2
        filters.append(
            np.pad(np.atleast_2d(convolved_filter),
                   ((padding, padding), (0,0)))
        )

    ees = []
    imgf = np.array(img, dtype=np.float32)
    for f in filters:
        imgh = convolve2d(imgf, f, mode='same', boundary='symm')
        imgv = convolve2d(img, f.T, mode='same', boundary='symm')
        ees.append(ee(imgh, imgv, p=2))

    energy_map = np.zeros(ees[0].shape)
    for e in ees:
        energy_map = np.maximum(energy_map, e)

    return energy_map


def clean_edge_map(edge_map):
    cleaned = cv2.medianBlur(cv2.normalize(edge_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U), 5)

    nb_blobs, im_with_separated_blobs, constats, _ = cv2.connectedComponentsWithStats(cleaned)
    sizes = constats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1

    min_size = 30

    cleaned = np.zeros_like(im_with_separated_blobs)

    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            cleaned[im_with_separated_blobs == blob + 1] = 255

    return cleaned


def segment_areas(img, mask, connectivity_map = np.ones((3,3), dtype=np.uint8)):
    areas, num_features = ndi.label(mask, connectivity_map)

    vals, counts = np.unique(areas, return_counts=True)
    # This excludes BG:
    vals = vals[1:]
    counts = counts[1:]

    threshold_min = img.size*0.01
    threshold_max = img.size*0.75
    grain_area_indices = []
    for val, cnt in zip(vals, counts):
        if cnt >= threshold_min and cnt <= threshold_max:
            grain_area_indices.append(val)

    return areas, grain_area_indices, num_features


def extract_segment(img_segment: np.ndarray, path: str, name: str, threshold = None):
    if threshold is not None:
        lower = img_segment.copy()
        lower[np.logical_or(lower>=threshold, lower<0)] = 0
        save_segment(lower, path, name + "_1")

        upper = img_segment.copy()
        upper[upper < threshold] = 0
        save_segment(upper, path, name + "_2")
        
    else:
        save_segment(img_segment, path, name)

def extract_segment_mask(img_segment: np.ndarray, path: str, name: str, threshold = None):
    if threshold is not None:
        lower = img_segment.copy()
        lower[np.logical_or(lower>=threshold, lower<0)] = 0
        lower[lower>0] = 1
        save_mask(lower, path, 'mask_' + name + '_1.txt')
    
        upper = img_segment.copy()
        upper[upper < threshold] = 0
        upper[upper >0] = 1
        upper = upper.astype(np.uint8)
        save_mask(upper, path, 'mask_' + name + '_2.txt')
        
    else:
        mask = img_segment.copy()
        mask[mask!=-1] = 1
        mask[mask==-1] = 0
        save_mask(mask, path, 'mask_' + name + '.txt')

def save_mask(img_segment: np.ndarray, path: str, name: str):
    mask = img_segment.copy()
    mask = mask.astype(np.uint8)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    np.savetxt(os.path.join(path, name),
               img_segment[rmin:rmax, cmin:cmax], fmt='%i')

   

def save_segment(img_segment: np.ndarray, path: str, name: str):
    mask = img_segment.copy()
    mask[mask>0] = 1
    mask[mask<0] = 0
    mask = mask.astype(np.uint8)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cv2.imwrite(os.path.join(path, "noisy_segment_" + name + ".png"), img_segment[rmin:rmax, cmin:cmax])
