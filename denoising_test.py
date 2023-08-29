#! /usr/bin/python

import cv2
import numpy as np
from scipy import ndimage

from matplotlib import pyplot as plt
from scipy import signal

import analysis_library as al

img = cv2.imread("analysis_output/069/noisy.png", cv2.IMREAD_GRAYSCALE)
denoised = cv2.fastNlMeansDenoising(img, h=9)

energy_map = al.calculate_energy_map(img)
energy_map[energy_map < 2*np.median(energy_map)] = 0
energy_map[energy_map != 0] = 1
energy_map = cv2.dilate(energy_map, np.ones((3,3), dtype=np.uint8), iterations=1)

plt.figure()
plt.imshow(energy_map, cmap='gray')

final_denoised = denoised.copy()
np.putmask(final_denoised, energy_map.astype(np.bool_), img)
final_denoised[energy_map==1] = img[energy_map==1]

plt.figure()
plt.subplot(2,1,1)
plt.imshow(img, cmap="gray")
plt.title("ORIGINAL")
plt.subplot(2,2,3)
plt.imshow(denoised, cmap="gray")
plt.title("DENOISED")
plt.subplot(2,2,4)
plt.imshow(final_denoised, cmap='gray')
plt.title("DENOISED SA IVICAMA")

plt.show()
