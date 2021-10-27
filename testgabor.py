from skimage.filters import gabor
#from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import pandas as pd
import glob
import os
import cv2
import matplotlib.pyplot as plt

SIZE = 512

#Image Input
image = cv2.imread("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Code/test1.jpg")
cv2.imshow("Input", image)
image = cv2.resize(image, (SIZE, SIZE)) #Resize images
       
# convert image to grayscale, then apply Otsu's thresholding
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gabor filter

img,gaborFilt_imag = gabor(img,frequency=0.6)
gaborFilt = (img**2+gaborFilt_imag**2)//2
# Displaying the filter response
fig, ax = plt.subplots(1,3)
cv2.imshow('Gabor Filtering', gaborFilt)
cv2.imwrite("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/output screenshots/gabor_filter.jpg", gaborFilt)

# Energy and Entropy of Gabor filter response
gabor_hist,_ = np.histogram(gaborFilt,8)
gabor_hist = np.array(gabor_hist,dtype=float)
gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
gabor_energy = np.sum(gabor_prob**2)
gabor_entropy = -np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))
print('Gabor energy = '+str(gabor_energy))
print('Gabor entropy = '+str(gabor_entropy))

img = cv2.threshold(gaborFilt, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresholding", img)
cv2.imwrite("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/output screenshots/thresholding.jpg", img)

print("done")
