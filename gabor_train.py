from skimage.feature import greycomatrix, greycoprops
from skimage.filters import gabor
import numpy as np
import pandas as pd
import glob
import os
import cv2
import matplotlib.pyplot as plt

SIZE = 512

filtered_images = []
filtered_labels = []
label_name = []

for directory_path in glob.glob("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Dataset/Train Cases/"):
    label = directory_path.split("\\")[-1]
    
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        label_name.append(os.path.basename(img_path))
        #print(img_path)
        #Reading color images and convert it to gray
        image = cv2.imread(img_path, 0)
        #Resizing with same sizes for all images
        image = cv2.resize(image, (SIZE, SIZE))
        
        filtered_images.append(image)

filtered_images = np.array(filtered_images)
print(np.array(filtered_images).shape)
label_name = np.array(label_name)

# Gabor filter
def gabor_filter(dataset):
    image_dataset = pd.DataFrame()
    i = -1
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()

        img = dataset[image, :, :]

        img,gaborFilt_imag = gabor(img,frequency=0.6)
        gaborFilt = (img**2+gaborFilt_imag**2)//2
        # Displaying the filter response
        #fig, ax = plt.subplots(1,3)
        #cv2.imshow('gabor_filtered', gaborFilt)
        
        #Applying Otsu's thresholding and Binary Thresholding
        img = cv2.threshold(gaborFilt, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #Save filtered image into a folder
        i = i + 1
        temp = label_name[i]
        #print(temp)
        cv2.imwrite('C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Dataset/Train Cases Filtered/' + temp, img)

        image_dataset = image_dataset.append(df)

    return image_dataset

filter_image = gabor_filter(filtered_images)

print("done")
