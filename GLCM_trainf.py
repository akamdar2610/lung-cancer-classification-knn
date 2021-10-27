import numpy as np 
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import re

#Resize images to
SIZE = 512

#Capture images and labels into arrays.
train_images = []
train_labels = [] 
label_name = []

# Train
for directory_path in glob.glob("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Dataset/Train Cases Filtered BM/"):
    label = directory_path.split("\\")[-1] 
    
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        temp = os.path.basename(img_path)
        filename, file_extension = os.path.splitext(os.path.basename(img_path))
        new_name3 = re.sub('[0-9]', '', filename)
        new_name2 = re.sub('\(', '', new_name3)
        new_name1 = re.sub('\)', '', new_name2)
        new_name = re.sub(' ', '', new_name1)
        
        label_name.append(new_name)
        #Reading images
        img = cv2.imread(img_path, 0)
        #Resizing all images with same size
        img = cv2.resize(img, (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train = train_images, train_labels_encoded

# FEATURE EXTRACTOR function

def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        
        img = dataset[image, :,:]
        
        #START ADDING DATA TO THE DATAFRAME
        #Extracting features pixel by pixel with different offsets and angles
        #1
        GLCM = greycomatrix(img, [1], [0])       
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss       
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_asm = greycoprops(GLCM, 'ASM')[0]
        df['ASM'] = GLCM_asm
        #2
        GLCM2 = greycomatrix(img, [1], [np.pi/2])       
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2      
        GLCM_asm2 = greycoprops(GLCM, 'ASM')[0]
        df['ASM2'] = GLCM_asm2
        #3
        GLCM3 = greycomatrix(img, [1], [np.pi/4])       
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3       
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3       
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3   
        GLCM_asm3 = greycoprops(GLCM, 'ASM')[0]
        df['ASM3'] = GLCM_asm3
        #4  
        GLCM4 = greycomatrix(img, [1], [3*np.pi/4])       
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4       
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4       
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4 
        GLCM_asm4 = greycoprops(GLCM, 'ASM')[0]
        df['ASM4'] = GLCM_asm4
        
        #Extracting shannon entropy
        entropy = shannon_entropy(img)
        df['Entropy'] = entropy
        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        
    return image_dataset

#Extract features from training images
image_features = feature_extractor(x_train)

#saving the extracted features from dataframe to csv
label_dataframe = pd.DataFrame(label_name,columns=['File Name'])
label_dataframe.to_csv(r'C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/temp.csv', index = False, header = True)
image_features.to_csv(r'C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/features_train.csv', index = False, header = True)
df = pd.read_csv("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/features_train.csv")
df["File Name"] = pd.read_csv('C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/temp.csv')
df.to_csv(r'C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/ExtractedFeatures_train.csv', index = False)

