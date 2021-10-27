# lung-cancer-classification-knn

We have conducted a research study to detect lung cancer by developing knn classification technique. CT Scans are converted to greyscale for pre-processing, Gabor filter is then used on the grayscale images to obtain enhanced images. These enhanced images are then passed into Otsu’s thresholding for noise reduction and brightening of faint pixels. The noise-reduced images are passed into GLCM to extract features such as homogeneity, energy, entropy, etc. into a .csv dataset. The dataset obtained is passed into a kNN classification model to classify lung cancer.
We have combined image enhancement, feature extraction, and machine learning methods which shows an accuracy of 92.37%.
