import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/ExtractedFeatures_train.csv")
data["Diagnosis"]=data["File Name"].map({'Bengincase':0,'Malignantcase':1}).astype(int)
data=data.drop(["File Name"],axis=1)
data.to_csv(r"C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/ExtractedFeatures_predicttrain.csv", index = False)

data=pd.read_csv("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/ExtractedFeatures_test.csv")
data["Diagnosis"]=data["File Name"].map({'Bengincase':0,'Malignantcase':1}).astype(int)
data=data.drop(["File Name"],axis=1)
data.to_csv(r"C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/ExtractedFeatures_predicttest.csv", index = False)


data1=pd.read_csv("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/ExtractedFeatures_predicttrain.csv")
data2=pd.read_csv("C:/Users/Aayush1999/Desktop/ACE/Final Year Project/Cancer Care/Implementation/Results/csv files/ExtractedFeatures_predicttest.csv")

print(data1.shape)
print(data2.shape)

x1 = data1[['Energy', 'Corr', 'Diss_sim', 'Homogen', 'ASM', 'Energy2', 'Corr2',
       'Diss_sim2', 'Homogen2', 'ASM2', 'Energy3', 'Corr3', 'Diss_sim3',
       'Homogen3', 'ASM3', 'Energy4', 'Corr4', 'Diss_sim4', 'Homogen4', 'ASM4',
       'Entropy']]
y1 = data1[['Diagnosis']]

x2 = data2[['Energy', 'Corr', 'Diss_sim', 'Homogen', 'ASM', 'Energy2', 'Corr2',
       'Diss_sim2', 'Homogen2', 'ASM2', 'Energy3', 'Corr3', 'Diss_sim3',
       'Homogen3', 'ASM3', 'Energy4', 'Corr4', 'Diss_sim4', 'Homogen4', 'ASM4',
       'Entropy']]
y2 = data2[['Diagnosis']]

x_train, y_train = x1, y1
x_test, y_test = x2, y2

#Applying kNN Classifier and predicting the model
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_train, y_train.values.ravel())
predict = model.predict(x_test)

accuracy=model.score(x_train,y_train)
print("Training Accuracy: ", accuracy)
accuracy2=model.score(x_test,y_test)
print("Testing Accuracy: ", accuracy2)

#print(predict)
