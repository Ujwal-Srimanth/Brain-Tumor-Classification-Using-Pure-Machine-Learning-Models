import os
import warnings
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
folder_path_no_tumor = "C:\\Users\\Srimanth\\directory to extract\\Training\\no_tumor"
folder_path_m_tumor = "C:\\Users\\Srimanth\\directory to extract\\Training\\meningioma_tumor"
folder_path_p_tumor = "C:\\Users\\Srimanth\\directory to extract\\Training\\pituitary_tumor"
folder_path_g_tumor = "C:\\Users\\Srimanth\\directory to extract\\Training\\glioma_tumor"

folder_no = os.listdir(folder_path_no_tumor)
folder_m = os.listdir(folder_path_m_tumor)
folder_p = os.listdir(folder_path_p_tumor)
folder_g = os.listdir(folder_path_g_tumor)

print(len(folder_no))
print(len(folder_m))
print(len(folder_p))
print(len(folder_g))
no_label = [0]*len(folder_no)
m_label = [1]*len(folder_m)
p_label = [2]*len(folder_p)
g_label = [3]*len(folder_g)

labels = no_label + m_label + p_label + g_label
print(len(labels))
imgdata = []

for img in folder_no:
    image = Image.open("C:/Users/Srimanth/directory to extract/Training/no_tumor/"+img)
    image = image.resize((224,224))
    image = image.convert("RGB")
    image = np.array(image)
    imgdata.append(image)
    
for img in folder_m:
    image = Image.open("C:/Users/Srimanth/directory to extract/Training/meningioma_tumor/"+img)
    image = image.resize((224,224))
    image = image.convert("RGB")
    image = np.array(image)
    imgdata.append(image)
    
for img in folder_p:
    image = Image.open("C:/Users/Srimanth/directory to extract/Training/pituitary_tumor/"+img)
    image = image.resize((224,224))
    image = image.convert("RGB")
    image = np.array(image)
    imgdata.append(image)

for img in folder_g:
    image = Image.open("C:/Users/Srimanth/directory to extract/Training/glioma_tumor/"+img)
    image = image.resize((224,224))
    image = image.convert("RGB")
    image = np.array(image)
    imgdata.append(image)

x = np.array(imgdata)
y = np.array(labels)

import numpy as np
x_2d = x.reshape(x.shape[0], -1)
import pandas as pd
data = pd.DataFrame(x_2d)
data['class_label'] = y
print(data)
X= data.drop(columns=['class_label'])
Y=data['class_label']
ica = FastICA(n_components=100)

# Fit the model to your data
rp = GaussianRandomProjection(n_components=100, random_state=42)

# Fit the model to your data
X_projected = rp.fit_transform(X)
data_pca1 = pd.DataFrame(X_projected,columns=[j for j in range(100)])
x_train,x_test,y_train,y_test = train_test_split(data_pca1,Y,test_size=0.20,shuffle=True,random_state=42)
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_classifier = RandomForestClassifier()

# Randomized Search for hyperparameter tuning
random_search_rf = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1)
random_search_rf.fit(x_train, y_train)




import pickle
pickle.dump(random_search_rf,open('mo5.pkl','wb'))
model=pickle.load(open('mo5.pkl','rb'))