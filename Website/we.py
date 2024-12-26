from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
from sklearn.decomposition import PCA
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
        
        # Save the uploaded image to the static folder
clf5=pickle.load(open('mo4.pkl','rb')) # Change the file name as needed
file_path = 'static/uploaded_image.jpg'
img = Image.open(file_path)
img = img.resize((224, 224))
img = img.convert("RGB")
img_data = np.array(img)

# Get the file path
img_data_flat = img_data.flatten()

# Reshape the 1D array to 2D
img_data_flat_2d = img_data_flat.reshape(1, -1)

# Create a GaussianRandomProjection object with n_components=100
rp = GaussianRandomProjection(n_components=100, random_state=42)

# Fit the model to your data
X_projected = rp.fit_transform(img_data_flat_2d)
data_pca1 = pd.DataFrame(X_projected)

# Assuming clf5 is your trained classifier
answer = clf5.predict(data_pca1)
print(answer)