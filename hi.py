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
import uuid
from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from bson import json_util

app = Flask(__name__)
clf4=pickle.load(open('mo2.pkl','rb'))
clf5=pickle.load(open('mo4.pkl','rb'))
clf6=pickle.load(open('mo5.pkl','rb'))
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']  # Change 'your_database_name' to your actual database name
collection = db['braintumor']

l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremities', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
      'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
      'red_sore_around_nose', 'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox',
           'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
           'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemorrhoids(piles)',
           'Heartattack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthritis',
           'Arthritis', '(vertigo) Paroxysmal Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']
@app.route('/')
def main():
    return render_template("ht.html")

@app.route('/hi')
def hello_world():
    return render_template("formm.html")
@app.route('/hibye')
def hello_world_bye():
    return render_template("upload.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    symptoms=[str(x) for x in request.form.values()]
    print(symptoms)
    input_data = [0] * len(l1)  # Initialize with zeros
    # Set the corresponding symptoms to 1
    for symptom in symptoms:
        input_data[l1.index(symptom)] = 1

    # Make a prediction using your machine learning model
    prediction = clf4.predict([input_data])[0]

    # Map the prediction to the corresponding disease
    predicted_disease = disease[prediction]

    return render_template("formm.html",pred = predicted_disease)

@app.route('/upload', methods=['POST','GET'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']
        
        # Save the uploaded image to the static folder
        unique_filename = str(uuid.uuid4()) + '.jpg'
        
        # Save the uploaded image with the unique file name
        image.save('static/' + unique_filename)

        # Get the file path
        file_path = 'static/' + unique_filename
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
        family_history = request.form['model']
        if(family_history == 'Random Forest Classifier'):
            answer = clf6.predict(data_pca1)
        else:
            answer = clf5.predict(data_pca1)


# Assuming clf5 is your trained classifier
        
        if(answer[0]==0):
            ans = 'No Tumor'
        elif(answer[0]==1):
            ans = 'Menognia Tumor'
        elif(answer[0]==2):
            ans = 'Pitutary Tumor'
        elif(answer[0]==3):
            ans = 'Glioma Tumor'
        name = request.form['name']
    age = int(request.form['age'])
    gender = request.form['gender']
    cancer_history = request.form['cancer_history']
    family_history = request.form['model']

    # Insert data into MongoDB
    



# Pass the result to your templansate
    return render_template("upload.html", answer=ans)

        # Now, you can pass the file_path to your Python script or perform any other actions.
        # For example, redirecting to another route with the file_path as a parameter.
        

    return "No image uploaded."
@app.route('/get_data')
def get_data():
    

    return render_template("visualization.html")



if __name__ == '__main__':
    app.run(debug=True)