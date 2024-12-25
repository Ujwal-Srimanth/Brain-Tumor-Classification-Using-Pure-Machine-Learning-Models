# Brain Tumor Classification using Machine Learning

## Overview
This project focuses on classifying brain tumors using machine learning models. The dataset comprises MRI scans that are grayscale images with a resolution of 224x224 pixels. These images are transformed into numerical data for machine learning model training and evaluation. Additionally, an application was developed using HTML, CSS, Flask, and MongoDB to predict tumor types and store patient details.

---

## Dataset
- **Image Format**: Grayscale MRI scans
- **Resolution**: 224x224 pixels
- **Structure**: Images represented as 3D arrays (224x224x3), resulting in a 4D data structure.
- **Dimensionality Reduction**: Reduced to 100 features per image after preprocessing.
- **Class Imbalance Handling**: Techniques used include:
  - Random sampling
  - Random over-sampling
  - SMOTE (Synthetic Minority Over-sampling Technique)

---

## Methodology
### Data Preprocessing
1. **Image Transformation**: Converted images into numerical arrays.
2. **Dimensionality Reduction**: Reduced dimensions from over 150,000 columns to 100 features using optimal hyperparameters.
3. **Class Balancing**: Addressed class imbalance using SMOTE to improve model performance.

### Machine Learning Models
Several models were evaluated, including:
- **KNN (K-Nearest Neighbors)**: Achieved the highest accuracy of 89.9%, leveraging distance-based similarity.
- **Random Forest**
- **XGBoost**
- **AdaBoost**
- **Gradient Boosting**
- **Stacking**
- **Decision Tree**
- **Naïve Bayes**
- **Gaussian Processes**

### Observations
- Probabilistic models underperformed for this dataset.
- High-computational-power algorithms demonstrated performance similar to neural networks.
- Fine-tuning hyperparameters resulted in variable outcomes, emphasizing the complexity of optimization in image recognition tasks.

---

## Application
### Features
- **Frontend**: Developed using basic HTML and CSS.
- **Backend**: Flask framework.
- **Database**: MongoDB to store patient details.
- **Functionality**:
  - Upload MRI scan images.
  - Predict tumor type.
  - Save patient details and prediction results in MongoDB.

### Running the Application
1. **Preprocessing Scripts**:
   - Run `bye.py`, `main.py`, and `next.py` to generate the necessary pickle files.
2. **Flask Application**:
   - Run `hi.py` as the main Flask application.
3. **Assets**:
   - Contains image resources and test images.
4. **Documentation**:
   - `brain tumour classification.zip` includes detailed methodology notebooks.
   - Presentation explaining the entire project.

---

## Project Structure
```
Brain Tumor Classification/
├── assets/                # Contains image resources and test images
├── bye.py                # Script for preprocessing
├── main.py               # Script for generating pickle files
├── next.py               # Additional preprocessing script
├── hi.py                 # Main Flask application
├── brain tumour classification.zip  # Methodology and detailed documentation
├── presentation/         # Project presentation
```

---

## Requirements
### Python Libraries
- Flask
- MongoDB
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

### Installation
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up MongoDB and ensure it is running.
3. Run the preprocessing scripts (`bye.py`, `main.py`, `next.py`).
4. Start the Flask application:
   ```bash
   python hi.py
   ```

---

## Results
- **Best Model**: KNN with 89.9% accuracy.
- **Performance**: Effective handling of image data with reduced dimensions and balanced classes.

---

## Contributions
- **Image Preprocessing and Model Training**
- **Application Development**: Basic HTML, CSS, Flask, and MongoDB.
- **Documentation**: Comprehensive explanation in `brain tumour classification.zip`.

---

## Future Work
- Enhance the application with advanced UI/UX.
- Experiment with deep learning models for improved accuracy.
- Optimize the handling of large datasets for faster computation.

---

## Acknowledgments
Special thanks to the contributors and resources that facilitated the completion of this project.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

