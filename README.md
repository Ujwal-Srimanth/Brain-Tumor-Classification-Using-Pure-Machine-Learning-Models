
---

# Brain Tumor Classification Using Machine Learning

## Overview

This project focuses on classifying brain tumors using machine learning models. The dataset comprises grayscale MRI scans with a resolution of 224x224 pixels. These images are transformed into numerical data for machine learning model training and evaluation. An application was developed using HTML, CSS, Flask, and MongoDB to predict tumor types and store patient details.

## Dataset

- **Image Format**: Grayscale MRI scans  
- **Resolution**: 224x224 pixels  
- **Structure**: Images represented as 3D arrays (224x224x3), resulting in a 4D data structure.  
- **Dimensionality Reduction**: Reduced to 100 features per image after preprocessing.  
- **Class Imbalance Handling**:  
  - Random sampling  
  - Random over-sampling  
  - SMOTE (Synthetic Minority Over-sampling Technique)  

## Methodology

### Data Preprocessing
- **Image Transformation**: Converted images into numerical arrays.  
- **Dimensionality Reduction**: Reduced dimensions from over 150,000 columns to 100 features using optimal hyperparameters.  
- **Class Balancing**: Addressed class imbalance using SMOTE to improve model performance.  

### Machine Learning Models
Several models were evaluated, including:  
- **KNN (K-Nearest Neighbors)**: Achieved the highest accuracy of 89.9%, leveraging distance-based similarity.  
- Random Forest  
- XGBoost  
- AdaBoost  
- Gradient Boosting  
- Stacking  
- Decision Tree  
- Naïve Bayes  
- Gaussian Processes  

**Observations**:  
- Probabilistic models underperformed for this dataset.  
- High-computational-power algorithms demonstrated performance similar to neural networks.  
- Fine-tuning hyperparameters resulted in variable outcomes, emphasizing the complexity of optimization in image recognition tasks.  

## Application

### Features
- **Frontend**: Developed using basic HTML and CSS.  
- **Backend**: Flask framework.  
- **Database**: MongoDB to store patient details.  

**Functionality**:  
- Upload MRI scan images.  
- Predict tumor type.  
- Save patient details and prediction results in MongoDB.  

### Running the Application

1. **Preprocessing Scripts**:  
   - Run `bye.py`, `main.py`, and `next.py` to generate the necessary pickle files.  

2. **Flask Application**:  
   - Navigate to the `website/` folder and run `hi.py` as the main Flask application.  

3. **Static Folder**:  
   - Contains image resources and test images, located inside the `website/static/` folder.  

4. **Documentation**:  
   - Detailed methodology can be found in the `notebook/brain_tumour.ipynb`.  
   - Presentation explaining the entire project is available in `presentation/presentation.pptx`.  

### Project Structure

```
Brain Tumor Classification/
├── notebook/             # Contains the methodology notebook
│   └── brain_tumour.ipynb
├── presentation/         # Contains the project presentation
│   └── presentation.pptx
├── website/              # Flask application files
│   ├── static/           # Contains image resources and test images
│   ├── templates/        # HTML files for the frontend
│   ├── bye.py            # Script for preprocessing
│   ├── main.py           # Script for generating pickle files
│   ├── next.py           # Additional preprocessing script
│   └── hi.py             # Main Flask application
```

### Requirements

- **Python Libraries**:  
  - Flask  
  - MongoDB  
  - Scikit-learn  
  - NumPy  
  - Pandas  
  - Matplotlib  

### Installation

1. Install Python dependencies

2. Set up MongoDB and ensure it is running.  

3. Run the preprocessing scripts (`bye.py`, `main.py`, `next.py`) located in the `website/` folder.  

4. Start the Flask application:  

   ```bash
   cd website
   python hi.py
   ```

## Results

- **Best Model**: KNN with 89.9% accuracy.  
- **Performance**: Effective handling of image data with reduced dimensions and balanced classes.  

## Contributions

- **Image Preprocessing and Model Training**: Data preprocessing and model training for tumor classification.  
- **Application Development**: Basic HTML, CSS, Flask, and MongoDB for application development.  
- **Documentation**: Comprehensive explanation in `notebook/brain_tumour.ipynb` and presentation in `presentation/presentation.pptx`.  

