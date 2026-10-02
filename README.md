# 🫁 Lung Cancer Prediction

## Project Overview

Lung Cancer Prediction is an educational machine learning project that predicts a lung cancer class using patient-related features such as age, gender, smoking, and other health-related attributes.

The project uses a Random Forest classification model and a Flask web application. A local explanation module provides a simple explanation of the machine learning output.

> **Disclaimer:** This project is developed for educational purposes only. It is not a medical diagnostic tool.

---

## Problem Statement

Develop a Python-based machine learning model that predicts whether a person belongs to the lung cancer YES or NO class based on available dataset features.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Random Forest
* Joblib
* Flask
* HTML
* CSS
* JavaScript
* Chart.js

---

## Dataset

The project uses a lung cancer survey dataset containing patient-related attributes.

### Features

* Gender
* Age
* Smoking
* Yellow Fingers
* Anxiety
* Peer Pressure
* Chronic Disease
* Fatigue
* Allergy
* Wheezing
* Alcohol Consuming
* Coughing
* Shortness of Breath
* Swallowing Difficulty
* Chest Pain

### Target

* LUNG_CANCER

The target values are encoded as:

* 0 = NO
* 1 = YES

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Removed duplicate records.
3. Converted Gender into numerical values.
4. Converted the target column into numerical values.
5. Checked missing values.
6. Separated features and target.
7. Split the dataset into training and testing data.

---

## Machine Learning Models

Several classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine

The Random Forest model is used by the final application based on the evaluation results from the project's held-out test split.

### Random Forest Configuration

* Number of estimators: 200
* Class weight: balanced
* Random state: 42

---

## Model Evaluation

The Random Forest model achieved the following results on the project's held-out test set:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 92.86% |
| Precision | 97.83% |
| Recall    | 93.75% |
| F1 Score  | 95.74% |
| ROC-AUC   | 96.35% |

These values are results from this particular dataset split and should not be interpreted as clinical performance.

---

## Local Explanation Module

The project includes a local explanation module that provides a simple explanation of the machine learning output.

After the Random Forest model generates a prediction, the explanation module:

* Displays the model prediction.
* Displays the model probability for the YES class.
* Identifies selected input factors.
* Explains the result in simple language.
* Clearly states that the result is not a medical diagnosis.

### Explanation Workflow

```text
User Input
    ↓
Flask Application
    ↓
Random Forest Model
    ↓
Prediction + Model Probability
    ↓
Local Explanation Module
    ↓
Simple Explanation
    ↓
User Interface
```

The probability displayed by the application is the model's output for the supplied input. It is not an individual's actual medical risk percentage.

---

## Flask Application

The Flask web application provides the following pages:

* Home / Prediction
* Prediction History
* Model Performance
* Feature Importance
* About

### Prediction

Users can enter patient-related input values and receive:

* Prediction
* Model probability
* Input summary
* Local explanation
* Educational disclaimer

### Prediction History

The application stores prediction records locally in:

```text
prediction_history.csv
```

This file is excluded from Git using `.gitignore`.

### Model Performance

The application displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### Feature Importance

The application displays Random Forest feature importance values using a Chart.js visualization.

---

## Project Structure

```text
Lung-Cancer-Prediction/
│
├── data/
│   └── lung_cancer.csv
│
├── models/
│   ├── feature_names.pkl
│   └── lung_cancer_model.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── about.html
│   ├── feature_importance.html
│   ├── history.html
│   ├── index.html
│   └── performance.html
│
├── app.py
├── eda.py
├── feature_importance.py
├── lung_cancer.py
├── model_evaluation.py
├── predict.py
├── save_model.py
├── train_model.py
├── PROJECT_REPORT.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## Run the Flask Application

Start the application:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Run Command-Line Prediction

The project also includes a command-line prediction program:

```powershell
python predict.py
```

It accepts the same input features used by the trained model and displays the prediction and local explanation.

---

## Important Limitations

* This is an educational machine learning project.
* The model is trained on a specific survey dataset.
* The test-set metrics depend on the selected dataset and split.
* Model probability is not a medical risk percentage.
* Feature importance indicates how the trained Random Forest used features; it does not establish medical causation.
* The system should not be used for medical diagnosis or treatment decisions.

For real medical concerns, consult a qualified healthcare professional.

---

## Conclusion

This project demonstrates an end-to-end educational machine learning workflow involving:

* Data preprocessing
* Exploratory data analysis
* Classification
* Model evaluation
* Random Forest
* Feature importance
* Flask web development
* Prediction history
* Local explanation
* Data visualization

The project combines machine learning and web development to demonstrate how a trained classification model can be integrated into an interactive educational application.
