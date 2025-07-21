# 🫀 Heart Failure Prediction using Logistic Regression

This project is part of the **Predictive Modelling Bootcamp**. It uses a Logistic Regression model to predict the likelihood of death in heart failure patients based on clinical records.

## 📊 Dataset
The dataset contains medical and demographic information of 299 heart failure patients, including:
- Age
- Ejection fraction
- Serum creatinine
- Blood pressure, diabetes, anaemia, and more

**Target Column:** `DEATH_EVENT` (0 = Survived, 1 = Died)

## 🧠 Model Used
- **Logistic Regression**
- Achieved **80% accuracy** on the test set
- Model trained using `scikit-learn` with preprocessing pipeline

## 🖥️ Web Application (Flask)
A simple and clean web interface built using Flask.

### Features:
- Input patient data through a form
- Displays prediction result
- Styled with custom CSS

