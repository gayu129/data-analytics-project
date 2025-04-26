# 🌍 World Bank Indicators Analysis

## Overview

This project focuses on analyzing World Bank Indicators to predict **unemployment rates** and **classify unemployment levels** using Machine Learning techniques. Two main models were developed:

- **Random Forest Regression:** Predicts continuous unemployment rates.
- **Random Forest Classification:** Categorizes unemployment into `Low`, `Medium`, and `High`.

The objective is to identify key factors influencing unemployment and evaluate the effectiveness of machine learning models in predicting and classifying unemployment scenarios globally.

---

## 📚 Project Structure

### 1. Data Acquisition
- **Dataset:** World Bank Indicators
- **Source:** [Kaggle](https://www.kaggle.com/)
- **Access Method:** Downloaded using `kagglehub`

### 2. Data Cleaning and Preprocessing
- **Handling Missing Values:** 
  - Dropping rows with missing target values.
  - Imputing missing feature values with column means.
- **Encoding Categorical Variables:**
  - `Country` column encoded using **Label Encoding**.
- **Dropping Uninformative Columns:**
  - Removed `updated_at` and `year`.
- **Feature Scaling:**
  - Standardized features using **StandardScaler**.

---

## 🔥 Project 1: World Bank Indicators Unemployment Prediction (Regression)

### Model Building
- **Model:** Random Forest Regressor
- **Training:** 
  - Data split into **80% training** and **20% testing** using `train_test_split`.

### Evaluation
- **Metrics:**
  - Root Mean Squared Error (RMSE)
  - R² Score (Coefficient of Determination)

### Visualization
- **Feature Importance Plot:** 
  - Highlights the most influential factors affecting unemployment.
- **Heatmap:** 
  - Displays correlations among top indicators.
- **Bar Plot:** 
  - Visualizes unemployment rates across different countries.

---

## 🔥 Project 2: World Bank Indicators Unemployment Classification

### Introduction
- **Goal:** Predict unemployment levels (`Low`, `Medium`, `High`).
- **Target Variable:**  
  `'Unemployment, total (% of total labor force) (modeled ILO estimate)'` transformed into categorical classes.

### Model Building
- **Model:** Random Forest Classifier
- **Hyperparameter Tuning:** 
  - **GridSearchCV** optimized:
    - `n_estimators` (Number of trees)
    - `max_depth` (Tree depth)
    - `min_samples_split` (Minimum samples to split a node)
- **Training:** 
  - Best hyperparameters used to train the model on the processed data.

### Evaluation
- **Metrics:**
  - Accuracy
  - Classification Report (Precision, Recall, F1-Score, Support)
  - Confusion Matrix

---

## 📊 Results Summary

| Model | Key Metric | Score |
|:---|:---|:---|
| Random Forest Regression | RMSE | 0.20 |
| Random Forest Regression | R² Score | *1.0* |
| Random Forest Classification | Accuracy | *0.97* |

> *(Note: Fill in the evaluation scores after running your models.)*

---

## ⚙️ Libraries Used

- `kagglehub`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn` 
  - `train_test_split`
  - `LabelEncoder`
  - `StandardScaler`
  - `RandomForestRegressor`
  - `RandomForestClassifier`
  - `GridSearchCV`
  - `mean_squared_error`
  - `r2_score`
  - `classification_report`
  - `confusion_matrix`

---

## 📝 Conclusion

- Successfully implemented **both regression and classification models** to predict and classify unemployment using global economic indicators.
- Feature importance analysis revealed which indicators most strongly influence unemployment.
- The classification model achieved reliable accuracy in categorizing countries into `Low`, `Medium`, and `High` unemployment levels.

---

## 🚀 Future Work

- Explore more advanced models like **XGBoost**, **LightGBM**, or **Neural Networks** for better performance.
- Incorporate **time-series forecasting** if longitudinal data is available.
- Perform **feature selection** or **dimensionality reduction** (e.g., PCA) to enhance model efficiency.
- Enrich the dataset by integrating external socio-economic factors such as education levels, inflation rates, etc.

---

> ✨ Feel free to fork the repository, contribute, or suggest improvements!


