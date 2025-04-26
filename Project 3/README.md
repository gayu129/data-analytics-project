# World Bank Indicators Unemployment Classification

## 1. Introduction

**Project Goal:**  
Predict unemployment levels (`Low`, `Medium`, `High`) based on World Bank indicators using a machine learning model.

**Dataset:**  
The project uses the `world_bank_indicators` dataset sourced from [Kaggle](https://www.kaggle.com/).

**Model:**  
A **Random Forest Classifier** is employed for prediction.

**Evaluation Metrics:**  
- Accuracy  
- Classification Report (Precision, Recall, F1-Score)  
- Confusion Matrix  

---

## 2. Data Understanding

**Data Source:**  
- Kaggle (world_bank_indicators dataset)

**Data Description:**  
- Contains various World Bank indicators for different countries.

**Target Variable:**  
- `'Unemployment, total (% of total labor force) (modeled ILO estimate)'`
- Transformed into categorical labels: `Low`, `Medium`, and `High`.

---

## 3. Data Preparation

**Data Cleaning:**  
- Dropped missing values from the target column.
- Removed irrelevant columns: `updated_at`, `year`, and the original target column after transformation.

**Feature Engineering:**  
- Transformed the continuous unemployment target into three categories: `Low`, `Medium`, `High`.
- Encoded the `country` column using **Label Encoding**.
- Dropped columns with more than 50% missing values.
- Imputed remaining missing values using the mean.

**Data Scaling:**  
- Standardized features using **StandardScaler**.

---

## 4. Model Building

**Model Selection:**  
- Chose **Random Forest Classifier** for its robustness and capability to handle complex datasets.

**Hyperparameter Tuning:**  
- Used **GridSearchCV** to optimize:
  - `n_estimators` (number of trees)
  - `max_depth` (maximum depth of trees)
  - `min_samples_split` (minimum samples to split a node)

**Training:**  
- Trained the model on the processed training data with the best hyperparameters.

---

## 5. Model Evaluation

**Evaluation Metrics:**  
- **Accuracy**: Measures overall prediction correctness.
- **Classification Report**: Includes precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: Shows true vs predicted classifications for a clearer performance view.

**Results:**  
- Evaluation metrics are calculated on the test dataset to assess and validate model performance.

---

## 6. Conclusion

**Summary:**  
- Successfully predicted unemployment categories (`Low`, `Medium`, `High`) using Random Forest on World Bank indicators.
- Achieved promising accuracy and reliable classification results.

**Limitations:**  
- The dataset might have biases due to missing or imputed values.
- Label Encoding of countries might ignore relationships between countries.
- Limited feature engineering due to missing domain-specific insights.

**Future Work:**  
- Explore more sophisticated imputation methods for missing values.
- Try different machine learning algorithms (e.g., XGBoost, LightGBM).
- Incorporate time-series analysis if yearly data is available.
- Perform feature selection or dimensionality reduction (like PCA) for better insights.

---

> ✨ Feel free to contribute or suggest improvements for the project!
