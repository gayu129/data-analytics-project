# 💬 Sentiment Analysis on Tweets 🧠

## 🚀 Project Overview
This project focuses on **Sentiment Analysis** of tweets using **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques. The goal is to classify tweets into three sentiment categories:
- 😊 **Positive**
- 😐 **Neutral**
- 😞 **Negative**

The dataset used is a labeled tweet dataset from **Kaggle**, making this a practical real-world NLP task.

---

## 🔁 Workflow Breakdown

### 📥 1. Dataset Acquisition
- The dataset is downloaded using the `kagglehub` library.
- Loaded into a **Pandas DataFrame** for processing.

### 🧹 2. Data Preprocessing
- Stripped unnecessary spaces from column names.
- Cleaned the tweet text by:
  - Removing non-alphabetic characters 🔡
  - Lowercasing text 🔽
  - Removing stopwords using **NLTK** 🚫🗣️

### 📊 3. Feature Extraction
- Transformed tweets into numerical vectors using **TF-IDF Vectorizer** with 5000 features.

### 🧭 4. Target Variable Encoding
Mapped sentiments as:
- `1` → 😊 Positive
- `0` → 😞 Negative
- `2` → 😐 Neutral

### ✂️ 5. Data Splitting
- Split into **80% training** and **20% testing** using `train_test_split`.

### 🤖 6. Model Training
- Trained a **Logistic Regression** model on the TF-IDF features.

### 📈 7. Model Evaluation
Assessed model using:
- ✅ Accuracy Score
- 📋 Classification Report
- 🔀 Confusion Matrix

### 🎨 8. Visualizations
- ☁️ WordCloud for most frequent words
- 📊 Bar chart for sentiment distribution
- 🔥 Heatmap for confusion matrix
- 🏆 Bar chart of top 10 frequent words

---

## 🧰 Libraries & Tools Used

| Purpose               | Libraries Used |
|-----------------------|----------------|
| Data Handling         | `pandas`, `numpy` |
| Text Processing       | `nltk`, `sklearn.feature_extraction.text.TfidfVectorizer` |
| Model & Evaluation    | `sklearn.linear_model`, `sklearn.metrics`, `sklearn.model_selection` |
| Visualization         | `matplotlib`, `seaborn`, `wordcloud` |
| Dataset Loading       | `kagglehub` |

---

## How to Run the Project
1. Install the required libraries:
    
bash
    pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud kagglehub

2. Download the dataset using the kagglehub library.
3. Run each cell in the Jupyter Notebook sequentially to preprocess the data, train the model, and visualize the results.

## Results
- The Logistic Regression model achieved an accuracy of **X%** (replace with actual accuracy).
- The confusion matrix and classification report provide detailed insights into the model's performance.
- Visualizations such as WordCloud and sentiment distribution help in understanding the dataset better.

## Future Improvements
- Experiment with other machine learning models like Random Forest, SVM, or Neural Networks.
- Perform hyperparameter tuning to improve model performance.
- Use pre-trained embeddings like Word2Vec or BERT for feature extraction.
- Add more advanced text preprocessing techniques like stemming or lemmatization.

## Acknowledgments
- Dataset: [Tweet Sentiment Classification Dataset](https://www.kaggle.com/sahideseker/tweet-sentiment-classification-dataset)
- Libraries: Python libraries used in this project.

## Conclusion
This project demonstrates the end-to-end process of sentiment analysis on tweets, from data preprocessing to model evaluation and visualization. It provides a solid foundation for building more advanced natural language processing (NLP) applications. 
