# Sentiment Analysis on Tweets

## Project Overview
This project focuses on sentiment analysis of tweets using machine learning techniques. The goal is to classify tweets into three categories: **positive**, **negative**, and **neutral** sentiments. The dataset used for this project is downloaded from Kaggle and contains labeled tweets.

## Workflow
The project is divided into the following steps:

### 1. **Dataset Acquisition**
    - The dataset is downloaded using the `kagglehub` library.
    - The dataset is loaded into a Pandas DataFrame for further processing.

### 2. **Data Preprocessing**
    - Column names are stripped of leading/trailing spaces.
    - Text data is cleaned by:
      - Removing non-alphabetic characters.
      - Converting text to lowercase.
      - Removing stopwords using the NLTK library.

### 3. **Feature Extraction**
    - Text data is transformed into numerical features using the **TF-IDF Vectorizer** with a maximum of 5000 features.

### 4. **Target Variable Encoding**
    - The target variable (`sentiment`) is encoded as:
      - `1` for positive
      - `0` for negative
      - `2` for neutral

### 5. **Data Splitting**
    - The dataset is split into training and testing sets using an 80-20 split.

### 6. **Model Training**
    - A Logistic Regression model is trained on the training data.

### 7. **Model Evaluation**
    - The model's performance is evaluated using:
      - Accuracy score
      - Classification report
      - Confusion matrix

### 8. **Visualization**
    - WordCloud is generated to visualize the most frequent words in the dataset.
    - Sentiment distribution is plotted using a bar chart.
    - A heatmap is created for the confusion matrix.
    - The top 10 most frequent words are visualized using a bar chart.

## Libraries Used
- **Data Manipulation**: `pandas`, `numpy`
- **Text Processing**: `nltk`, `sklearn.feature_extraction.text.TfidfVectorizer`
- **Machine Learning**: `sklearn` (Logistic Regression, train-test split, metrics)
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`
- **Dataset Download**: `kagglehub`

## How to Run the Project
1. Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud kagglehub
    ```
2. Download the dataset using the `kagglehub` library.
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
