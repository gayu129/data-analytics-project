# data-analytics-project
# Customer Churn Analysis and Visualization

## Overview
This project analyzes customer churn data to identify patterns and trends that can help businesses retain customers. The dataset contains information about customer demographics, account details, and service usage.

## Dataset
The dataset contains 7,032 rows and 21 columns. Key columns include:
- `customerID`: Unique identifier for each customer.
- `gender`: Gender of the customer.
- `SeniorCitizen`: Indicates if the customer is a senior citizen (1) or not (0).
- `tenure`: Number of months the customer has been with the company.
- `MonthlyCharges`: Monthly charges for the customer.
- `TotalCharges`: Total charges incurred by the customer.
- `Churn`: Indicates whether the customer has churned (`Yes`) or not (`No`).

## Key Variables
- **Categorical Columns**: `['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']`
- **Numerical Columns**: `['tenure', 'MonthlyCharges', 'TotalCharges']`

## Steps Performed
1. **Data Loading and Exploration**:
    - Loaded the dataset using `pandas`.
    - Explored the dataset using `.info()`, `.describe()`, and `.head()` methods.
    - Checked the shape and column names of the dataset.

2. **Data Visualization**:
    - **Bar Plot**: Visualized the distribution of churn using a bar plot.
    - **Pie Chart**: Displayed the churn rate as a percentage.
    - **Box Plot**: Analyzed the relationship between `Churn` and `TotalCharges`.
    - **Histogram**: Visualized the distribution of `TotalCharges`.
    - **Scatter Plot**: Examined the relationship between `MonthlyCharges` and `TotalCharges`.
    - **Heatmap**: Displayed correlations between numerical features.
    - **Count Plots**: Visualized the distribution of categorical variables.
    - **Pair Plot**: Analyzed relationships between numerical features by churn.

3. **Data Preprocessing**:
    - Converted `TotalCharges` to numeric and handled missing values.
    - Standardized numerical features using `StandardScaler`.

4. **Clustering and Correlation Analysis**:
    - Created a clustermap of feature correlations.
    - Generated a clustermap of customers based on numerical features.

## Libraries Used
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib` and `seaborn`: For data visualization.
- `sklearn.preprocessing.StandardScaler`: For feature scaling.

## Visualizations
- **Churn Distribution**: Bar plot and pie chart showing the proportion of churned and non-churned customers.
- **Feature Distributions**: Histograms and box plots for numerical features.
- **Correlation Heatmap**: Highlighted relationships between numerical features.
- **Clustermap**: Grouped customers based on numerical features.

## Insights
- Customers with higher `TotalCharges` and `MonthlyCharges` are more likely to churn.
- Senior citizens and customers with month-to-month contracts have higher churn rates.
- Certain services, such as `OnlineSecurity` and `TechSupport`, are associated with lower churn rates.

## Future Work
- Build predictive models to classify customers as churned or non-churned.
- Perform feature engineering to create new insights.
- Explore additional datasets for deeper analysis.

## How to Run
1. Install the required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.
2. Load the dataset using the provided URL or a local file.
3. Execute the cells in the Jupyter Notebook sequentially to reproduce the analysis and visualizations.
