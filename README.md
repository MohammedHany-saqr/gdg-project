# Data Science Track - GDG CIC Graduation Project
![Alt text](C:\Users\HP\Documents\WhatsApp Image 2025-08-28 at 19.19.08_91545080.jpg)



# 📌 Customer Churn Prediction

**Authors:** Mohamed Kasm • Mohamed Saqr • Ziad Abdallbasset

**Created:** 2025-08-26 (UTC)

**Dataset:** Telco Customer Churn (Kaggle)  
**Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn


# 📌 Overview

This project focuses on predicting customer churn—the likelihood that a customer will stop using a company’s products or services. By identifying customers at high risk of leaving, businesses can take proactive measures to improve retention, reduce churn rates, and increase lifetime value.

# 🎯 Objectives

Perform exploratory data analysis (EDA) to understand customer behavior.

Identify key features influencing churn.

Build and evaluate machine learning models for churn prediction.

Provide business insights and actionable recommendations.

# 📂 Project Structure
├── data:                 Raw and processed datasets

├── notebooks:            Jupyter notebooks for EDA and modeling

├── src:                  Python scripts for preprocessing, training, evaluation

├── models:               Trained machine learning models

├── reports:              EDA reports, model performance results

├── requirements.txt:      Project dependencies

└── README.md:             Project documentation


# 🛠️ Tools & Technologies

Languages: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn

Visualization: Matplotlib, Seaborn, Plotly

Environment: Jupyter Notebook / VS Code

# 📊 Methodology

Data Preprocessing

Handle missing values and outliers

Encode categorical variables

Scale/normalize features

Exploratory Data Analysis (EDA)

Analyze customer demographics & usage patterns

Visualize churn distribution and correlations

Feature Engineering

Create meaningful features from raw data

Select top predictors using feature importance

Modeling

Train ML models (Logistic Regression, Random Forest, XGBoost, etc.)

Handle class imbalance (SMOTE, class weights)

Optimize hyperparameters using GridSearchCV/RandomizedSearchCV

Evaluation

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

Compare model performance

Business Insights

Interpret key drivers of churn

Recommend retention strategies

# 📈 Expected Outcomes

A robust predictive model for customer churn.

Insights into why customers leave.

Data-driven strategies to improve retention.

# 🚀 How to Run the Project

Clone this repository:

git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction


Create a virtual environment & install dependencies:

pip install -r requirements.txt


Run Jupyter notebooks for EDA & model training:

jupyter notebook


Check reports/ for results and insights.

# 📌 Future Improvements

Deploy model as an API (Flask/FastAPI).

Build an interactive dashboard for churn monitoring.

Integrate with real-time customer data streams. 
  

# Final Notes
- The tuned XGBoost usually performs best on this dataset by ROC-AUC.
- Always validate results with proper cross-validation and consider class imbalance techniques (e.g., stratified splits, class weights, or SMOTE) if needed.
- Next steps: threshold tuning to balance precision/recall for the business objective.

