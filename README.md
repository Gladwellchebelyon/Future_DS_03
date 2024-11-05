# Future_DS_03
![image](https://github.com/user-attachments/assets/e0ffbf7e-352e-4c6b-aeb9-008c9d7c49b7)

# Loan Approval Prediction


# Table of Contents

    Project Overview
    Dataset
    Project Workflow
    Modeling and Evaluation
    Technologies Used
    Results
    Insights and Recommendations
    Conclusion
    Installation and Usage
    Acknowledgments

    
# Project Overview

The Loan Approval Prediction Project aims to create a machine learning model to predict whether a loan application will be approved based on applicant information. The project leverages data about loan applicants, including details about income, loan amount, credit history, and more. This tool is designed to assist financial institutions in making better, data-driven decisions about loan approvals.


# Dataset

The dataset used in this project includes several key features:

Loan_ID: Unique Loan ID

Gender, Married, Dependents: Applicant demographic information

Education, Self_Employed: Applicant work and education details

ApplicantIncome, CoapplicantIncome: Income information for the applicant and co-applicant

LoanAmount, Loan_Amount_Term: Details of the loan amount and the loan term

Credit_History: Applicant's credit history (1: meets guidelines, 0: does not meet guidelines)

Property_Area: Urban, Semiurban, or Rural area of the property

Loan_Status: Target variable; whether the loan was approved or not (Y/N)


# Data Source

The dataset was from Kaggle, creditcard.csv.


# Project Workflow

Data Exploration and Preprocessing:

Data Cleaning: Handling missing values and outliers.

Data Transformation: Encoding categorical variables and normalizing numerical features.

Feature Engineering:

Creating new features from existing ones (e.g., income-to-loan ratio).

Handling Class Imbalance:

Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes in the target variable.

 Modeling:

Tested several machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine (SVM).

Tuned hyperparameters using RandomizedSearchCV to improve model performance.

 Evaluation:

Used metrics such as accuracy, precision, recall, and f1-score to evaluate model performance.


# Modeling and Evaluation

Models Trained

Baseline Model: Logistic Regression

Advanced Models: Random Forest, Gradient Boosting, SVM, and XGBoost.

Evaluation Metrics

Accuracy: Percentage of correctly classified instances.

Precision, Recall, F1-Score: Metrics used to assess the model’s effectiveness in predicting loan approval and denial cases.

Final Model Performance

Best Model: Based on experimentation, the Random Forest model was selected, achieving an accuracy of 78.86%.


Classification Report: Provided details on precision, recall, and f1-score for each class.


# Technologies Used

Python: Programming language for implementation.

Pandas: Data manipulation and analysis.

Scikit-learn: Model building and evaluation.

XGBoost: Advanced machine learning model for better accuracy.

Imbalanced-learn (imblearn): For handling class imbalance with SMOTE.


# Results

The final model demonstrated a reasonably high accuracy in predicting loan approvals.

The use of SMOTE improved the model’s recall for the minority class (loan denials), although precision was slightly impacted.


# Insights and Recommendations

Income Levels and Loan Amount: A higher income-to-loan amount ratio was correlated with loan approvals, suggesting applicants with higher income or lower loan amounts are more likely to get approved.

Credit History: As expected, applicants with a credit history in good standing had a significantly higher chance of loan approval.

Property Area Impact: Loan approvals were slightly higher for properties in semi-urban areas, indicating that urban and semi-urban areas may have a higher approval rate.


Recommendation: Financial institutions can utilize this model to streamline the approval process, prioritize high-probability applications, and focus resources on applications that require further review.


# Conclusion

This project provides a foundational model to predict loan approvals. Future improvements could include:

Incorporating more applicant information such as employment type, asset details, and more granular credit history information.

Experimenting with additional models and techniques to enhance prediction accuracy and robustness.


# Installation and Usage

Install Dependencies:

bash
Copy code

pip install -r requirements.txt

Run the Project:
Clone the repository.

Load the dataset.

Run loan_approval_prediction.py to train the model and view evaluation metrics.


# Acknowledgments

We thank Kaggle for providing the data for this project.

