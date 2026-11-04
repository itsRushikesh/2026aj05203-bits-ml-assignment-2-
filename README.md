# Telco Customer Churn – ML Classification Dashboard

**BITS Pilani | M.Tech (AIML) | Machine Learning Assignment 2**

---

## a. Problem Statement

Predict whether a telecom customer will churn (leave the service) based on their
demographic profile, account information, and subscribed services. This is a
**binary classification** problem: the target variable `Churn` is `Yes` (1) or `No` (0).

Reducing customer churn is a key business problem in the telecom industry. By
identifying at-risk customers in advance, companies can take proactive retention
actions. The goal here is to compare five classical ML classifiers using six
standardised evaluation metrics and deploy the results as an interactive Streamlit
web application.

---

## b. Dataset Description

| Property                  | Detail                                                                  |
| ------------------------- | ----------------------------------------------------------------------- |
| **Name**            | Telco Customer Churn                                                    |
| **Source**          | Kaggle – IBM Sample Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) |
| **Task**            | Binary Classification                                                   |
| **Total Instances** | 7 043                                                                   |
| **Instances Used**  | 7 032 (11 rows dropped due to blank TotalCharges)                       |
| **Features**        | 19 (customerID dropped as it is a non-predictive identifier)            |
| **Target**          | `Churn` — did the customer leave? (Yes=1, No=0)                      |
| **Class Balance**   | ~73.5% No (retained), ~26.5% Yes (churned)                              |

### Features

| #  | Feature          | Type        | Description                                                              |
| -- | ---------------- | ----------- | ------------------------------------------------------------------------ |
| 1  | gender           | Categorical | Customer gender (Male/Female)                                            |
| 2  | SeniorCitizen    | Numeric     | Is the customer a senior citizen? (0/1)                                  |
| 3  | Partner          | Categorical | Has a partner? (Yes/No)                                                  |
| 4  | Dependents       | Categorical | Has dependents? (Yes/No)                                                 |
| 5  | tenure           | Numeric     | Number of months with the company                                        |
| 6  | PhoneService     | Categorical | Has phone service? (Yes/No)                                              |
| 7  | MultipleLines    | Categorical | Has multiple lines? (Yes/No/No phone service)                            |
| 8  | InternetService  | Categorical | Internet service type (DSL/Fiber optic/No)                               |
| 9  | OnlineSecurity   | Categorical | Has online security add-on?                                              |
| 10 | OnlineBackup     | Categorical | Has online backup add-on?                                                |
| 11 | DeviceProtection | Categorical | Has device protection add-on?                                            |
| 12 | TechSupport      | Categorical | Has tech support add-on?                                                 |
| 13 | StreamingTV      | Categorical | Has streaming TV?                                                        |
| 14 | StreamingMovies  | Categorical | Has streaming movies?                                                    |
| 15 | Contract         | Categorical | Contract type (Month-to-month/One year/Two year)                         |
| 16 | PaperlessBilling | Categorical | Uses paperless billing? (Yes/No)                                         |
| 17 | PaymentMethod    | Categorical | Payment method (Electronic check/Mailed check/Bank transfer/Credit card) |
| 18 | MonthlyCharges   | Numeric     | Monthly charge amount (USD)                                              |
| 19 | TotalCharges     | Numeric     | Total charges to date (USD)                                              |

Categorical features were label-encoded; all features scaled with `StandardScaler`.

---

## c. GitHub Repository

> **[https://github.com/YOUR_USERNAME/ml-classification-app](https://github.com/YOUR_USERNAME/ml-classification-app)**

*(Replace with your actual GitHub URL after pushing)*

### Repository Structure

```
ml-classification-app/
├── app.py                  # Streamlit web application
├── train_models.py         # Model training + artifact generation
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test split (~1407 rows, integer-encoded)
├── raw_data/               # Original Kaggle CSV (not committed to GitHub)
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── scaler.joblib
    ├── label_encoders.joblib
    └── feature_names.joblib
```

---

## d. Models Used

### Comparison Table

*(Fill in after running `python train_models.py`)*

| ML Model Name          | Accuracy         | AUC              | Precision        | Recall           | F1               | MCC              |
| ---------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Logistic Regression    | 0.7939           | **0.8345** | 0.6243           | 0.5642           | **0.5927** | **0.4562** |
| Decision Tree          | 0.7512           | 0.7291           | 0.5411           | 0.4225           | 0.4745           | 0.3189           |
| KNN                    | 0.7484           | 0.7782           | 0.5263           | 0.5348           | 0.5305           | 0.3587           |
| Naive Bayes            | 0.7328           | 0.8120           | 0.4982           | **0.7219** | 0.5895           | 0.4163           |
| Support Vector Machine | **0.7854** | 0.7843           | **0.6268** | 0.4759           | 0.5410           | 0.4109           |

### Observations

| ML Model                         | Observation about model performance                                                                                                                                                                                                                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**    | Good linear baseline for the churn problem. Tenure, contract type, and monthly charges show strong linear signal. Class imbalance (~73/27) causes lower recall for the minority Churn=Yes class. Coefficients provide interpretable feature importance.                                                                                                   |
| **Decision Tree**          | Captures non-linear interactions (e.g., Month-to-month contract AND high MonthlyCharges → higher churn). With `max_depth=10` it avoids overfitting. Feature importance reveals Contract and tenure as top splits. Slightly lower AUC than SVM/LR.                                                                                                      |
| **KNN**                    | Sensitive to the high-cardinality PaymentMethod and MultipleLines features. StandardScaler is critical here — without it, MonthlyCharges and TotalCharges would dominate. Moderate AUC; recall for churners is decent at k=7. Slower to predict at inference time.                                                                                       |
| **Naive Bayes**            | Gaussian NB assumes feature independence, which does not hold here (MonthlyCharges and TotalCharges are highly correlated). This causes lower precision. However, it achieves reasonable recall for Churn=Yes and trains in milliseconds.                                                                                                                 |
| **Support Vector Machine** | Best overall performer on this dataset. RBF kernel captures the non-linear boundary between churners and non-churners. Well-scaled features (StandardScaler) are essential. Highest AUC and MCC, confirming superior discrimination.                                                                                                                      |
| **Overall Winner**         | **Logistic Regression** — highest AUC (0.8345), F1 (0.5927), and MCC (0.4562), indicating the best overall discrimination and balance between precision and recall on this dataset. SVM achieves highest accuracy and precision but lower recall. Naive Bayes leads on recall (0.7219), making it useful if catching all churners is the priority. |

---

## e. Streamlit App Features

- CSV upload (test data only — no retraining)
- Model selection dropdown (5 models)
- 6 evaluation metric cards (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion Matrix heatmap
- ROC Curve plot
- Classification Report (per-class precision, recall, F1)
- All-models comparison table with colour highlighting (green = best)
- Bar chart comparison across all models

**Live App:** [https://YOUR-APP.streamlit.app](https://YOUR-APP.streamlit.app)
*(Replace with your Streamlit Cloud URL after deployment)*

---

## How to Run Locally

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/ml-classification-app.git
cd ml-classification-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates models/ folder and test_data.csv)
python train_models.py

# 4. Launch Streamlit app
streamlit run app.py
```

---

## Deployment Steps (Streamlit Community Cloud)

1. Push all files to GitHub — including the `models/` folder and `test_data.csv`
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **New App**
5. Select your repository
6. Choose branch: `main`
7. Set main file path: `app.py`
8. Click **Deploy**
9. Copy the generated `*.streamlit.app` URL for your submission PDF