# 📊 Telecom Customer Churn Prediction

A complete **Machine Learning and Flask web application** for predicting customer churn in the telecommunications sector.

🌐 **Live Demo:** https://telecom-churn-b1lm.onrender.com

The project covers the complete Data Science workflow, from data preprocessing and exploratory analysis to machine learning model development, threshold optimization, evaluation, and deployment as a web application.

---

## 🚀 Project Overview

Customer churn is a major challenge for telecommunications companies. Identifying customers who are likely to leave can help companies take preventive actions and improve customer retention.

This project aims to build a machine learning system capable of predicting whether a customer is likely to **churn** based on demographic information, services, contract type, payment method, tenure, and billing information.

The final model is integrated into a **Flask web application**, allowing users to enter customer information and obtain a churn prediction and probability.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze customer churn data
- Perform data cleaning and preprocessing
- Explore relationships between customer characteristics and churn
- Encode categorical variables
- Prepare numerical and binary variables
- Train several classification models
- Optimize model hyperparameters
- Optimize the classification threshold using Out-of-Fold predictions
- Evaluate models using appropriate classification metrics
- Select a model for deployment
- Build a Flask web application
- Deploy the application as a web service

---

## 📁 Dataset

The project uses a telecommunications customer churn dataset.

The target variable is:

```text
Churn

```

It is transformed into a binary variable:

```text
No  → 0
Yes → 1

```

The `customerID` column is not used as a predictive feature.

The dataset is divided into training and testing sets using an **80/20 stratified split** with `random_state=42`.

---

## 🔎 Features

The final preprocessing pipeline uses three groups of variables.

### Numerical features

```text
tenure
MonthlyCharges
TotalCharges

```

### Binary features

```text
SeniorCitizen

```

### Categorical features

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod

```

---

# 🧹 Data Preprocessing

The preprocessing pipeline is implemented using Scikit-learn's `ColumnTransformer`.

The numerical and binary variables are kept as numerical variables, while categorical variables are transformed using **One-Hot Encoding**.

```python
preprocessor_final = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("bin", "passthrough", binary_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

```

The `OneHotEncoder` uses:

```python
handle_unknown="ignore"

```

which allows the deployed application to handle previously unseen categorical values safely.

---

# 🤖 Machine Learning Models

Several classification approaches were evaluated during the project, including:

- Logistic Regression
- Random Forest
- XGBoost
- Stacking

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Because churn prediction is a classification problem with potentially different costs for false positives and false negatives, the classification threshold was also optimized instead of relying exclusively on the default threshold of `0.5`.

---

# ⚙️ XGBoost

XGBoost was tuned using `GridSearchCV` with **ROC-AUC** as the optimization metric.

The final XGBoost model was then trained using the encoded training data.

```python
best_xgb.fit(X_train_encoded, y_train)

```

The model generates churn probabilities using:

```python
predict_proba()

```

and the final class prediction is obtained using the optimized threshold.

---

# 🎚️ Threshold Optimization

Instead of automatically using:

```text
threshold = 0.5

```

the project determines an optimal threshold using **Out-of-Fold predictions**.

The threshold is selected based on the F1-score obtained from the precision-recall curve.

For the final XGBoost model, the optimized threshold is approximately:

```text
0.587

```

This threshold is saved and reused by the Flask application.

---

# 📊 Model Performance

The final models were evaluated on the independent test set.

| ModelThresholdAccuracyPrecisionRecallF1-scoreROC-AUC |       |            |            |            |            |            |
| ---------------------------------------------------- | ----- | ---------- | ---------- | ---------- | ---------- | ---------- |
| XGBoost                                              | 0.587 | 0.7683     | 0.5476     | 0.7380     | **0.6287** | **0.8403** |
| Stacking                                             | 0.347 | **0.7740** | **0.5586** | 0.7139     | 0.6268     | **0.8403** |
| Random Forest                                        | 0.317 | 0.7569     | 0.5297     | **0.7620** | 0.6250     | 0.8383     |
| Logistic Regression                                  | 0.324 | 0.7534     | 0.5257     | 0.7380     | 0.6140     | 0.8363     |

### XGBoost test performance

```text
Accuracy  : 0.7683
Precision : 0.5476
Recall    : 0.7380
F1-score  : 0.6287
ROC-AUC   : 0.8403

```

These results correspond to the final test-set evaluation from the project.

> **Note:** The model comparison shows different trade-offs between accuracy, precision, recall, F1-score and ROC-AUC. The XGBoost model is used for the deployed application with its optimized threshold of approximately `0.587`.

---

# 🌐 Flask Web Application

The trained model is integrated into a Flask web application.

The user can enter information about a customer through a web interface, including:

- Customer demographics
- Tenure
- Monthly charges
- Total charges
- Telephone services
- Internet services
- Online security
- Technical support
- Streaming services
- Contract type
- Billing information
- Payment method

The application then:

1. Collects the user's inputs
2. Creates a Pandas DataFrame
3. Applies the saved preprocessing pipeline
4. Generates a churn probability
5. Applies the optimized XGBoost threshold
6. Displays the prediction

---

# 🖥️ Application Structure

```text
telecom-churn-flask/
│
├── app.py
│
├── model/
│   ├── preprocessor.pkl
│   ├── xgboost_churn.pkl
│   └── threshold.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
│
└── README.md

```

---

# 💾 Saved Machine Learning Files

The trained components are saved using `joblib`.

### Preprocessing pipeline

```text
model/preprocessor.pkl

```

### XGBoost model

```text
model/xgboost_churn.pkl

```

### Optimized classification threshold

```text
model/threshold.pkl

```

Keeping the preprocessing pipeline and model together ensures that the same transformations used during training are also applied when making predictions in the web application.

---

# 🛠️ Technologies Used

### Programming

- Python

### Data Science

- Pandas
- NumPy
- Scikit-learn
- XGBoost

### Visualization / Analysis

- Matplotlib
- Seaborn

### Machine Learning

- Logistic Regression
- Random Forest
- XGBoost
- Stacking
- GridSearchCV
- Cross-validation
- Out-of-Fold predictions

### Web Development

- Flask
- HTML
- CSS

### Deployment

- GitHub
- Render
- Gunicorn

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/walidsikhelifa/telecom-churn-flask.git

```

Move into the project directory:

```bash
cd telecom-churn-flask

```

Create a virtual environment:

```bash
python -m venv venv

```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate

```

Install the dependencies:

```bash
pip install -r requirements.txt

```

---

# ▶️ Run the Application Locally

Start the Flask application:

```bash
python app.py

```

The application will be available at:

```text
http://127.0.0.1:5000

```

Open this address in your web browser.

---

# ☁️ Deployment

The application is designed to be deployed as a Flask web service.

### Build Command

```bash
pip install -r requirements.txt

```

### Start Command

```bash
gunicorn app:app

```

The application can then be deployed using a cloud platform such as Render.

---

# 🔮 Prediction Output

The application returns:

### Customer likely to stay

```text
Client susceptible de rester

```

or:

### Customer likely to churn

```text
Client susceptible de partir

```

It also displays the estimated churn probability.

For example:

```text
Churn probability: 72.35%

```

The prediction is determined using the optimized XGBoost threshold rather than simply using `0.5`.

---

# 📌 Project Workflow

The overall workflow can be summarized as:

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Preparation
     │
     ▼
Train / Test Split
     │
     ▼
Preprocessing
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     ├── Random Forest
     ├── XGBoost
     └── Stacking
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Cross-Validation / OOF Predictions
     │
     ▼
Threshold Optimization
     │
     ▼
Model Evaluation
     │
     ▼
XGBoost Model
     │
     ▼
Flask Application
     │
     ▼
Web Deployment

```

---

# 📈 Business Use Case

A telecom company can use this type of system to identify customers with a higher estimated probability of churn.

Potential applications include:

- Customer retention campaigns
- Targeted offers
- Proactive customer support
- Contract renewal strategies
- Identification of high-risk customer profiles

The model should be considered a decision-support tool rather than a replacement for business judgment.

---

# 🔐 Important Note

The model's predictions are based on patterns learned from the available training data.

Machine learning predictions can be affected by:

- Data quality
- Feature distribution
- Class imbalance
- Model assumptions
- Changes in customer behavior
- Differences between historical and future data

Therefore, predictions should be interpreted together with relevant business context.

---

# 👨‍💻 Author

**Walid Si Khelifa**

Master 2 — Probability and Statistics
Université Mouloud Mammeri de Tizi-Ouzou (UMMTO), Algeria

---

# 📄 License

This project is intended for educational and portfolio purposes.