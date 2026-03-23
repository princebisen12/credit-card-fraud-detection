# 💳 Credit Card Fraud Detection System

A Machine Learning based web application that detects fraudulent credit card transactions using a trained Random Forest model and an interactive Streamlit dashboard.

---

## 🚀 Live Demo

🔗 https://your-streamlit-app-link

---

## 📊 Project Overview

Credit card fraud is a major problem in financial systems.  
This project builds a **machine learning model to detect fraudulent transactions** and provides an **interactive dashboard for risk analysis**.

The application predicts whether a transaction is **Normal or Fraudulent** and shows:

- Fraud probability
- Risk level
- Fraud alerts
- Transaction history

---

## 🧠 Machine Learning Pipeline

1. Data Loading
2. Exploratory Data Analysis
3. Handling Class Imbalance using **SMOTE**
4. Feature Scaling using **StandardScaler**
5. Model Training using **Random Forest**
6. Model Evaluation
7. Model Deployment with **Streamlit**

---

## 📂 Project Structure
credit-card-fraud-detection
│
├── app.py
├── requirements.txt
├── README.md
│
├── data
│ └── creditcard.csv
│
├── model
│ ├── fraud_model.pkl
│ └── scaler.pkl
│
├── notebook
│ └── fraud_detection.ipynb


---

## ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Plotly
- Streamlit
- SMOTE (imbalanced-learn)

---

## 🖥️ Features

✔ Fraud risk prediction  
✔ Fraud probability visualization  
✔ Transaction history tracking  
✔ Fraud alert detection  
✔ Interactive dashboard  

---

## 📈 Dashboard Preview

![Dashboard Screenshot](dashboard.png)

---

## ▶️ How to Run the Project

Clone the repository:

git clone https://github.com/your-username/credit-card-fraud-detection.git

Go to the project folder:

cd credit-card-fraud-detection

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Open in browser:

http://localhost:8501

---

## 📊 Dataset
Dataset used:  
Credit Card Fraud Detection Dataset

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
---
## 👨‍💻 Author
Prince Singh
Machine Learning & Data Science Enthusiast
---