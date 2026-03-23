import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open("model/fraud_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Load dataset
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    return pd.read_csv(url)

df = load_data()

st.title("💳 Credit-Card Fraud Detection Dashboard")

st.write("Analyze transaction risk using machine learning.")

# -----------------------------
# Transaction Inputs
# -----------------------------

amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, value=100.0)

time = st.slider(
    "⏰ Transaction Time (seconds of day)",
    0,
    86400,
    36000
)

location_risk = st.selectbox(
    "🌍 Location Risk Level",
    ["Low", "Medium", "High"]
)

risk_map = {"Low":0.2, "Medium":0.5, "High":0.9}
location_value = risk_map[location_risk]

# -----------------------------
# Demo Sample Buttons
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Normal Example"):
        sample = df[df["Class"] == 0].sample(1).drop("Class", axis=1).values[0]
        st.session_state["demo_sample"] = sample
        st.success("Normal example loaded")

with col2:
    if st.button("Load Fraud Example"):
        sample = df[df["Class"] == 1].sample(1).drop("Class", axis=1).values[0]
        st.session_state["demo_sample"] = sample
        st.warning("Fraud example loaded")

# Initialize history
if "history" not in st.session_state:
    st.session_state["history"] = []

# -----------------------------
# Analyze Transaction
# -----------------------------

if st.button("🔍 Analyze Transaction"):

    # Use demo sample if selected, otherwise random
    sample = st.session_state.get(
        "demo_sample",
        df.sample(1).drop("Class", axis=1).values[0]
    )

    sample[0] = time
    sample[-1] = amount
    sample[1] = sample[1] * location_value

    scaled_transaction = scaler.transform(sample.reshape(1,-1))

    prediction = model.predict(scaled_transaction)
    probability = model.predict_proba(scaled_transaction)[0]

    fraud_prob = probability[1]
    normal_prob = probability[0]

    st.subheader("Transaction Risk Result")

    if fraud_prob < 0.3:
        risk_level = "Low Risk"
        st.success("🟢 LOW RISK TRANSACTION")

    elif fraud_prob < 0.7:
        risk_level = "Medium Risk"
        st.warning("🟡 MEDIUM RISK TRANSACTION")

    else:
        risk_level = "High Risk"
        st.error("🔴 HIGH RISK - POSSIBLE FRAUD")

    st.write(f"Fraud Probability: {fraud_prob*100:.2f}%")

    # Chart
    fig, ax = plt.subplots()
    ax.bar(["Normal","Fraud"], [normal_prob,fraud_prob])
    ax.set_ylabel("Probability")
    ax.set_title("Fraud Risk Score")
    st.pyplot(fig)

    # Save transaction to history
    st.session_state["history"].append({
        "Amount": amount,
        "Risk Level": risk_level,
        "Fraud Probability (%)": round(fraud_prob*100,2)
    })

# -----------------------------
# Transaction History
# -----------------------------

if st.session_state["history"]:
    st.subheader("📜 Transaction History")

    history_df = pd.DataFrame(st.session_state["history"])

    st.dataframe(history_df)

# -----------------------------
# Fraud Alerts
# -----------------------------

if st.session_state["history"]:

    alerts = [
        tx for tx in st.session_state["history"]
        if tx["Risk Level"] == "High Risk"
    ]

    if alerts:

        st.subheader("🚨 Fraud Alerts")

        alerts_df = pd.DataFrame(alerts)

        st.dataframe(alerts_df)