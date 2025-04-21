import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load trained models
xgb_model = joblib.load("xgb_model.pkl")
lstm_model = tf.keras.models.load_model("lstm_model.h5")

# Dummy label encoders and scaler (replace with actual ones saved during training)
label_encoder_payment = LabelEncoder()
label_encoder_device = LabelEncoder()
label_encoder_merchant = LabelEncoder()
scaler = MinMaxScaler()

# Preprocessing logic for input
def preprocess_input(user_input):
    user_input['Payment Method'] = label_encoder_payment.transform(user_input['Payment Method'])
    user_input['Device Type'] = label_encoder_device.transform(user_input['Device Type'])
    user_input['Merchant Category'] = label_encoder_merchant.transform(user_input['Merchant Category'])

    numerical_columns = ['Transaction Amount (INR)', 'Previous Fraud Records', 'Credit Score', 'User Age']
    user_input[numerical_columns] = scaler.transform(user_input[numerical_columns])

    return np.array(user_input).reshape(1, 1, user_input.shape[1])

# UI
st.title("Fraud Detection Prediction")
form = st.form("prediction_form")

# Form inputs
col1, col2 = st.columns(2)
with col1:
    transaction_amount = st.number_input("Transaction Amount (INR)", value=0)
    previous_fraud = st.number_input("Previous Fraud Records", value=0)
    credit_score = st.number_input("Credit Score", value=600)
    user_age = st.number_input("User Age", value=30)
with col2:
    payment_method = st.selectbox("Payment Method", ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Wallet'])
    device_type = st.selectbox("Device Type", ['Mobile', 'Desktop', 'Tablet'])
    merchant_category = st.selectbox("Merchant Category", ['Electronics', 'Clothing', 'Groceries', 'Entertainment', 'Travel', 'Others'])
    transaction_location = st.text_input("Transaction Location")
    bank_name = st.text_input("Bank Name")
    high_risk_country = st.selectbox("Is High-Risk Country?", ['Yes', 'No'])

submit_button = form.form_submit_button("Predict")

if submit_button:
    # Construct DataFrame
    user_input = pd.DataFrame({
        "Transaction Amount (INR)": [transaction_amount],
        "Payment Method": [payment_method],
        "Device Type": [device_type],
        "Merchant Category": [merchant_category],
        "Transaction Status": [None],
        "Fraudulent": [None],
        "Previous Fraud Records": [previous_fraud],
        "Credit Score": [credit_score],
        "Transaction Location": [transaction_location],
        "Bank Name": [bank_name],
        "User Age": [user_age],
        "Is High-Risk Country": [high_risk_country]
    })

    processed = preprocess_input(user_input)
    lstm_features = lstm_model.predict(processed)
    prediction = xgb_model.predict(lstm_features)

    if prediction == 0:
        st.write("The transaction is **NOT Fraudulent**.")
    else:
        st.write("The transaction is **Fraudulent**.")
