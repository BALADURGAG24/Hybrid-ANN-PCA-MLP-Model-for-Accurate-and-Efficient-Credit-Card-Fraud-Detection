import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import joblib

# Load the trained model
model = load_model("models/fraud_model.h5")

# Load PCA transformer (you must save it during preprocessing)
pca = joblib.load("models/pca_transformer.pkl")

st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter the transaction details below:")

# Feature input form (V1 to V28 + Amount + Time)
features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

amount = st.number_input("Amount", value=0.0)
features.append(amount)

time = st.number_input("Time", value=0.0)
features.append(time)

if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)

    # Optional: Debugging shape
    # st.write(f"Input shape: {input_data.shape}")  # Should be (1, 30)

    # Apply PCA
    transformed_input = pca.transform(input_data)

    prediction = model.predict(transformed_input)[0][0]
    label = "Fraud" if prediction > 0.5 else "Not Fraud"

    st.subheader("Prediction:")
    st.success(f"🚨 {label} (Confidence: {prediction:.2f})")
