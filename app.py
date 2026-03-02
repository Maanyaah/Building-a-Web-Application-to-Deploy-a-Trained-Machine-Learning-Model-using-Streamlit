# app.py

import streamlit as st
import pickle
import numpy as np

# -------------------------
# Load Trained Model
# -------------------------

model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Streamlit App UI
# -------------------------

st.title("💼 Developer Salary Prediction App")

st.write("Enter the details below to predict salary.")

# Example Input Fields (modify according to your dataset)

experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)

country = st.number_input("Country (Encoded Value)", min_value=0, step=1)

education = st.number_input("Education Level (Encoded Value)", min_value=0, step=1)

# -------------------------
# Prediction Button
# -------------------------

if st.button("Predict Salary"):

    # Arrange inputs in same order as training
    input_data = np.array([[experience, country, education]])

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Salary: {prediction[0]:,.2f}")
