import streamlit as st
import joblib
import numpy as np
import os

# Load models safely
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

binary_model = joblib.load(os.path.join(BASE_DIR, "models", "binary_model.pkl"))
multi_model = joblib.load(os.path.join(BASE_DIR, "models", "multiclass_model.pkl"))
reg_model = joblib.load(os.path.join(BASE_DIR, "models", "regression_model.pkl"))

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("🩺 Diabetes Prediction App")
st.write("Enter clinical indicators to generate predictions.")

# Sidebar Inputs
st.sidebar.header("Patient Information")

fasting_glucose = st.sidebar.number_input(
    "Fasting Glucose", value=0, step=1, format="%d"
)

post_glucose = st.sidebar.number_input(
    "Postprandial Glucose", value=0, step=1, format="%d"
)

hba1c = st.sidebar.number_input(
    "HbA1c", value=0, step=1, format="%d"
)

insulin = st.sidebar.number_input(
    "Insulin Level", value=0, step=1, format="%d"
)

bmi = st.sidebar.number_input(
    "BMI", value=0, step=1, format="%d"
)

features = np.array([[fasting_glucose, post_glucose, hba1c, insulin, bmi]])

tabs = st.tabs(["Binary Prediction", "Multiclass Prediction", "Risk Score"])

# Binary
with tabs[0]:

    if st.button("Predict Diabetes"):
        pred = binary_model.predict(features)[0]

        if pred == 1:
            st.error("⚠️ High likelihood of Diabetes")
        else:
            st.success("✅ No Diabetes Detected")

# Multiclass
with tabs[1]:

    if st.button("Predict Diabetes Stage"):
        pred = multi_model.predict(features)[0]
        st.info(f"Predicted Stage: {pred}")

# Regression
with tabs[2]:

    if st.button("Predict Risk Score"):
        pred = reg_model.predict(features)[0]

        st.metric("Diabetes Risk Score", round(pred, 2))

        if pred < 10:
            st.success("Low Risk")
        elif pred < 20:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")