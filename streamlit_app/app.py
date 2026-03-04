import streamlit as st
import joblib
import numpy as np

binary_model = joblib.load("../models/binary_model.pkl")
multi_model = joblib.load("../models/multiclass_model.pkl")
reg_model = joblib.load("../models/regression_model.pkl")

st.title("Diabetes Prediction App")

tab1, tab2, tab3 = st.tabs(["Binary", "Multiclass", "Risk Score"])

glucose_fasting = st.number_input("Fasting Glucose")
glucose_postprandial = st.number_input("Postprandial Glucose")
hba1c = st.number_input("HbA1c")
insulin = st.number_input("Insulin Level")
bmi = st.number_input("BMI")

features = np.array([[glucose_fasting, glucose_postprandial, hba1c, insulin, bmi]])

with tab1:
    if st.button("Predict Diabetes"):
        pred = binary_model.predict(features)
        st.write("Prediction:", pred)

with tab2:
    if st.button("Predict Stage"):
        pred = multi_model.predict(features)
        st.write("Stage:", pred)

with tab3:
    if st.button("Predict Risk Score"):
        pred = reg_model.predict(features)
        st.write("Risk Score:", pred)