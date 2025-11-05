import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="ICU Mortality Predictor", page_icon="🏥", layout="wide")

@st.cache_resource
def load_model():
    with open('icu_mortality_model.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
le_icu = model_data['le_icu']
le_month = model_data['le_month']

st.title("🏥 ICU Mortality Prediction System")
st.write("Enter ICU data to predict monthly mortality cases")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Basic Information")
    icu_name = st.selectbox("ICU Unit Name", le_icu.classes_)
    month = st.selectbox("Month", le_month.classes_)
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
    apache_score = st.slider("APACHE II Score", 0, 50, 25)
    total_cases = st.number_input("Total Cases per Month", 1, 200, 50)

with col2:
    st.subheader("📊 Infection & Complications")
    vap = st.number_input("VAP Cases", 0, 50, 0)
    clabsi = st.number_input("CLABSI Cases", 0, 50, 0)
    cauti = st.number_input("CAUTI Cases", 0, 50, 0)
    vent_days = st.number_input("Ventilatory Days > 10", 0, 100, 0)
    icu_stay = st.number_input("ICU Stay > 10 Days", 0, 100, 0)

if st.button("🔮 Predict Mortality", type="primary", use_container_width=True):
    icu_encoded = le_icu.transform([icu_name])[0]
    month_encoded = le_month.transform([month])[0]
    features = np.array([[apache_score, total_cases, vap, clabsi, cauti, 
                         vent_days, icu_stay, icu_encoded, month_encoded, year]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    predicted_cases = max(0, round(prediction))
    mortality_rate = (predicted_cases / total_cases) * 100 if total_cases else 0
    
    st.markdown("---")
    st.subheader("📈 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Mortality Cases", predicted_cases)
    with col2:
        st.metric("Mortality Rate", f"{mortality_rate:.1f}%")
    with col3:
        risk = "🟢 Low" if mortality_rate < 15 else "🟡 Medium" if mortality_rate < 25 else "🔴 High"
        st.metric("Risk Level", risk)

st.sidebar.header("ℹ️ About")
st.sidebar.info("ICU Mortality Prediction System using Machine Learning")
