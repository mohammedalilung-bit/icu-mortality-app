import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ICU Mortality Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load the saved model
@st.cache_resource
def load_model():
    with open('icu_mortality_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
le_icu = model_data['le_icu']
le_month = model_data['le_month']

# Title and description
st.title("🏥 ICU Mortality Prediction System")
st.markdown("""
This app predicts the number of mortality cases in ICU units based on clinical parameters.
Enter the ICU data below to get a prediction.
""")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 ICU Information")
    icu_name = st.selectbox(
        "ICU Unit Name",
        options=le_icu.classes_
    )
    
    month = st.selectbox(
        "Month",
        options=le_month.classes_
    )
    
    year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
    
    apache_score = st.slider(
        "APACHE II Score",
        min_value=0,
        max_value=50,
        value=25,
        help="Acute Physiology and Chronic Health Evaluation score"
    )
    
    total_cases = st.number_input(
        "Total Case Number per Month",
        min_value=1,
        max_value=200,
        value=50
    )

with col2:
    st.subheader("📊 Infection & Complications Data")
    
    vap = st.number_input(
        "Number of VAP Cases",
        min_value=0,
        max_value=50,
        value=0,
        help="Ventilator-Associated Pneumonia"
    )
    
    clabsi = st.number_input(
        "Number of CLABSI Cases",
        min_value=0,
        max_value=50,
        value=0,
        help="Central Line-Associated Bloodstream Infection"
    )
    
    cauti = st.number_input(
        "Number of CAUTI Cases",
        min_value=0,
        max_value=50,
        value=0,
        help="Catheter-Associated Urinary Tract Infection"
    )
    
    vent_days = st.number_input(
        "Ventilatory Days > 10 Days",
        min_value=0,
        max_value=100,
        value=0
    )
    
    icu_stay = st.number_input(
        "ICU Length of Stay > 10 Days",
        min_value=0,
        max_value=100,
        value=0
    )

# Prediction button
if st.button("🔮 Predict Mortality", type="primary", use_container_width=True):
    
    # Encode categorical variables
    icu_encoded = le_icu.transform([icu_name])[0]
    month_encoded = le_month.transform([month])[0]
    
    # Create feature array
    features = np.array([[
        apache_score, total_cases, vap, clabsi, cauti,
        vent_days, icu_stay, icu_encoded, month_encoded, year
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    predicted_cases = max(0, round(prediction))
    mortality_rate = (predicted_cases / total_cases) * 100 if total_cases > 0 else 0
    
    # Display results
    st.markdown("---")
    st.subheader("📈 Prediction Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Mortality Cases",
            value=f"{predicted_cases}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Predicted Mortality Rate",
            value=f"{mortality_rate:.1f}%",
            delta=None
        )
    
    with col3:
        risk_level = "🟢 Low" if mortality_rate < 15 else "🟡 Medium" if mortality_rate < 25 else "🔴 High"
        st.metric(
            label="Risk Level",
            value=risk_level,
            delta=None
        )
    
    # Additional insights
    st.markdown("### 🔍 Key Insights")
    total_infections = vap + clabsi + cauti
    
    insights = []
    if apache_score > 30:
        insights.append("⚠️ High APACHE II score indicates severe patient condition")
    if total_infections > 5:
        insights.append("⚠️ High infection rate detected - review infection control protocols")
    if icu_stay > 15:
        insights.append("⚠️ High number of prolonged ICU stays")
    if mortality_rate > 20:
        insights.append("⚠️ Predicted mortality rate is above average")
    
    if insights:
        for insight in insights:
            st.warning(insight)
    else:
        st.success("✅ All parameters within normal ranges")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**ICU Mortality Prediction System**

This machine learning model predicts mortality cases based on:
- APACHE II scores
- Patient case numbers
- Infection rates (VAP, CLABSI, CAUTI)
- Length of stay metrics

Model Type: Ensemble ML (Random Forest/Gradient Boosting)
""")

st.sidebar.header("📚 Instructions")
st.sidebar.markdown("""
1. Select the ICU unit and month
2. Enter APACHE II score and case numbers
3. Input infection and complication data
4. Click 'Predict Mortality' to get results
""")
