import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pickle files for model, scaler, encoders, features
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
label_encoder_icu = data['label_encoder_icu']
label_encoder_month = data['label_encoder_month']
feature_cols = data['features']

st.title('ICU Mortality Risk Prediction')

uploaded_file = st.file_uploader('Upload your Excel file', type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name='ML PROJECT')

    # Encode categorical features
    df['ICUENCODED'] = label_encoder_icu.transform(df['ICU NAME'])
    df['MONTHENCODED'] = label_encoder_month.transform(df['MONTH'])

    X = df[feature_cols]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)
    df['MORTALITY RISK PREDICTION'] = np.where(preds == 1, 'High', 'Low')

    st.subheader('Prediction Results')
    st.dataframe(df[['ICU NAME', 'MONTH', 'YEAR', 'MORTALITY RISK PREDICTION']])
