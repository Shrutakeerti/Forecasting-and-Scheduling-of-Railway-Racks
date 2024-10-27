import streamlit as st
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load(r'D:\Forecasting and Scheduling of Railway Racks\xgboost_model.pkl')  # Make sure this path is correct

# Streamlit UI
st.title("Railway Rack Demand Forecasting")
st.write("Enter details below to forecast demand:")

# Input fields
travel_from = st.selectbox("Travel From", options=['City A', 'City B'])  # Replace with actual locations
travel_to = st.selectbox("Travel To", options=['City C', 'City D'])
car_type = st.selectbox("Car Type", options=['Type 1', 'Type 2'])
payment_method = st.selectbox("Payment Method", options=['Method A', 'Method B'])
travel_date = st.date_input("Travel Date", value=datetime.date.today())
travel_time = st.time_input("Travel Time", value=datetime.time(8, 0))

# Convert inputs to model-compatible format
day = travel_date.day
month = travel_date.month
year = travel_date.year
weekday = travel_date.weekday()
hour = travel_time.hour
minute = travel_time.minute

# Label Encoding for Categorical Data
label_encoders = {}
for column in ['payment_method', 'travel_from', 'travel_to', 'car_type']:
    le = LabelEncoder()
    le.fit([travel_from, travel_to, car_type, payment_method])  # Ensure consistent encoding
    label_encoders[column] = le

payment_method_encoded = label_encoders['payment_method'].transform([payment_method])[0]
travel_from_encoded = label_encoders['travel_from'].transform([travel_from])[0]
travel_to_encoded = label_encoders['travel_to'].transform([travel_to])[0]
car_type_encoded = label_encoders['car_type'].transform([car_type])[0]

# Prepare the input data
input_data = pd.DataFrame([[payment_method_encoded, travel_from_encoded, travel_to_encoded, car_type_encoded, day, month, year, weekday, hour, minute]],
                          columns=['payment_method', 'travel_from', 'travel_to', 'car_type', 'day', 'month', 'year', 'weekday', 'travel_hour', 'travel_minute'])

# Make prediction
prediction = model.predict(input_data)

st.write(f"Predicted Rack Demand: {round(prediction[0])}")
