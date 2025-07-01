import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and column names
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("car_price_model_columns.pkl")
scaler = joblib.load("car_price_scaler.pkl")

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("Car Price Predictor")
st.write("Provide car details below to estimate the price.")

# Extract unique categories from column names
def get_categories(prefix):
    return sorted(col.split("_", 1)[1] for col in model_columns if col.startswith(prefix + "_"))


# Generate options from model columns
make_option = get_categories("Make")
model_option = get_categories("Model")
years =get_categories("Year")
engine_option = get_categories("Engine")
cylinders = get_categories("Cylinders")
mileages = get_categories("Mileage")
doors = get_categories("Doors")
fuel_option = get_categories("Fuel")
transmission_option = get_categories("Transmission")
trim_option = get_categories("Trim")
body_option = get_categories("Body")
ext_colors = get_categories("ExteriorColor")
int_colors = get_categories("InteriorColor")
drivetrains = get_categories("Drivetrain")

# Numeric inputs
years = st.number_input("Year", min_value=1990, max_value=2025, step=1, value=2024)
cylinders = st.selectbox("Cylinders", [4, 6, 8])
mileages = st.number_input("Mileage (km)", min_value=0.0, step=1.0, value=10.0)
doors = st.selectbox("Number of Doors", [2, 3, 4])

# Categorical inputs
make_selected = st.selectbox("Make", make_option)
model_selected = st.selectbox("Model", model_option)
fuel_selected = st.selectbox("Fuel Type", fuel_option)
transmission_selected = st.selectbox("Transmission", transmission_option)
trim_selected = st.selectbox("Trim", trim_option)
body_selected = st.selectbox("Body Type", body_option)
ext_colorSelected = st.selectbox("Exterior Color", ext_colors)
int_colorSelected = st.selectbox("Interior Color", int_colors)
drivetrains_selected = st.selectbox("Drivetrain", drivetrains)

# Prepare the input row
input_dict = {
    "Year": years,
    "Cylinders": cylinders,
    "Mileage": mileages,
    "Doors": doors,
    f"Make_{make_selected}": 1,
    f"Model_{model_selected}": 1,
    f"Fuel_{fuel_selected}": 1,
    f"Transmission_{transmission_selected}": 1,
    f"Trim_{trim_selected}": 1,
    f"Body_{body_selected}": 1,
    f"ExteriorColor_{ext_colorSelected}": 1,
    f"InteriorColor_{int_colorSelected}": 1,
    f"Drivetrain_{drivetrains_selected}": 1,
}

# Encode categorical variables using one-hot encoding
input_data_encoded = pd.get_dummies(input_dict)
# Align the input data with the model's expected input
expected_columns =joblib.load('car_price_model_columns.pkl')

# Ensure all expected columns are present in the input data
input_data_encoded = input_data_encoded.reindex(columns=expected_columns, fill_value=0)

# Scale the input data
input_data_scaled = scaler.transform(input_data_encoded)

# Make prediction
prediction = model.predict(input_data_scaled)

# Display the prediction
st.subheader("Predicted Price")
st.write(f"${prediction[0]:,.2f}")
# Display the input data for reference
st.subheader("Input Data")
st.write(input_dict)



