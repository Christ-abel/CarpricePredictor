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
makes_options = get_categories("Make")
models_options = get_categories("Model")
engines_optioons = get_categories("Engine")
fuels_options = get_categories("Fuel")
transmissions_options = get_categories("Transmission")
trims_options = get_categories("Trim")
bodies_options = get_categories("Body")
ext_colors_options = get_categories("ExteriorColor")
int_colors_options = get_categories("InteriorColor")
drivetrains_options = get_categories("Drivetrain")

# Numeric inputs
years = st.number_input("Year", min_value=1990, max_value=2025, step=1, value=2024)
cylinders = st.selectbox("Cylinders", [4, 6, 8])
mileages = st.number_input("Mileage (km)", min_value=0.0, step=1.0, value=10.0)
doors = st.selectbox("Number of Doors", [2, 3, 4])

# Categorical inputs
makes = st.selectbox("Make", makes_options)
models = st.selectbox("Model", models_options)
fuels = st.selectbox("Fuel Type", fuels_options)
transmissions = st.selectbox("Transmission", transmissions_options)
trims = st.selectbox("Trim", trims_options)
bodies = st.selectbox("Body Type", bodies_options)
ext_color = st.selectbox("Exterior Color", ext_colors_options)
int_color = st.selectbox("Interior Color", int_colors_options)
drivetrains = st.selectbox("Drivetrain", drivetrains_options)

# Prepare the input row
input_dict = {
    "Year": years,
    "Cylinders": cylinders,
    "Mileage": mileages,
    "Doors": doors,
    f"Make_{makes}": 1,
    f"Model_{models}": 1,
    f"Fuel_{fuels}": 1,
    f"Transmission_{transmissions}": 1,
    f"Trim_{trims}": 1,
    f"Body_{bodies}": 1,
    f"ExteriorColor_{ext_color}": 1,
    f"InteriorColor_{int_color}": 1,
    f"Drivetrain_{drivetrains}": 1,
}

# Encode categorical variables using one-hot encoding
input_data_encoded = pd.DataFrame([input_dict])
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
st.write(f"$ {prediction[0]:,.2f}")
# Display the input data for reference
st.subheader("Input Data")
st.write(input_dict)





