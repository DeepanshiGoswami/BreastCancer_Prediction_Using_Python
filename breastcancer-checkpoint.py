import streamlit as st
import numpy as np
import joblib  # For loading the trained model

# Load the trained model
model = joblib.load("cancerr_prediction.joblib")

# Title and description
st.title("Breast Cancer Detection")
st.write("""
This application predicts whether a tumor is **Malignant** or **Benign** 
based on input features. Please provide the details below.
""")

# Input fields for user features
st.header("Input Features")

mean_radius = st.number_input("Mean Radius", min_value=0.0, step=0.1)
mean_texture = st.number_input("Mean Texture", min_value=0.0, step=0.1)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, step=0.1)
mean_area = st.number_input("Mean Area", min_value=0.0, step=0.1)
mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, step=0.001)

mean_compactness = st.number_input("Mean Compactness", min_value=0.0, step=0.001)
mean_concavity = st.number_input("Mean Concavity", min_value=0.0, step=0.001)
mean_concavepoints = st.number_input("Mean Concave Points", min_value=0.0, step=0.001)
mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0, step=0.001)
mean_facialdimension = st.number_input("Mean Fractal Dimension", min_value=0.0, step=0.001)

# Prediction button
if st.button("Predict"):
    # Collect input data
    input_data = np.array([
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concavepoints, mean_symmetry, mean_facialdimension
    ]).reshape(1, -1)

    # Ensure no missing inputs
    if np.any(input_data == 0.0):
        st.error("Please fill out all the fields before making a prediction.")
    else:
        # Make prediction
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0] == 1:
            st.success("The tumor is **Malignant**.")
        else:
            st.success("The tumor is **Benign**.")
