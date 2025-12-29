
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# Load the trained model
model = tf.keras.models.load_model('linear_classification_model.keras')

# Load the scaler from file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for feature names
cancer = load_breast_cancer()
feature_names = cancer.feature_names

# Streamlit UI
st.title("Breast Cancer Classification App")
st.write("Enter the values for each feature below to predict if the tumor is malignant or benign.")

# Input fields for all features
def user_input_features():
    data = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", min_value=0.0, max_value=10000.0, value=0.0, format="%.4f")
        data.append(value)
    return np.array(data).reshape(1, -1)

input_data = user_input_features()

# Use the loaded scaler to transform input data
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    pred_class = int(np.round(prediction[0][0]))
    result = cancer.target_names[pred_class]
    st.write(f"### Prediction: {result}")
    st.write(f"Probability: {prediction[0][0]:.4f}")

st.markdown("---")
st.write("Model: linear_classification_model.keras | Powered by Streamlit & TensorFlow")

# To run this app, use the command:
# streamlit run app.py