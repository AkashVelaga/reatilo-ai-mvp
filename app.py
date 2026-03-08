import streamlit as st
import xgboost as xgb
import numpy as np

st.title("Reatilo-AI Demand Forecast MVP")

# Load model
model = xgb.Booster()
model.load_model("demand_model.json")

st.write("Enter input features:")

feature1 = st.number_input("Feature 1", value=10)
feature2 = st.number_input("Feature 2", value=2)
feature3 = st.number_input("Feature 3", value=1)
feature4 = st.number_input("Feature 4", value=0)

if st.button("Predict Demand"):
    data = np.array([[feature1, feature2, feature3, feature4]])
    dmatrix = xgb.DMatrix(data)
    prediction = model.predict(dmatrix)
    st.success(f"Predicted Demand: {prediction[0]:.2f}")
