import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Reatilo AI", layout="wide")

st.title("📊 Reatilo AI")
st.subheader("Smart Retail Demand Prediction Platform")

# Load model
model = joblib.load("demand_model.pkl")

st.markdown("---")

# Layout columns
col1, col2 = st.columns(2)

# LEFT SIDE → INPUTS
with col1:
    st.header("Input Product Data")

    price = st.slider("Product Price", 1, 100, 20)

    promotion = st.selectbox(
        "Promotion Running?",
        ["No", "Yes"]
    )

    visitors = st.slider(
        "Daily Store Visitors",
        50, 500, 150
    )

    competitor_price = st.slider(
        "Competitor Price",
        1, 100, 18
    )

    promotion = 1 if promotion == "Yes" else 0

    predict = st.button("Predict Demand")

# RIGHT SIDE → RESULTS + GRAPH
with col2:

    st.header("Prediction Output")

    if predict:

        features = np.array([[price, promotion, visitors, competitor_price]])

        prediction = model.predict(features)[0]

        st.success(f"Predicted Demand: {round(prediction,2)} units")

        # Create simple graph
        data = pd.DataFrame({
            "Metric": ["Price", "Visitors", "Competitor Price", "Predicted Demand"],
            "Value": [price, visitors, competitor_price, prediction]
        })

        fig, ax = plt.subplots()

        ax.bar(data["Metric"], data["Value"])

        ax.set_title("Retail Demand Insights")

        st.pyplot(fig)

st.markdown("---")

st.write("AI-powered retail insights to help businesses optimize inventory.")
