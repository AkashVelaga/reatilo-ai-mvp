import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Reatilo AI - Retail Demand Predictor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Reatilo AI")
st.subheader("Smart Retail Demand Prediction Platform")

st.markdown("---")

# Load model
model = joblib.load("demand_model.pkl")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("Input Product Data")

    price = st.slider("Product Price", 1, 100, 20)

    promotion = st.selectbox(
        "Promotion Running?",
        ["No", "Yes"]
    )

    store_visitors = st.slider(
        "Daily Store Visitors",
        50, 500, 150
    )

    competitor_price = st.slider(
        "Competitor Price",
        1, 100, 18
    )

    promotion = 1 if promotion == "Yes" else 0

    predict_button = st.button("Predict Demand")

with col2:
    st.header("Prediction Result")

    if predict_button:

        features = np.array([[price, promotion, store_visitors, competitor_price]])

        prediction = model.predict(features)[0]

        st.success(f"Predicted Demand: {round(prediction,2)} units")

        if prediction > 80:
            st.warning("High demand expected. Increase inventory.")
        elif prediction < 40:
            st.error("Low demand expected. Reduce stock.")
        else:
            st.info("Moderate demand predicted.")

st.markdown("---")

st.write("AI-powered retail insights for smarter inventory decisions.")
