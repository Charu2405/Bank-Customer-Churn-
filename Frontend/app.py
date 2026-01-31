import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load trained model and scaler
# -----------------------------
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# App Title
# -----------------------------
st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will churn.")

st.divider()

# -----------------------------
# User Input Section
# -----------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 5)

balance = st.number_input("Account Balance", value=50000.0)

products_number = st.selectbox("Number of Products", [1, 2, 3, 4])

credit_card = st.selectbox("Has Credit Card", [0, 1])
active_member = st.selectbox("Is Active Member", [0, 1])

estimated_salary = st.number_input("Estimated Salary", value=60000.0)

# -----------------------------
# Encode Categorical Variables
# -----------------------------
country_map = {
    "France": 0,
    "Germany": 1,
    "Spain": 2
}

gender_map = {
    "Male": 1,
    "Female": 0
}

# -----------------------------
# Prepare Input for Prediction
# -----------------------------
input_data = np.array([[
    credit_score,
    country_map[country],
    gender_map[gender],
    age,
    tenure,
    balance,
    products_number,
    credit_card,
    active_member,
    estimated_salary
]])

input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction Button
# -----------------------------
st.divider()

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ Customer is likely to churn\n\n"
            f"**Churn Probability:** {probability:.2f}"
        )
    else:
        st.success(
            f"✅ Customer is likely to stay\n\n"
            f"**Retention Probability:** {1 - probability:.2f}"
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Machine Learning Project | Bank Customer Churn Prediction")
