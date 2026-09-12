import streamlit as st
import pandas as pd
import joblib

# Force the page to use the wide layout
st.set_page_config(layout="wide")

# Inject custom CSS to lock the layout and prevent scrolling
st.markdown("""
    <style>
        /* Hide global scrollbar and lock body */
        html, body, [data-testid="stAppViewContainer"] {
            overflow: hidden;
            height: 100vh;
        }
        /* Reduce top padding for a tighter, cleaner fit */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load the saved model and scaler
# (Wrapped in try/except so code doesn't crash if files are missing locally)
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    model = None
    scaler = None

st.title("🏦 Bank Customer Churn Predictor")
st.write("Enter customer details to instantly estimate their churn probability.")

# Create two equal columns to split inputs and outputs side-by-side
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📋 Customer Details")
    
    # Nested columns inside the left pane to stack inputs tightly
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 92, 35)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 250000.0, 0.0)
        
    with sub_col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)
        
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    st.subheader("📊 Prediction Results")
    st.write("Click the button below to process the metrics.")
    
    if st.button("Predict Churn", use_container_width=True):
        if model is None or scaler is None:
            st.error("Model or Scaler files ('churn_model.pkl' / 'scaler.pkl') not found.")
        else:
            # Build a single-row dataframe matching the training feature structure
            input_dict = {
                'CreditScore': [credit_score],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_products],
                'HasCrCard': [1 if has_cr_card == "Yes" else 0],
                'IsActiveMember': [1 if is_active == "Yes" else 0],
                'EstimatedSalary': [estimated_salary],
                'Geography_Germany': [1 if geography == "Germany" else 0],
                'Geography_Spain': [1 if geography == "Spain" else 0],
                'Gender_Male': [1 if gender == "Male" else 0],
            }
            input_df = pd.DataFrame(input_dict)

            # Scale the input the same way training data was scaled
            input_scaled = scaler.transform(input_df)

            # Predict probability of churn (class 1)
            churn_proba = model.predict_proba(input_scaled)[0][1]

            st.metric(label="Churn Probability", value=f"{churn_proba:.1%}")

            if churn_proba >= 0.5:
                st.warning("⚠️ High Risk: This customer is highly likely to churn.")
            else:
                st.success("✅ Low Risk: This customer is stable.")
