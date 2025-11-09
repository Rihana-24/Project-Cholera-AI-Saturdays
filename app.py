import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------- Page Config ---------------------------
st.set_page_config(
    page_title="🦠 Nigeria Cholera Outbreak Cases Prediction",
    page_icon="🦠",
    layout="wide"
)

# --------------------------- CSS ---------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 10em;
        border-radius:10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------- Load Model ---------------------------
model_path = Path("models/best_model.joblib")
if not model_path.exists():
    st.error("❌ Model not found! Run train_save.py first.")
else:
    model = joblib.load(model_path)

# --------------------------- Input Sidebar ---------------------------
st.sidebar.header("🦠 Input Features")

# List of Nigerian States (without 'Unknown')
state_list = [
    'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 
    'Borno', 'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 
    'FCT', 'Gombe', 'Imo', 'Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 
    'Kogi', 'Kwara', 'Lagos', 'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 
    'Oyo', 'Plateau', 'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara'
]

year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2023, step=1)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1, step=1)
state = st.sidebar.selectbox("State", state_list)
cases = st.sidebar.number_input("Cases", min_value=0, value=0)
cfr = st.sidebar.number_input("CFR (%)", min_value=0.0, max_value=100.0, value=1.0)
population = st.sidebar.number_input("Population", min_value=0, value=1000)

# --------------------------- Prepare Input ---------------------------
input_df = pd.DataFrame({
    'Year': [year],
    'State': [state],
    'Cases': [cases],
    'CFR (%)': [cfr],
    'Population': [population]
})

# --------------------------- Prediction ---------------------------
if st.sidebar.button("Predict Cases per 100,000"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Cases/100,000: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --------------------------- Main Title ---------------------------
st.title("🦠 NIGERIA CHOLERA CASES PREDICTION")
st.markdown(
    """
    Enter the relevant details in the sidebar and click **Predict** to see cholera estimated Cases per 100,000.
    """
)

# --------------------------- Display Images ---------------------------
# --------------------------- Display Images ---------------------------
st.markdown("### Cholera Bacteria")
col1, col2 = st.columns(2)
col1.image("Statics/cholera_bacteria_image.png", use_container_width=True)
col2.image("Statics/cholera_on_petrishbox.png", use_container_width=True)


