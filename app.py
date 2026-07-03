import streamlit as st
import numpy as np
import pandas as pd
import pickle
 
# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor for Women",
    page_icon="💗",
    layout="centered",
    initial_sidebar_state="expanded"
)
 
# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Roboto', 'Segoe UI', sans-serif;
    }
 
    .stApp {
        background-color: #fff5f8;
    }
 
    /* Header bar */
    .header-bar {
        background-color: #ffffff;
        border-bottom: 1px solid #f8bbd0;
        padding: 18px 0 16px 0;
        margin-bottom: 20px;
        text-align: center;
    }
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #4a0025;
        margin: 0;
    }
    .main-title span {
        color: #d81b60;
    }
    .sub-title {
        font-size: 14px;
        color: #8e5266;
        margin-top: 4px;
    }
 
    /* Women-focused info banner */
    .women-banner {
        background-color: #fce4ec;
        border: 1px solid #f8bbd0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #6a1b3d;
        margin-bottom: 22px;
    }
 
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #fff0f5;
        border-right: 1px solid #f8bbd0;
    }
    section[data-testid="stSidebar"] * {
        color: #4a0025 !important;
    }
    .sidebar-heading {
        font-size: 15px;
        font-weight: 700;
        color: #4a0025 !important;
        margin-bottom: 2px;
    }
    .sidebar-caption {
        font-size: 12px;
        color: #8e5266 !important;
        line-height: 1.5;
    }
 
    /* Section headings on main area */
    .section-heading {
        font-size: 14px;
        font-weight: 700;
        color: #4a0025;
        margin-bottom: 6px;
        margin-top: 4px;
    }
 
    /* Inputs */
    .stNumberInput input {
        background-color: #ffffff !important;
        color: #4a0025 !important;
        border: 1px solid #f48fb1 !important;
        border-radius: 6px !important;
    }
 
    /* Slider - filled track uses a light-to-deep pink gradient,
       so it visually deepens in color as the value increases */
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #f8bbd0, #ad1457) !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #d81b60 !important;
        border-color: #d81b60 !important;
    }
 
    /* Predict button */
    div.stButton > button {
        width: 100%;
        background-color: #d81b60;
        color: #ffffff;
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 0.3px;
        padding: 10px 8px;
        border-radius: 6px;
        border: none;
        white-space: normal;
        height: auto;
        line-height: 1.4;
        transition: background-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 1px 2px rgba(173,20,87,0.3);
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #ad1457;
        box-shadow: 0 1px 3px rgba(173,20,87,0.45);
    }
 
    /* Result cards */
    .result-card {
        border-radius: 8px;
        padding: 22px 26px;
        margin-top: 16px;
        border: 1px solid #f8bbd0;
        box-shadow: 0 1px 3px rgba(173,20,87,0.12);
    }
    .result-card.low {
        background-color: #e6f4ea;
        border-color: #ceead6;
    }
    .result-card.moderate {
        background-color: #fef7e0;
        border-color: #fde293;
    }
    .result-card.high {
        background-color: #fce4ec;
        border-color: #f48fb1;
    }
    .result-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8e5266;
    }
    .result-value {
        font-size: 24px;
        font-weight: 700;
        margin: 4px 0 2px 0;
    }
    .result-value.low { color: #188038; }
    .result-value.moderate { color: #b06000; }
    .result-value.high { color: #ad1457; }
    .result-sub {
        font-size: 13px;
        color: #8e5266;
    }
 
    /* Summary table */
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 14px;
        font-size: 13px;
    }
    .summary-table td {
        padding: 7px 4px;
        border-top: 1px solid #fce4ec;
        color: #4a0025;
    }
    .summary-table td:first-child {
        color: #8e5266;
    }
    .summary-table td:last-child {
        text-align: right;
        font-weight: 500;
    }
 
    /* Disclaimer box */
    .disclaimer {
        background-color: #fff8e1;
        border: 1px solid #ffe58f;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 12px;
        color: #7a5b00;
        margin-top: 22px;
    }
 
    .footer {
        text-align: center;
        font-size: 12px;
        color: #b17692;
        padding: 20px 0 6px 0;
    }
</style>
""", unsafe_allow_html=True)
 
# ---------------------- HEADER ----------------------
st.markdown("""
<div class="header-bar">
    <div class="main-title">💗 <span>Diabetes</span> Risk Predictor for Women</div>
    <div class="sub-title">A machine-learning based diabetes screening tool designed for women's health</div>
</div>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class="women-banner">
    💗 <b>Built for women:</b> This tool uses a model trained specifically on female
    patient health data (including pregnancy history), so results are tailored to
    women rather than the general population.
</div>
""", unsafe_allow_html=True)
 
# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", 'rb'))
    sc = pickle.load(open("sc.pkl", 'rb'))
    return model, sc
 
model, sc = load_artifacts()
 
# ---------------------- SIDEBAR ----------------------
st.sidebar.markdown('<div class="sidebar-heading">About this tool</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-caption">
This tool estimates the likelihood of diabetes in women based on common clinical
measurements, using a machine learning model trained on female patient health data.
<br><br>
Adjust the values in the main panel to match the patient's readings, then
click <b>Predict</b> to see the result.
</div>
""", unsafe_allow_html=True)
 
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-heading">Reference ranges</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-caption">
<b>Glucose:</b> 70–140 mg/dL (fasting, normal)<br>
<b>Blood Pressure:</b> below 120 mm Hg (normal)<br>
<b>BMI:</b> 18.5–24.9 (normal weight)
</div>
""", unsafe_allow_html=True)
 
# ---------------------- INPUTS ----------------------
col1, col2 = st.columns(2)
 
with col1:
    st.markdown('<div class="section-heading">Patient History</div>', unsafe_allow_html=True)
    Pregnancies = st.slider("Pregnancies", 0, 17, 1)
    BloodPressure = st.slider("Blood Pressure (mm Hg)", 40, 140, 72)
    Insulin = st.slider("Insulin (mu U/mL)", 15, 300, 80)
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.05,
        max_value=3.0,
        value=0.47,
        step=0.001,
        format="%.3f",
        help="A score reflecting diabetes likelihood based on family history."
    )
 
with col2:
    st.markdown('<div class="section-heading">Vitals & Measurements</div>', unsafe_allow_html=True)
    Glucose = st.slider("Glucose (mg/dL)", 50, 200, 120)
    SkinThickness = st.slider("Skin Thickness (mm)", 7, 99, 20)
    BMI = st.slider("BMI", 18.0, 50.0, 32.0, step=0.1)
    if BMI < 18.5:
        bmi_category = "Underweight"
    elif BMI < 25:
        bmi_category = "Normal"
    elif BMI < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    st.caption(f"BMI category: **{bmi_category}**")
    Age = st.slider("Age", 21, 81, 33)
 
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Predict")
 
# ---------------------- PREDICTION ----------------------
if predict_btn:
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    raw_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DiabetesPedigreeFunction, Age]]
    myinput = pd.DataFrame(data=raw_input, columns=columns)
    myinput_scaled = pd.DataFrame(data=sc.transform(myinput), columns=columns)
 
    result = model.predict(myinput_scaled)
 
    # Try to get a probability/confidence score if the model supports it
    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(myinput_scaled)[0]
        probability = proba[1] * 100  # probability of class 1 (diabetic)
 
    if result[0] == 0:
        risk_class = "low" if (probability is None or probability < 30) else "moderate"
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <div class="result-label">Prediction Result</div>
            <div class="result-value {risk_class}">Not Diabetic</div>
            <div class="result-sub">
                {"Estimated risk score: " + f"{probability:.1f}%" if probability is not None else "The model did not detect a diabetic pattern in the provided values."}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        risk_class = "high" if (probability is None or probability >= 60) else "moderate"
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <div class="result-label">Prediction Result</div>
            <div class="result-value {risk_class}">Diabetic</div>
            <div class="result-sub">
                {"Estimated risk score: " + f"{probability:.1f}%" if probability is not None else "The model detected a pattern consistent with diabetes in the provided values."}
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    # Input summary table
    st.markdown(f"""
    <table class="summary-table">
        <tr><td>Pregnancies</td><td>{Pregnancies}</td></tr>
        <tr><td>Glucose</td><td>{Glucose} mg/dL</td></tr>
        <tr><td>Blood Pressure</td><td>{BloodPressure} mm Hg</td></tr>
        <tr><td>Skin Thickness</td><td>{SkinThickness} mm</td></tr>
        <tr><td>Insulin</td><td>{Insulin} mu U/mL</td></tr>
        <tr><td>BMI</td><td>{BMI} ({bmi_category})</td></tr>
        <tr><td>Diabetes Pedigree Function</td><td>{DiabetesPedigreeFunction:.3f}</td></tr>
        <tr><td>Age</td><td>{Age}</td></tr>
    </table>
    """, unsafe_allow_html=True)
 
    st.markdown("""
    <div class="disclaimer">
        This tool provides a statistical estimate only and is not a substitute for
        professional medical advice, diagnosis, or treatment. Please consult a
        qualified healthcare provider for an accurate diagnosis.
    </div>
    """, unsafe_allow_html=True)
 
# ---------------------- FOOTER ----------------------
st.markdown('<div class="footer">Built with Streamlit — Designed for Women\'s Health</div>', unsafe_allow_html=True)
 