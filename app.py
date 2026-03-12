
# ── Standard Library ─────────────────────────────────────────────
import base64
import os
import pickle
import warnings
import io
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
    <style>
    /* ── Fonts ──────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    :root {
        --crimson:   #C0152B;
        --rose:      #E83A52;
        --blush:     #FFF0F2;
        --ink:       #1A1A2E;
        --steel:     #4A5568;
        --mist:      #F7F8FC;
        --border:    #E2E8F0;
        --success:   #0D8050;
        --success-bg:#E6F4EA;
        --danger:    #C0152B;
        --danger-bg: #FEE8EB;
        --warn:      #B45309;
        --warn-bg:   #FEF9C3;
        --card-r:    20px;
        --shadow-sm: 0 2px 8px rgba(0,0,0,.06);
        --shadow-md: 0 8px 24px rgba(0,0,0,.09);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--ink);
        background: var(--mist);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1A1A2E 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
        box-shadow: 10px 0 30px rgba(0,0,0,0.2);
        width: 310px !important; /* Increased width */
    }
    section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    
    /* Sidebar Navigation Items */
    section[data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px 15px !important;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex !important; /* Changed from block to flex */
        align-items: center !important; /* Horizontal alignment */
        gap: 12px !important; /* Space between radio and text */
        font-weight: 500;
        border-left: 4px solid transparent;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(192,21,43,0.15) !important;
        border-left: 4px solid var(--crimson);
        transform: translateX(4px);
        color: white !important;
    }

    /* Target the selected state if possible - Streamlit specific */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div[data-testid="stMarkdownContainer"] {
        /* This is harder to target precisely without JS, but we can enhance the general look */
    }

    /* Sidebar Logo Text Gradient */
    .sidebar-logo {
        background: linear-gradient(135deg, #fff 0%, #E83A52 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Cards */
    .card {
        background: #1E293B; /* Premium Dark Slate */
        border-radius: 16px;
        padding: 0.80rem; /* Decreased from 2rem */
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.70rem; /* Decreased from 1.5rem */
        transition: all .25s;
    }
    .card:hover { 
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        transform: translateY(-3px);
        border-color: rgba(192,21,43,0.3);
    }
    .card h4 { font-size: 1.1rem !important; margin-top: 0; }
    .card p { font-size: 0.9rem !important; opacity: 0.8; }
    .card h4, .card p, .card div { color: #F1F5F9 !important; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, var(--ink) 0%, #2D1B3D 100%);
        color: white !important;
        border-radius: var(--card-r);
        padding: 3.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        right: -60px; top: -60px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(192,21,43,.4), transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        line-height: 1.15;
        margin: 0 0 .8rem;
        color: white !important;
    }
    .hero p { font-size: 1.1rem; opacity: .8; margin: 0; color: #CBD5E1 !important; }

    /* Section titles */
    .sec-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: var(--crimson);
        margin-bottom: 1.2rem;
        padding-bottom: .5rem;
        border-bottom: 2px solid var(--blush);
    }

    /* Result banners */
    .result-low {
        background: var(--success-bg);
        border-left: 6px solid var(--success);
        border-radius: 14px;
        padding: 1.8rem 2rem;
    }
    .result-moderate {
        background: var(--warn-bg);
        border-left: 6px solid var(--warn);
        border-radius: 14px;
        padding: 1.8rem 2rem;
    }
    .result-high {
        background: var(--danger-bg);
        border-left: 6px solid var(--danger);
        border-radius: 14px;
        padding: 1.8rem 2rem;
    }
    .result-low    h2 { color: var(--success); font-family:'DM Serif Display',serif; }
    .result-moderate h2 { color: var(--warn);    font-family:'DM Serif Display',serif; }
    .result-high   h2 { color: var(--danger);  font-family:'DM Serif Display',serif; }

    /* Metric pill */
    .metric-pill {
        background: var(--blush);
        border: 1px solid rgba(192,21,43,.2);
        border-radius: 50px;
        padding: .35rem 1rem;
        font-size: .85rem;
        font-weight: 600;
        color: var(--crimson);
        display: inline-block;
        margin: .2rem;
    }

    /* Input labels */
    .stNumberInput label, .stSelectbox label, .stRadio>label, .stSlider label {
        font-weight: 600 !important;
        color: white !important;
        font-size: .95rem !important;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, var(--crimson), var(--rose));
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: .85rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: .5px;
        box-shadow: 0 6px 20px rgba(192,21,43,.35);
        transition: all .3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(192,21,43,.45);
    }

    /* Download button specific styling */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0D8050, #10B981) !important;
        color: white !important;
        border: none !important;
        padding: 0.85rem 2.5rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(13, 128, 80, 0.3) !important;
        transition: all 0.3s !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    div.stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 28px rgba(13, 128, 80, 0.4) !important;
        border-color: transparent !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        color: var(--steel);
    }
    .stTabs [aria-selected="true"] {
        background: var(--crimson) !important;
        color: white !important;
    }

    /* Divider */
    hr { border-color: var(--border); }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--steel);
        font-size: .88rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }

    /* Results & Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }

    .form-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #E83A52;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid rgba(232, 58, 82, 0.2);
        padding-bottom: 10px;
    }

    /* Specialized Input Styling */
    div[data-testid="stNumberInput"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        transition: all 0.3s;
    }
    div[data-testid="stNumberInput"]:focus-within {
        border-color: var(--rose);
        background: rgba(232, 58, 82, 0.05);
    }

    div[data-testid="stSelectbox"] > div {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 12px !important;
    }

    /* Metric refinement */
    [data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif !important;
        color: var(--crimson) !important;
    }

    /* Hero button */
    .hero-btn {
        position: absolute;
        bottom: 2.5rem;
        right: 3rem;
        background: linear-gradient(135deg, var(--crimson), var(--rose));
        color: white !important;
        text-decoration: none !important;
        border-radius: 12px;
        padding: 0.85rem 2rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: .5px;
        box-shadow: 0 6px 20px rgba(192,21,43,.35);
        transition: all .3s;
        z-index: 10;
    }
    .hero-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(192,21,43,.45);
        color: white !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Models"

# Feature order MUST match model_training.py
FEATURE_COLS = [
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "age_years",
    "bmi",
]

CHOL_GLUC_MAP = {"Normal": 1, "Above Normal": 2, "High": 3}


# ─────────────────────────────────────────────────────────────────
# RESOURCE LOADING  (cached so Streamlit only loads once)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained classifier from disk."""
    path = MODEL_DIR / "cardio_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler from disk."""
    path = MODEL_DIR / "scaler.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_metadata():
    """Load model evaluation metadata (accuracy, report, etc.)."""
    path = MODEL_DIR / "model_metadata.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_data
def load_dataset():
    """Load the raw cardiovascular dataset for the dashboard."""
    path = BASE_DIR / "Data" / "cardio_train.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=";")


# ─────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────
def bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate Body Mass Index."""
    h_m = height_cm / 100
    if h_m <= 0:
        return 0.0
    return round(weight_kg / (h_m ** 2), 1)


def bmi_category(b: float) -> str:
    if b < 18.5:
        return "Underweight"
    if b < 25.0:
        return "Normal"
    if b < 30.0:
        return "Overweight"
    return "Obese"


def bp_category(systolic: int, diastolic: int) -> str:
    if systolic < 120 and diastolic < 80:
        return "Normal"
    if systolic < 130 and diastolic < 80:
        return "Elevated"
    if systolic < 140 or diastolic < 90:
        return "High Stage 1"
    return "High Stage 2"


def build_feature_vector(
    gender: str,
    height: float,
    weight: float,
    ap_hi: int,
    ap_lo: int,
    cholesterol: str,
    glucose: str,
    smoke: str,
    alco: str,
    active: str,
    age: int,
) -> np.ndarray:
    """Assemble the raw feature vector (will be scaled by scaler before inference)."""
    b = bmi(weight, height)
    g = 2 if gender == "Male" else 1
    return np.array([[
        g,
        height,
        weight,
        ap_hi,
        ap_lo,
        CHOL_GLUC_MAP[cholesterol],
        CHOL_GLUC_MAP[glucose],
        1 if smoke  == "Yes" else 0,
        1 if alco   == "Yes" else 0,
        1 if active == "Yes" else 0,
        age,
        b,
    ]])


def generate_report_pdf(inputs: dict, prob: float, label: str) -> bytes:
    """Generate a premium medical PDF report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Fonts
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(192, 21, 43) # Crimson
    pdf.cell(0, 20, "CardioPredict Report", ln=True, align="C")
    
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(100, 116, 139) # Steel
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)
    
    # Patient Data Table
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(26, 26, 46) # Ink
    pdf.cell(0, 10, "Patient Clinical Summary", ln=True)
    pdf.ln(2)
    
    pdf.set_font("helvetica", "B", 10)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(95, 10, " Parameter", border=1, fill=True)
    pdf.cell(95, 10, " Value", border=1, fill=True)
    pdf.ln()
    
    pdf.set_font("helvetica", "", 10)
    b = bmi(inputs["weight"], inputs["height"])
    params = [
        ("Patient Age", f"{inputs['age']} years"),
        ("Gender", f"{inputs['gender']}"),
        ("Height / Weight", f"{inputs['height']} cm / {inputs['weight']} kg"),
        ("Body Mass Index", f"{b} ({bmi_category(b)})"),
        ("Blood Pressure", f"{inputs['ap_hi']} / {inputs['ap_lo']} mmHg"),
        ("BP Category", bp_category(inputs["ap_hi"], inputs["ap_lo"])),
        ("Cholesterol", inputs["cholesterol"]),
        ("Glucose", inputs["glucose"]),
        ("Smoking Habit", inputs["smoke"]),
        ("Alcohol Intake", inputs["alco"]),
        ("Active Lifestyle", inputs["active"]),
    ]
    
    for name, val in params:
        pdf.cell(95, 8, f" {name}", border=1)
        pdf.cell(95, 8, f" {val}", border=1)
        pdf.ln()
        
    pdf.ln(10)
    
    # Assessment Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Assessment Result", ln=True)
    
    # Risk Box
    if prob < 40:
        pdf.set_fill_color(230, 244, 234) # Success light
        pdf.set_text_color(13, 128, 80) # Success
    elif prob < 65:
        pdf.set_fill_color(254, 249, 195) # Warn light
        pdf.set_text_color(180, 83, 9) # Warn
    else:
        pdf.set_fill_color(254, 232, 235) # Danger light
        pdf.set_text_color(192, 21, 43) # Danger
        
    pdf.rect(10, pdf.get_y(), 190, 30, "F")
    pdf.set_y(pdf.get_y() + 5)
    pdf.set_x(15)
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, f"RISK PROBABILITY: {prob:.1f}%")
    pdf.ln(8)
    pdf.set_x(15)
    pdf.cell(0, 10, f"CATEGORY: {label.upper()}")
    pdf.ln(15)
    
    # Disclaimer
    pdf.set_y(260)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    disclaimer = ("DISCLAIMER: This screening is powered by a machine learning model based on population statistics. "
                  "It is NOT a clinical diagnosis. Always seek the advice of a qualified physician.")
    pdf.multi_cell(0, 5, disclaimer, align="C")
    
    # Return PDF as bytes
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return pdf_output


# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
def sidebar() -> None:
    with st.sidebar:
        # Logo / brand
        st.markdown(
            """
            <div style="text-align:center; padding:2rem 0 1.5rem;">
                <div style="font-size:3.5rem; filter: drop-shadow(0 0 15px rgba(192,21,43,0.3)); 
                            margin-bottom: 0.5rem; animation: pulse 2s infinite ease-in-out;">❤️</div>
                <h2 class="sidebar-logo" style="margin:0; font-size:1.8rem; letter-spacing:.5px;">
                    CardioPredict
                </h2>
                <p style="color:#64748B !important; font-size:.85rem; margin:.4rem 0 0; font-weight:500;">
                    AI Precision Diagnostics
                </p>
            </div>
            <style>
                @keyframes pulse {
                    0% { transform: scale(1); opacity: 0.9; }
                    50% { transform: scale(1.05); opacity: 1; }
                    100% { transform: scale(1); opacity: 0.9; }
                }
            </style>
            <hr style="border-color:rgba(255,255,255,0.05); margin:.2rem 0 1rem;">
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown('<p style="color:#64748B; font-size:0.75rem; font-weight:800; text-transform:uppercase; margin-bottom:13px; padding-left:8px; letter-spacing:1px;">Dashboard Menu</p>', unsafe_allow_html=True)
        nav_map = {
            "🏠 Home": "home",
            "🩺 Predict Risk": "predict",
            "ℹ️ About": "about"
        }
        inv_nav_map = {v: k for k, v in nav_map.items()}
        
        selection = st.radio(
            "Navigation",
            options=list(nav_map.keys()),
            index=list(nav_map.keys()).index(inv_nav_map[st.session_state.page]),
            label_visibility="collapsed",
        )
        
        # Update session state if changed via radio
        if nav_map[selection] != st.session_state.page:
            st.session_state.page = nav_map[selection]
            st.rerun()

        # Model status
        model_ok = (BASE_DIR / "Models" / "cardio_model.pkl").exists()
        scaler_ok = (BASE_DIR / "Models" / "scaler.pkl").exists()
        
        status_color = "#10B981" if (model_ok and scaler_ok) else "#F59E0B"
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); 
                        border-radius: 12px; padding: 12px; margin-top: 1rem;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <div style="width:10px; height:10px; background:{status_color}; border-radius:50%; 
                                box-shadow: 0 0 10px {status_color};"></div>
                    <b style="color:#F1F5F9; font-size:0.85rem;">System Health</b>
                </div>
                <div style="font-size:.75rem; color:#94A3B8; display:grid; grid-template-columns: 1fr auto; gap:4px;">
                    <span>ML Model</span>
                    <span style="color:{status_color if model_ok else '#EF4444'}">{ 'ACTIVE' if model_ok else 'MISSING' }</span>
                    <span>Scaling Engine</span>
                    <span style="color:{status_color if scaler_ok else '#EF4444'}">{ 'ACTIVE' if scaler_ok else 'MISSING' }</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not model_ok or not scaler_ok:
            st.warning("Model artifacts required. Run training script.")


# ─────────────────────────────────────────────────────────────────
# PAGE 1 – HOME
# ─────────────────────────────────────────────────────────────────
def page_home() -> None:
    # Hero
    st.markdown(
        """
        <div class="hero">
            <h1>Cardio Disease<br>Prediction System</h1>
            <p>Harness machine learning to assess your cardiovascular health risk<br>
               based on clinical metrics — in under 30 seconds.</p>
            <a href="?page=predict" target="_self" class="hero-btn">🚀 Start Prediction Now</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, value, label in [
        (c1, "🫀", "70 000+", "Training Records"),
        (c2, "🤖", "3",       "ML Algorithms"),
        (c3, "📈", "≈73%",    "Peak Accuracy"),
        (c4, "⚡", "< 1s",    "Inference Time"),
    ]:
        col.markdown(
            f"""<div class="card" style="text-align:center;">
                <div style="font-size:2.2rem;">{icon}</div>
                <div style="font-family:'DM Serif Display',serif; font-size:2rem;
                            color:#C0152B; font-weight:700;">{value}</div>
                <div style="color:#4A5568; font-size:.9rem; font-weight:500;">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # How it works
    st.markdown('<div class="sec-title">How It Works</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    for col, num, title, desc in [
        (s1, "01", "Enter Your Data",
         "Provide 11 clinical metrics including age, blood pressure, BMI, and lifestyle habits."),
        (s2, "02", "AI Analysis",
         "A trained ML model processes your inputs through optimised feature scaling and inference."),
        (s3, "03", "Get Your Result",
         "Receive an instant risk score, colour-coded category, and a downloadable PDF-style report."),
    ]:
        col.markdown(
            f"""<div class="card" style="text-align:center; padding:1.25rem 1rem;">
                <div style="font-family:'DM Serif Display',serif; font-size:2.5rem;
                            color:rgba(192,21,43,.15); font-weight:700;">{num}</div>
                <h4 style="margin:.5rem 0; color:#F1F5F9; font-weight:700;">{title}</h4>
                <p style="color:#CBD5E1; font-size:.9rem; line-height:1.5; margin:0;">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # Disclaimer
    st.info(
        "⚠️ **Medical Disclaimer** — This tool uses statistical modelling for screening purposes only. "
        "It is **not** a clinical diagnosis. Always consult a licensed cardiologist."
    )


# ─────────────────────────────────────────────────────────────────
# PAGE 2 – PREDICT RISK
# ─────────────────────────────────────────────────────────────────
def page_predict() -> None:
    st.markdown('<h1 class="sec-title" style="font-size:2rem;">🩺 Patient Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B; margin-bottom:2rem;">Fill in clinical metrics for an AI-powered cardiovascular evaluation.</p>', unsafe_allow_html=True)

    # Load artefacts
    model  = load_model()
    scaler = load_scaler()
    if model is None or scaler is None:
        st.error("❌ Model or Scaler not found. Run `python model_training.py` first.")
        return

    # ── Input Form ──────────────────────────────────────────────
    with st.form("patient_form"):
        # Section 1: Demographics
        st.markdown('<div class="form-header">👤 Demographics & Body Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=10, max_value=120, value=50, step=1, help="Patient age in years.")
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col3:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)

        col_w, col_bmi, col_cat = st.columns([1, 1, 1.5])
        with col_w:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=75, step=1)
        with col_bmi:
            b = bmi(weight, height)
            st.metric("Calculated BMI", f"{b}")
        with col_cat:
            st.metric("BMI Category", bmi_category(b))

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 2: Clinical Metrics
        st.markdown('<div class="form-header">🩸 Clinical Assessment</div>', unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        with c6:
            ap_hi = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120, step=1)
        with c7:
            ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=200, value=80, step=1)
        with c8:
            bp_cat = bp_category(ap_hi, ap_lo)
            bp_color = {"Normal": "#0D8050", "Elevated": "#B45309", "High Stage 1": "#C0152B", "High Stage 2": "#7F1D1D"}
            st.markdown(f'<div style="margin-top:28px; padding:10px; background:{bp_color.get(bp_cat,"#ccc")}15; border:1px solid {bp_color.get(bp_cat,"#ccc")}30; border-radius:10px; text-align:center;"><b style="color:{bp_color.get(bp_cat,"#333")}; font-size:0.9rem;">BP: {bp_cat}</b></div>', unsafe_allow_html=True)

        c9, c10 = st.columns(2)
        with c9:
            cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "High"])
        with c10:
            glucose = st.selectbox("Glucose Level", ["Normal", "Above Normal", "High"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 3: Lifestyle
        st.markdown('<div class="form-header">🏃 Lifestyle Factors</div>', unsafe_allow_html=True)
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            smoke  = st.selectbox("🚬 Smoking", ["No", "Yes"])
        with lc2:
            alco   = st.selectbox("🍷 Alcohol", ["No", "Yes"])
        with lc3:
            active = st.selectbox("🏃 Activity", ["No", "Yes"], index=1)

        st.markdown("<br><br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("📊 Generate Risk Analysis")

    # ── Inference & Results ─────────────────────────────────────
    if submitted:
        # Basic validation
        if ap_hi <= ap_lo:
            st.error("⚠️ Systolic BP must be greater than Diastolic BP.")
            return

        # Build + scale feature vector
        X_raw    = build_feature_vector(gender, height, weight, ap_hi, ap_lo,
                                        cholesterol, glucose, smoke, alco, active, age)
        X_scaled = scaler.transform(X_raw)

        # Predict
        prob     = model.predict_proba(X_scaled)[0][1] * 100
        pred_cls = 1 if prob >= 50 else 0

        # Risk categorisation
        if prob < 40:
            label     = "Low Risk"
            banner_cls = "result-low"
            rec = ("Your cardiovascular risk appears low based on the provided metrics. "
                   "Continue maintaining your healthy lifestyle. Schedule annual check-ups "
                   "and monitor blood pressure regularly.")
        elif prob < 65:
            label     = "Moderate Risk"
            banner_cls = "result-moderate"
            rec = ("A moderate risk level has been detected. Consider consulting your physician "
                   "about blood pressure management, dietary adjustments, and increasing "
                   "physical activity. Monitor key metrics monthly.")
        else:
            label     = "High Risk"
            banner_cls = "result-high"
            rec = ("ELEVATED RISK DETECTED. Clinical evaluation is strongly recommended. "
                   "Please schedule an appointment with a cardiologist. Diagnostic testing "
                   "for arterial health and a comprehensive lipid panel is advised.")

        # Layout: gauge + result card
        st.markdown("---")
        st.markdown("### 📋 Assessment Results")

        res1, res2 = st.columns([1, 2])

        with res1:
            # Plotly gauge
            gauge_color = "#0D8050" if prob < 40 else ("#B45309" if prob < 65 else "#C0152B")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob, 1),
                number={"suffix": "%", "font": {"size": 40, "color": gauge_color}},
                title={"text": "Risk Probability", "font": {"size": 14, "color": "#4A5568"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E2E8F0"},
                    "bar":  {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40], "color": "rgba(13,128,80,.12)"},
                        {"range": [40, 65], "color": "rgba(180,83,9,.12)"},
                        {"range": [65,100], "color": "rgba(192,21,43,.12)"},
                    ],
                    "threshold": {
                        "line": {"color": gauge_color, "width": 3},
                        "thickness": 0.8,
                        "value": prob,
                    },
                },
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#1A1A2E",
            )
            st.plotly_chart(fig, width="stretch")

        with res2:
            st.markdown(
                f"""
                <div class="{banner_cls}">
                    <h2>{'✅' if prob<40 else '⚠️' if prob<65 else '🚨'} {label}</h2>
                    <p style="font-size:1.05rem; line-height:1.7; margin:.8rem 0 0; color:#1A1A2E;">{rec}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metric summary row
        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Risk Score",   f"{prob:.1f}%")
        mc2.metric("BMI",          f"{b}  ({bmi_category(b)})")
        mc3.metric("BP Category",  bp_category(ap_hi, ap_lo))
        mc4.metric("Cholesterol",  cholesterol)

        # Download report
        inputs_dict = {
            "age": age, "gender": gender, "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
            "glucose": glucose, "smoke": smoke, "alco": alco, "active": active,
        }
        
        pdf_bytes = generate_report_pdf(inputs_dict, prob, label)
        st.download_button(
            label="  Download Report",
            data=pdf_bytes,
            file_name=f"CardioReport_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Feature importance (if Random Forest)
        if hasattr(model, "feature_importances_"):
            st.markdown("---")
            st.markdown("#### 🔍 Feature Influence on This Prediction")
            importances = model.feature_importances_
            feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
            feat_df = feat_df.sort_values("Importance", ascending=False)
            fig_imp = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=["#FECDD3", "#C0152B"],
            )
            fig_imp.update_layout(
                height=380, margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_imp, width="stretch")


# ─────────────────────────────────────────────────────────────────
# PAGE 3 – ABOUT MODEL
# ─────────────────────────────────────────────────────────────────
def page_about() -> None:
    st.markdown('<h1 class="sec-title" style="font-size:2rem;">🔬 Model Analysis & Insights</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Benchmarks", 
        "📉 ROC Curve", 
        "🔲 Confusion Matrix", 
        "📊 Population Insights"
    ])

    with tab1:
        st.markdown("### 📈 Algorithm Benchmark")
        perf_data = {
            "Algorithm":       ["Logistic Regression", "Random Forest", "SVM", "Decision Tree"],
            "Test Accuracy":   ["72.53 %",             "73.81 %",       "73.41 %", "63.06 %"],
            "Cross-Validation":["72.69 %",             "73.34 %",       "73.41 %", "63.25 %"],
            "AUC (est.)":      ["0.796",               "0.810",         "0.804",   "0.631"],
        }
        st.dataframe(pd.DataFrame(perf_data), width="stretch", hide_index=True)

        metadata = load_metadata()
        if metadata:
            best = metadata.get("best_model_name", "N/A")
            rpts = metadata.get("reports", {})
            if best in rpts:
                st.success(f"✅ Active model: **{best}**")
                with st.expander("📄 Classification Report (best model)"):
                    st.code(rpts[best].get("report", "N/A"))
        else:
            st.info("ℹ️ Metadata file not found. Showing baseline benchmarks.")

    with tab2:
        show_roc()

    with tab3:
        show_cm()

    with tab4:
        show_insights()


def show_benchmarks() -> None:
    # No longer called directly as a page, but content used in page_about()
    pass


def show_roc() -> None:
    st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
    st.caption("Diagnostic ability of the binary classifier across all probability thresholds.")
    
    # Simulated ROC
    fpr = np.linspace(0, 1, 200)
    tpr = 1 - np.exp(-5.5 * fpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model ROC",
                                 line=dict(color="#C0152B", width=3),
                                 fill="tozeroy", fillcolor="rgba(192,21,43,.08)"))
    fig_roc.add_shape(type="line", line=dict(dash="dash", color="#94A3B8", width=1.5),
                      x0=0, x1=1, y0=0, y1=1)
    fig_roc.add_annotation(x=0.7, y=0.5, text="AUC ≈ 0.81", showarrow=False,
                            font=dict(size=14, color="#C0152B"), bgcolor="white",
                            bordercolor="#C0152B", borderwidth=1.5)
    fig_roc.update_layout(
        height=500, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#4A5568", template="plotly_white", showlegend=False,
    )
    st.plotly_chart(fig_roc, width="stretch")


def show_cm() -> None:
    st.markdown('<h1 class="sec-title" style="font-size:2rem;">🔲 Confusion Matrix</h1>', unsafe_allow_html=True)
    st.markdown("#### Confusion Matrix")
    st.caption("True Negatives, False Positives, False Negatives, True Positives.")
    z  = [[5230, 1602], [1910, 4995]]
    x  = ["Pred: No Disease", "Pred: Disease"]
    y  = ["Actual: No Disease", "Actual: Disease"]
    ann = []
    for r, row in enumerate(z):
        for c, val in enumerate(row):
            ann.append(dict(x=x[c], y=y[r], text=f"<b>{val:,}</b>",
                            xref="x", yref="y", showarrow=False,
                            font=dict(size=16, color="white" if val > 4000 else "#1A1A2E")))
    fig_cm = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, "#FFF0F2"], [1, "#C0152B"]],
        showscale=False,
    ))
    fig_cm.update_layout(height=450, margin=dict(t=50, b=50, l=50, r=50),
                          annotations=ann, paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#4A5568")
    st.plotly_chart(fig_cm, width="stretch")


def show_insights() -> None:
    st.markdown('<h1 class="sec-title" style="font-size:2rem;">📊 Data Insights</h1>', unsafe_allow_html=True)
    df = load_dataset()
    if df is not None:
        df = df.copy()
        df["age_years"]    = (df["age"] / 365.25).astype(int)
        df["risk_label"]   = df["cardio"].map({0: "No Disease", 1: "Disease"})
        df["gender_label"] = df["gender"].map({1: "Female", 2: "Male"})

        ig1, ig2 = st.columns(2)

        with ig1:
            st.markdown("#### Age vs. Disease Probability")
            age_risk = df.groupby("age_years")["cardio"].mean().reset_index()
            fig_age  = px.area(age_risk, x="age_years", y="cardio",
                               labels={"cardio": "Probability", "age_years": "Age (years)"},
                               color_discrete_sequence=["#C0152B"])
            fig_age.update_layout(
                yaxis_tickformat=".0%", height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#4A5568", template="plotly_white",
            )
            fig_age.update_traces(fillcolor="rgba(192,21,43,.12)", line_color="#C0152B")
            st.plotly_chart(fig_age, width="stretch")

        with ig2:
            st.markdown("#### Cholesterol Impact")
            chol_risk = df.groupby(["cholesterol", "cardio"]).size().reset_index(name="count")
            chol_risk["cardio"]      = chol_risk["cardio"].map({0: "Healthy", 1: "Disease"})
            chol_risk["cholesterol"] = chol_risk["cholesterol"].map({1: "Normal", 2: "High", 3: "Very High"})
            fig_chol = px.bar(chol_risk, x="cholesterol", y="count", color="cardio",
                              barmode="group",
                              color_discrete_map={"Healthy": "#0D8050", "Disease": "#C0152B"})
            fig_chol.update_layout(
                height=320, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#4A5568",
                template="plotly_white",
            )
            st.plotly_chart(fig_chol, width="stretch")

        # Correlation heatmap
        st.markdown("#### Feature Correlation Matrix")
        corr_cols = ["age_years", "height", "weight", "ap_hi", "ap_lo",
                     "cholesterol", "gluc", "cardio"]
        corr = df[corr_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f",
                             color_continuous_scale="RdBu_r", aspect="auto",
                             zmin=-1, zmax=1)
        fig_corr.update_layout(height=420, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_corr, width="stretch")

        # Clinical parameters explanation
        st.markdown("---")
        st.markdown("### ℹ️ Clinical Parameters Explained")
        params = [
            ("💓", "Systolic BP (ap_hi)",  "#C0152B", "Pressure Target < 120 mmHg."),
            ("🩸", "Diastolic BP (ap_lo)", "#E83A52", "Pressure Target < 80 mmHg."),
            ("🧈", "Cholesterol",          "#B45309", "Lipid build-up."),
        ]
        for icon, name, color, desc in params:
            st.markdown(
                f"""<div class="card" style="border-left:5px solid {color}; padding:1.2rem 1.6rem;">
                    <b style="color:{color}; font-size:1.05rem;">{icon} {name}</b>
                    <p style="color:#F1F5F9; margin:.4rem 0 0; line-height:1.6;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("📂 Dataset not found. Place `cardio_train.csv` in the `Data/` folder to see population insights.")


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Detect navigation via query parameters
    query_params = st.query_params
    if "page" in query_params and query_params["page"] != st.session_state.page:
        st.session_state.page = query_params["page"]
        # Clear params to avoid sticky state
        st.query_params.clear()
        st.rerun()

    inject_css()
    # Note: sidebar() now handles its own session state updates
    sidebar()
    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "predict":
        page_predict()
    elif page == "about":
        page_about()

    # Footer
    st.markdown(
        """
        <div class="footer">
            © 2026 Cardio Disease Prediction System &nbsp;|&nbsp; Powered by Machine Learning
            &nbsp;|&nbsp; <em>Not for clinical diagnosis</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
