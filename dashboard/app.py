import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import AQIPredictor
from src.alerts import generate_alert

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AirVista  |  Air Quality Intelligence",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>A</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN SYSTEM - Premium Glassmorphic Dark Theme
# ============================================================================
st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Root Variables ---- */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: rgba(255, 255, 255, 0.03);
    --bg-card-hover: rgba(255, 255, 255, 0.06);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-glow: rgba(99, 102, 241, 0.3);
    --text-primary: #f0f0f5;
    --text-secondary: #8b8b9e;
    --text-muted: #5a5a6e;
    --accent-indigo: #6366f1;
    --accent-cyan: #22d3ee;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-violet: #8b5cf6;
    --gradient-primary: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    --gradient-danger: linear-gradient(135deg, #f43f5e, #e11d48);
    --gradient-success: linear-gradient(135deg, #10b981, #059669);
    --gradient-warning: linear-gradient(135deg, #f59e0b, #d97706);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.15);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
}

/* ---- Global Reset ---- */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp > header { background: transparent !important; }

/* ---- Animations ---- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes pulse-ring {
    0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
    70% { box-shadow: 0 0 0 12px rgba(99, 102, 241, 0); }
    100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(99, 102, 241, 0.2); }
    50% { border-color: rgba(99, 102, 241, 0.5); }
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0c14 0%, #08080e 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* Nav radio group label (hidden) */
section[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}

/* Nav items container */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}

/* Individual nav items */
section[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    padding: 11px 18px !important;
    border-radius: 10px !important;
    margin: 0 8px 0 8px !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
    color: #6b6b80 !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
}

/* Nav item hover */
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99, 102, 241, 0.06) !important;
    color: #c7c7d6 !important;
    transform: translateX(3px) !important;
    border-color: rgba(99, 102, 241, 0.08) !important;
}

/* Nav item active/selected */
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(99, 102, 241, 0.1) !important;
    color: #e0e0f0 !important;
    font-weight: 600 !important;
    border-color: rgba(99, 102, 241, 0.15) !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.08), inset 0 0 20px rgba(99, 102, 241, 0.03) !important;
}

/* Active indicator bar on the left */
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"]::before,
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked)::before {
    content: '' !important;
    position: absolute !important;
    left: 0 !important;
    top: 20% !important;
    bottom: 20% !important;
    width: 3px !important;
    background: linear-gradient(180deg, #6366f1, #8b5cf6) !important;
    border-radius: 0 4px 4px 0 !important;
    animation: fadeIn 0.3s ease-out !important;
}

/* Hide radio circle */
section[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}

/* Selectbox styling */
section[data-testid="stSidebar"] .stSelectbox label {
    color: #4a4a5e !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px !important;
    color: #c7c7d6 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.08) !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.12) !important;
}

/* ---- Glass Card ---- */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 28px;
    margin-bottom: 16px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeInUp 0.6s ease-out both;
}
.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}

/* ---- Metric Card ---- */
.av-metric {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px 20px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeInUp 0.5s ease-out both;
    position: relative;
    overflow: hidden;
}
.av-metric::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-primary);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    opacity: 0;
    transition: opacity 0.3s ease;
}
.av-metric:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    transform: translateY(-4px);
}
.av-metric:hover::before { opacity: 1; }

.av-metric-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}
.av-metric-value {
    font-size: 38px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 8px;
    background: linear-gradient(135deg, var(--val-color, #f0f0f5), var(--val-color-end, #f0f0f5));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.av-metric-sub {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 0.5px;
}
.av-metric-caption {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 10px;
    line-height: 1.5;
    font-weight: 400;
}

/* ---- Section Headers ---- */
.av-section-title {
    font-size: 32px;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 8px;
    letter-spacing: -0.5px;
    animation: fadeIn 0.5s ease-out both;
}
.av-section-sub {
    font-size: 15px;
    color: var(--text-secondary);
    margin-bottom: 32px;
    line-height: 1.6;
    animation: fadeIn 0.6s ease-out both;
}

/* ---- Hero Banner ---- */
.av-hero {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08), rgba(34, 211, 238, 0.06));
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: var(--radius-xl);
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s ease-out both;
}
.av-hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
    animation: gradientShift 8s ease-in-out infinite;
    background-size: 200% 200%;
}
.av-hero-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.15;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #f0f0f5 0%, #a78bfa 50%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: shimmer 4s linear infinite;
}
.av-hero-desc {
    font-size: 16px;
    color: var(--text-secondary);
    line-height: 1.7;
    max-width: 600px;
}

/* ---- Alert Banners ---- */
.av-alert {
    border-radius: var(--radius-md);
    padding: 20px 24px;
    margin-bottom: 16px;
    border-left: 4px solid;
    animation: slideInLeft 0.4s ease-out both;
    backdrop-filter: blur(10px);
}
.av-alert-danger {
    background: rgba(244, 63, 94, 0.08);
    border-left-color: var(--accent-rose);
    color: #fda4af;
}
.av-alert-warning {
    background: rgba(245, 158, 11, 0.08);
    border-left-color: var(--accent-amber);
    color: #fcd34d;
}
.av-alert-success {
    background: rgba(16, 185, 129, 0.08);
    border-left-color: var(--accent-emerald);
    color: #6ee7b7;
}
.av-alert-info {
    background: rgba(99, 102, 241, 0.08);
    border-left-color: var(--accent-indigo);
    color: #a5b4fc;
}
.av-alert-title {
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.av-alert-body {
    font-size: 14px;
    line-height: 1.6;
    opacity: 0.9;
}

/* ---- AQI Scale Table ---- */
.av-scale-row {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    border-radius: var(--radius-sm);
    margin-bottom: 4px;
    background: var(--bg-card);
    transition: all 0.3s ease;
}
.av-scale-row:hover { background: var(--bg-card-hover); }
.av-scale-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 14px;
    flex-shrink: 0;
}
.av-scale-range {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text-primary);
    width: 90px;
    font-weight: 500;
}
.av-scale-cat {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    width: 110px;
}
.av-scale-desc {
    font-size: 12px;
    color: var(--text-muted);
    flex: 1;
}

/* ---- Buttons ---- */
.stButton > button {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ---- Download Button ---- */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.3s ease !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent-indigo) !important;
    color: var(--accent-indigo) !important;
    background: rgba(99, 102, 241, 0.06) !important;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ---- Divider ---- */
.av-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
    margin: 32px 0;
}

/* ---- About Grid ---- */
.av-about-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 28px;
    height: 100%;
    transition: all 0.4s ease;
    animation: fadeInUp 0.5s ease-out both;
}
.av-about-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}
.av-about-icon {
    font-size: 28px;
    margin-bottom: 16px;
    display: block;
}
.av-about-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}
.av-about-text {
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.7;
}

/* ---- Hide Streamlit defaults ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ---- Smooth scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & MODELS
# ============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/featured_air_quality.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Data load error: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_predictor():
    try:
        return AQIPredictor(data_path='data/processed/featured_air_quality.csv', models_dir='models')
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

df = load_data()
predictor = get_predictor()

# ============================================================================
# HELPERS
# ============================================================================
def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def aqi_color(aqi):
    if aqi <= 50: return "#10b981", "#059669"
    elif aqi <= 100: return "#22d3ee", "#06b6d4"
    elif aqi <= 200: return "#f59e0b", "#d97706"
    elif aqi <= 300: return "#f43f5e", "#e11d48"
    elif aqi <= 400: return "#8b5cf6", "#7c3aed"
    else: return "#7E0023", "#5c0019"

def health_advice(cat):
    advice = {
        "Good": "Ideal air quality for all outdoor activities.",
        "Satisfactory": "Acceptable air quality. Sensitive individuals should be cautious with prolonged exertion.",
        "Moderate": "People with respiratory or heart conditions should limit prolonged outdoor exertion.",
        "Poor": "Everyone should reduce prolonged outdoor exertion and avoid heavy physical activity.",
        "Very Poor": "Avoid all outdoor physical activity. Keep windows closed and stay indoors.",
        "Severe": "Health emergency conditions. Stay indoors and use air purifiers if available."
    }
    return advice.get(cat, "")

def render_metric(label, value, color1, color2, subtitle="", caption="", delay=0):
    st.markdown(f"""
    <div class="av-metric" style="animation-delay: {delay}s;">
        <div class="av-metric-label">{label}</div>
        <div class="av-metric-value" style="--val-color: {color1}; --val-color-end: {color2};">{value}</div>
        {'<div class="av-metric-sub">' + subtitle + '</div>' if subtitle else ''}
        {'<div class="av-metric-caption">' + caption + '</div>' if caption else ''}
    </div>
    """, unsafe_allow_html=True)

def render_alert(alert_type, title, body):
    st.markdown(f"""
    <div class="av-alert av-alert-{alert_type}">
        <div class="av-alert-title">{title}</div>
        <div class="av-alert-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)

def plotly_theme(fig, height=400):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#8b8b9e"),
        title_font=dict(size=18, color="#f0f0f5", family="Inter, sans-serif"),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("""
<div style="padding: 24px 16px 16px 16px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="
            width: 36px; height: 36px; 
            background: linear-gradient(135deg, #6366f1, #8b5cf6); 
            border-radius: 10px; 
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; font-weight: 900; color: white;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        ">AV</div>
        <div>
            <div style="font-size: 18px; font-weight: 800; color: #f0f0f5; letter-spacing: -0.3px; line-height: 1.2;">AirVista</div>
            <div style="font-size: 10px; color: #4a4a5e; letter-spacing: 1.5px; text-transform: uppercase;">v2.0  /  Intelligence</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)

# Navigation Section Label
st.sidebar.markdown("""
<div style="padding: 4px 24px 8px 24px; font-size: 10px; font-weight: 700; color: #3a3a4e; text-transform: uppercase; letter-spacing: 2px;">
    Navigation
</div>
""", unsafe_allow_html=True)

pages = ["Home", "AQI Forecast", "Pollution Analytics", "Trend Analysis", "Risk Analysis", "Alerts", "Model Performance", "About"]
selected_page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

st.sidebar.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)

# Filters Section Label
st.sidebar.markdown("""
<div style="padding: 4px 24px 8px 24px; font-size: 10px; font-weight: 700; color: #3a3a4e; text-transform: uppercase; letter-spacing: 2px;">
    Filters
</div>
""", unsafe_allow_html=True)

# Global Filters
if not df.empty:
    cities = sorted(df['city'].unique())
    selected_city = st.sidebar.selectbox("City", cities)
    city_df = df[df['city'] == selected_city].sort_values('date')
    dates = city_df['date'].dt.date.unique()
    default_idx = len(dates) - 1 if len(dates) > 0 else 0
    selected_date = st.sidebar.selectbox("Date", dates, index=default_idx)
else:
    selected_city = None
    selected_date = None

st.sidebar.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)

# Engine Info
st.sidebar.markdown("""
<div style="margin: 0 12px; padding: 14px 16px; background: rgba(99, 102, 241, 0.04); border: 1px solid rgba(99, 102, 241, 0.08); border-radius: 10px;">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div style="font-size: 10px; color: #3a3a4e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">Engine</div>
            <div style="font-size: 13px; color: #a5b4fc; font-weight: 600; margin-top: 3px;">LightGBM</div>
        </div>
        <div style="
            padding: 3px 8px; 
            background: rgba(99, 102, 241, 0.12); 
            border-radius: 6px; 
            font-size: 10px; 
            color: #8b8bfa; 
            font-weight: 600;
            letter-spacing: 0.5px;
        ">v4.x</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Status indicator
st.sidebar.markdown("""
<div style="margin: 12px 12px 0 12px; padding: 10px 16px; display: flex; align-items: center; gap: 8px;">
    <div style="width: 6px; height: 6px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); animation: pulse-ring 2s ease-out infinite;"></div>
    <div style="font-size: 11px; color: #4a4a5e; font-weight: 500;">System Online</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGES
# ============================================================================

# ---------- HOME ----------
if selected_page == "Home":
    st.markdown("""
    <div class="av-hero">
        <div class="av-hero-title">Air Quality<br>Intelligence System</div>
        <div class="av-hero-desc">
            Real-time AQI forecasting powered by gradient-boosted decision trees. 
            Monitor air quality across major Indian cities with 24, 48, and 72-hour predictive horizons.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not df.empty and selected_city and selected_date:
        row = city_df[city_df['date'].dt.date == selected_date]
        if not row.empty:
            row = row.iloc[0]
            st.markdown(f"""
            <div class="av-section-title">City Snapshot</div>
            <div class="av-section-sub">{selected_city}  /  {selected_date}</div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            pollutant_data = [
                ("PM 2.5", row.get('pm2_5', 0), "#f43f5e", "#e11d48"),
                ("PM 10", row.get('pm10', 0), "#f59e0b", "#d97706"),
                ("NO2", row.get('no2', 0), "#8b5cf6", "#7c3aed"),
                ("CO", row.get('co', 0), "#22d3ee", "#06b6d4")
            ]
            for col, (name, val, c1_color, c2_color) in zip([c1, c2, c3, c4], pollutant_data):
                with col:
                    render_metric(name, f"{val:.1f}", c1_color, c2_color, delay=0.1)
            
            # Quick AQI Overview
            if 'aqi_24' in row and not pd.isna(row['aqi_24']):
                aqi_val = row['aqi_24']
                cat = aqi_category(aqi_val)
                c1_c, c2_c = aqi_color(aqi_val)
                st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    render_metric("Current AQI", f"{aqi_val:.0f}", c1_c, c2_c, subtitle=cat, caption=health_advice(cat))

# ---------- AQI FORECAST ----------
elif selected_page == "AQI Forecast":
    st.markdown(f"""
    <div class="av-section-title">AQI Forecast</div>
    <div class="av-section-sub">{selected_city}  /  Base Date: {selected_date}</div>
    """, unsafe_allow_html=True)
    
    # AQI Reference Scale
    with st.expander("AQI Reference Scale"):
        scale_data = [
            ("#10b981", "0 - 50", "Good", "Minimal health impact"),
            ("#22d3ee", "51 - 100", "Satisfactory", "Minor breathing discomfort to sensitive people"),
            ("#f59e0b", "101 - 200", "Moderate", "Breathing discomfort to people with lung/heart disease"),
            ("#f43f5e", "201 - 300", "Poor", "Breathing discomfort to most people on prolonged exposure"),
            ("#8b5cf6", "301 - 400", "Very Poor", "Respiratory illness on prolonged exposure"),
            ("#7E0023", "401 - 500+", "Severe", "Affects healthy people and seriously impacts existing conditions")
        ]
        for color, rng, cat, desc in scale_data:
            st.markdown(f"""
            <div class="av-scale-row">
                <div class="av-scale-dot" style="background: {color};"></div>
                <div class="av-scale-range">{rng}</div>
                <div class="av-scale-cat" style="color: {color};">{cat}</div>
                <div class="av-scale-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("Generate Forecast"):
        if predictor and selected_city and selected_date:
            with st.spinner("Running inference..."):
                res = predictor.predict(selected_city, str(selected_date), save_csv=False)
            
            if "error" in res:
                render_alert("danger", "Forecast Unavailable", res["error"])
            else:
                # Forecast Cards
                horizons = [
                    ("Next 24 Hours", res.get('Predicted_AQI_24h')),
                    ("Next 48 Hours", res.get('Predicted_AQI_48h')),
                    ("Next 72 Hours", res.get('Predicted_AQI_72h'))
                ]
                
                c1, c2, c3 = st.columns(3)
                for i, (col, (label, aqi_val)) in enumerate(zip([c1, c2, c3], horizons)):
                    if aqi_val is not None:
                        cat = aqi_category(aqi_val)
                        col1, col2 = aqi_color(aqi_val)
                        with col:
                            render_metric(label, f"{aqi_val:.1f}", col1, col2, subtitle=cat, caption=health_advice(cat), delay=i*0.15)
                
                st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)
                
                # Risk Gauge
                st.markdown('<div class="av-section-title" style="font-size: 22px;">Risk Assessment</div>', unsafe_allow_html=True)
                risk_val = res.get('Risk_Score_24h', 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_val,
                    number={'suffix': ' / 10', 'font': {'size': 36, 'family': 'JetBrains Mono'}},
                    title={'text': f"Risk Score  |  {res.get('AQI_Category_24h', 'N/A')}", 'font': {'size': 14, 'color': '#8b8b9e'}},
                    gauge={
                        'axis': {'range': [0, 10], 'tickcolor': '#5a5a6e', 'tickwidth': 1},
                        'bar': {'color': "#6366f1", 'thickness': 0.3},
                        'bgcolor': "rgba(255,255,255,0.02)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 2], 'color': "rgba(16,185,129,0.15)"},
                            {'range': [2, 4], 'color': "rgba(34,211,238,0.12)"},
                            {'range': [4, 6], 'color': "rgba(245,158,11,0.12)"},
                            {'range': [6, 8], 'color': "rgba(244,63,94,0.12)"},
                            {'range': [8, 10], 'color': "rgba(126,0,35,0.15)"}
                        ],
                        'threshold': {'line': {'color': "#f0f0f5", 'width': 3}, 'thickness': 0.8, 'value': risk_val}
                    }
                ))
                fig = plotly_theme(fig, height=320)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Alert
                aqi_24 = res.get('Predicted_AQI_24h', 0)
                cat_24 = aqi_category(aqi_24)
                if cat_24 in ['Severe', 'Very Poor']:
                    render_alert("danger", "Critical Air Quality",
                        f"Air quality in {selected_city} is forecasted to be {cat_24} (AQI {aqi_24:.1f}) within the next 24 hours. {health_advice(cat_24)}")
                elif cat_24 in ['Poor', 'Moderate']:
                    render_alert("warning", "Elevated Air Quality",
                        f"Air quality in {selected_city} is forecasted to be {cat_24} (AQI {aqi_24:.1f}) within the next 24 hours. {health_advice(cat_24)}")
                else:
                    render_alert("success", "Healthy Air Quality",
                        f"Air quality in {selected_city} is forecasted to be {cat_24} (AQI {aqi_24:.1f}) within the next 24 hours. {health_advice(cat_24)}")
                
                st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
                csv_data = pd.DataFrame([res]).to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast CSV", data=csv_data, file_name=f'forecast_{selected_city}_{selected_date}.csv', mime='text/csv')

# ---------- POLLUTION ANALYTICS ----------
elif selected_page == "Pollution Analytics":
    st.markdown(f"""
    <div class="av-section-title">Pollution Analytics</div>
    <div class="av-section-sub">Pollutant composition analysis for {selected_city} on {selected_date}</div>
    """, unsafe_allow_html=True)
    
    if not df.empty and selected_city and selected_date:
        row = city_df[city_df['date'].dt.date == selected_date]
        if not row.empty:
            row = row.iloc[0]
            pollutants = ['pm2_5', 'pm10', 'no2', 'nh3', 'so2', 'co', 'o3']
            labels = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']
            vals = [row.get(p, 0) for p in pollutants]
            colors = ['#f43f5e', '#f59e0b', '#8b5cf6', '#06b6d4', '#10b981', '#22d3ee', '#a78bfa']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors, marker_line_width=0))
                fig.update_layout(title="Pollutant Concentrations")
                fig = plotly_theme(fig, height=420)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig2 = go.Figure(go.Pie(labels=labels, values=vals, hole=0.55, marker=dict(colors=colors, line=dict(width=0)),
                    textinfo='label+percent', textfont=dict(size=12, color='#f0f0f5')))
                fig2.update_layout(title="Relative Contribution")
                fig2 = plotly_theme(fig2, height=420)
                st.plotly_chart(fig2, use_container_width=True)

# ---------- TREND ANALYSIS ----------
elif selected_page == "Trend Analysis":
    st.markdown(f"""
    <div class="av-section-title">Trend Analysis</div>
    <div class="av-section-sub">Historical air quality patterns for {selected_city}</div>
    """, unsafe_allow_html=True)
    
    if not df.empty and selected_city:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=city_df['date'], y=city_df['aqi_24'], mode='lines',
            line=dict(color='#6366f1', width=1.5), fill='tozeroy', fillcolor='rgba(99,102,241,0.06)', name='AQI'))
        fig.update_layout(title="Historical AQI Trend (24h Target)")
        fig = plotly_theme(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
        pollutant_options = ['pm2_5', 'pm10', 'no2', 'co', 'so2', 'o3']
        selected_pollutant = st.selectbox("Compare Pollutant Trend", pollutant_options)
        
        pollutant_colors = {'pm2_5': '#f43f5e', 'pm10': '#f59e0b', 'no2': '#8b5cf6', 'co': '#22d3ee', 'so2': '#10b981', 'o3': '#a78bfa'}
        pc = pollutant_colors.get(selected_pollutant, '#6366f1')
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=city_df['date'], y=city_df[selected_pollutant], mode='lines',
            line=dict(color=pc, width=1.5), fill='tozeroy', fillcolor=f'rgba({int(pc[1:3],16)},{int(pc[3:5],16)},{int(pc[5:7],16)},0.06)', name=selected_pollutant.upper()))
        fig2.update_layout(title=f"Historical {selected_pollutant.upper()} Trend")
        fig2 = plotly_theme(fig2, height=380)
        st.plotly_chart(fig2, use_container_width=True)

# ---------- RISK ANALYSIS ----------
elif selected_page == "Risk Analysis":
    st.markdown(f"""
    <div class="av-section-title">Risk Analysis</div>
    <div class="av-section-sub">Health risk assessment for {selected_city} based on forecasted conditions</div>
    """, unsafe_allow_html=True)
    
    if predictor and selected_city and selected_date:
        res = predictor.predict(selected_city, str(selected_date), save_csv=False)
        if "error" not in res:
            aqi = res['Predicted_AQI_24h']
            cat = res['AQI_Category_24h']
            risk = res['Risk_Score_24h']
            c1, c2 = aqi_color(aqi)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric("Forecasted AQI", f"{aqi:.1f}", c1, c2, subtitle=cat)
            with col2:
                render_metric("Risk Score", f"{risk}", "#6366f1", "#8b5cf6", subtitle="out of 10.0")
            with col3:
                severity = "Low" if risk < 3 else ("Medium" if risk < 6 else ("High" if risk < 8 else "Critical"))
                sev_colors = {"Low": ("#10b981", "#059669"), "Medium": ("#f59e0b", "#d97706"), "High": ("#f43f5e", "#e11d48"), "Critical": ("#7E0023", "#5c0019")}
                sc1, sc2 = sev_colors[severity]
                render_metric("Severity", severity, sc1, sc2)
            
            st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)
            render_alert("info", "Risk Assessment Methodology",
                "The Risk Score is a normalized 0-10 metric that scales linearly with AQI (capped at 500). "
                "Scores above 6.0 indicate dangerous levels where prolonged outdoor exposure should be strictly avoided. "
                "Scores above 8.0 represent emergency conditions requiring immediate precautionary measures.")
        else:
            render_alert("danger", "Data Unavailable", "No data available for risk calculation on this date.")

# ---------- ALERTS ----------
elif selected_page == "Alerts":
    st.markdown(f"""
    <div class="av-section-title">Active Alerts</div>
    <div class="av-section-sub">Real-time health and safety alerts for {selected_city}</div>
    """, unsafe_allow_html=True)
    
    if predictor and selected_city and selected_date:
        res = predictor.predict(selected_city, str(selected_date), save_csv=False)
        if "error" not in res:
            alert = generate_alert(category=res['AQI_Category_24h'])
            level = alert['Alert Level']
            
            level_map = {'Critical': 'danger', 'Severe': 'danger', 'High': 'warning', 'Moderate': 'warning', 'Low': 'info', 'None': 'success'}
            alert_type = level_map.get(level, 'info')
            
            render_alert(alert_type, f"Alert Level: {level}",
                f"Based on a forecasted AQI of {res['Predicted_AQI_24h']:.1f} ({res['AQI_Category_24h']}) for {selected_city}.")
            
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 12px;">Health Recommendation</div>
                    <div style="font-size: 14px; color: var(--text-primary); line-height: 1.7;">{alert['Health Recommendation']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 12px;">Warning Message</div>
                    <div style="font-size: 14px; color: var(--text-primary); line-height: 1.7;">{alert['Warning Message']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            render_alert("warning", "No Active Alerts", "Cannot generate alerts without valid forecast data for this date.")

# ---------- MODEL PERFORMANCE ----------
elif selected_page == "Model Performance":
    st.markdown("""
    <div class="av-section-title">Model Performance</div>
    <div class="av-section-sub">Feature importance analysis and model diagnostics for the LightGBM ensemble</div>
    """, unsafe_allow_html=True)
    
    if predictor and '24h' in predictor.models:
        model = predictor.models['24h']
        if hasattr(model, 'feature_importances_'):
            drop_cols = ['city', 'date', 'split', 'aqi_24', 'aqi_48', 'aqi_72', 'aqi_bucket']
            features = [c for c in df.columns if c not in drop_cols]
            importances = model.feature_importances_
            
            if len(features) == len(importances):
                imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
                
                fig = go.Figure(go.Bar(
                    x=imp_df['Importance'].values[::-1], y=imp_df['Feature'].values[::-1],
                    orientation='h',
                    marker=dict(
                        color=imp_df['Importance'].values[::-1],
                        colorscale=[[0, '#1e1b4b'], [0.5, '#6366f1'], [1, '#a78bfa']],
                        line=dict(width=0)
                    )
                ))
                fig.update_layout(title="Top 20 Feature Importances  |  24h Model")
                fig = plotly_theme(fig, height=550)
                fig.update_layout(yaxis=dict(tickfont=dict(size=11)))
                st.plotly_chart(fig, use_container_width=True)
            else:
                render_alert("warning", "Feature Mismatch", "The number of features does not match model expectations.")
        else:
            render_alert("info", "Not Available", "This model does not expose feature importance scores.")
    else:
        render_alert("danger", "Model Not Loaded", "The 24h model could not be loaded.")

# ---------- ABOUT ----------
elif selected_page == "About":
    st.markdown("""
    <div class="av-hero" style="padding: 40px;">
        <div class="av-hero-title" style="font-size: 36px;">About AirVista</div>
        <div class="av-hero-desc">
            An end-to-end machine learning pipeline for air quality forecasting across major Indian cities. 
            Built with production-grade engineering practices and modern ML infrastructure.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="av-about-card">
            <div class="av-about-title">Data Pipeline</div>
            <div class="av-about-text">
                Robust data ingestion with automated imputation, outlier handling, and temporal feature engineering 
                including lag variables, rolling averages, and calendar-based interaction features.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="av-about-card" style="animation-delay: 0.1s;">
            <div class="av-about-title">ML Engine</div>
            <div class="av-about-text">
                LightGBM gradient-boosted decision trees selected after rigorous comparison against Random Forest, 
                XGBoost, CatBoost, LSTM, and Prophet. Achieves best RMSE across all cities.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="av-about-card" style="animation-delay: 0.2s;">
            <div class="av-about-title">Application Layer</div>
            <div class="av-about-text">
                Streamlit-powered interactive dashboard with Plotly visualizations. Fully decoupled from the 
                prediction and alert modules for clean separation of concerns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 12px; color: var(--text-muted); letter-spacing: 1px;">BUILT WITH</div>
        <div style="font-size: 15px; color: var(--text-secondary); margin-top: 8px; font-weight: 500;">
            Python  &middot;  Pandas  &middot;  LightGBM  &middot;  Plotly  &middot;  Streamlit  &middot;  Scikit-learn
        </div>
    </div>
    """, unsafe_allow_html=True)
