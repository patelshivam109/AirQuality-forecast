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
    page_title="AirVista Pro — Live Air Quality Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN SYSTEM
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #040508;
    --bg-card: rgba(16, 18, 27, 0.7);
    --bg-card-hover: rgba(26, 29, 44, 0.85);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(99, 102, 241, 0.45);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-violet: #8b5cf6;
    --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    --shadow-glow: 0 0 35px rgba(99, 102, 241, 0.18);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

.stApp {
    background: var(--bg-primary) !important;
    background-image: 
        radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px),
        radial-gradient(circle at 50% -20%, rgba(99, 102, 241, 0.12) 0%, transparent 70%) !important;
    background-size: 32px 32px, 100% 100% !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}
.stApp > header { background: transparent !important; }

@keyframes pulseDot {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

section[data-testid="stSidebar"] {
    background: #07080c !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
section[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    padding: 11px 16px !important;
    border-radius: var(--radius-md) !important;
    margin: 0 8px !important;
    transition: all 0.25s ease !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(99, 102, 241, 0.12) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none !important; }

/* Top Header Bar */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    margin-bottom: 24px;
}
.top-nav-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.top-nav-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 700;
    color: #34d399;
}
.live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    animation: pulseDot 2s infinite ease-in-out;
}

.bento-card {
    background: var(--bg-card);
    backdrop-filter: blur(24px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px;
    transition: all 0.35s ease;
    margin-bottom: 16px;
}
.bento-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-glow);
    transform: translateY(-2px);
}

.kpi-title {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 34px;
    font-weight: 700;
    line-height: 1.1;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.stButton > button {
    background: var(--gradient-primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    width: auto !important;
    min-width: 220px !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & INGESTION
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
        import importlib
        import src.predict
        importlib.reload(src.predict)
        from src.predict import AQIPredictor
        return AQIPredictor(data_path='data/processed/featured_air_quality.csv', models_dir='models')
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

df = load_data()
predictor = get_predictor()
if predictor and not hasattr(predictor, 'fetch_live_air_quality'):
    import importlib
    import src.predict
    importlib.reload(src.predict)
    from src.predict import AQIPredictor
    predictor = AQIPredictor(data_path='data/processed/featured_air_quality.csv', models_dir='models')

def get_aqi_meta(aqi):
    if aqi is None or pd.isna(aqi):
        aqi = 0.0
    aqi = float(aqi)
    
    if aqi <= 50:
        return "Good", "#10b981", "#059669", "Minimal health impact. Air quality is ideal."
    elif aqi <= 100:
        return "Satisfactory", "#06b6d4", "#0891b2", "Minor breathing discomfort to sensitive individuals."
    elif aqi <= 200:
        return "Moderate", "#f59e0b", "#d97706", "Discomfort for asthma or heart disease patients."
    elif aqi <= 300:
        return "Poor", "#f43f5e", "#e11d48", "Breathing discomfort to most individuals on exposure."
    elif aqi <= 400:
        return "Very Poor", "#8b5cf6", "#7c3aed", "Respiratory illness risk on prolonged exposure."
    else:
        return "Severe", "#e11d48", "#9f1239", "Emergency conditions. High risk for all population groups."

def plotly_theme(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8"),
        title_font=dict(size=15, color="#f8fafc", family="Plus Jakarta Sans, sans-serif"),
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(255,255,255,0.03)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(255,255,255,0.03)"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# ============================================================================
# SIDEBAR NAVIGATION & MODES
# ============================================================================
st.sidebar.markdown("""
<div style="padding: 20px 16px 16px 16px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="
            width: 38px; height: 38px; 
            background: linear-gradient(135deg, #6366f1, #06b6d4); 
            border-radius: 10px; 
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; font-weight: 800; color: white;
        ">AV</div>
        <div>
            <div style="font-size: 17px; font-weight: 800; color: #f8fafc;">AirVista Pro</div>
            <div style="font-size: 10px; color: #64748b; font-weight: 700; letter-spacing: 1px;">REAL-TIME AI ENGINE</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Data Mode Switcher
st.sidebar.markdown("""
<div style="font-size: 10px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 10px 16px 4px 16px;">
    Execution Mode
</div>
""", unsafe_allow_html=True)

data_mode = st.sidebar.radio(
    "Execution Mode",
    ["🔴 Live Real-Time Forecast (Today)", "📜 Historical Dataset Explorer"],
    key="data_mode_radio"
)

st.sidebar.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

pages = ["Overview", "AQI Forecast", "Pollution Analytics", "Trend Analysis", "Risk Assessment", "Alert System", "Model Diagnostics", "System Info"]
selected_page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

# Global Filters based on mode
if not df.empty:
    cities = sorted(df['city'].unique())
    selected_city = st.sidebar.selectbox("Target Location", cities)
    
    if "Historical" in data_mode:
        city_df = df[df['city'] == selected_city].sort_values('date')
        dates = city_df['date'].dt.date.unique()
        default_idx = len(dates) - 1 if len(dates) > 0 else 0
        selected_date = st.sidebar.selectbox("Observation Date", dates, index=default_idx)
    else:
        selected_date = datetime.now().strftime("%Y-%m-%d")
        st.sidebar.info(f"Target Date: **Today ({selected_date})**")
else:
    selected_city, selected_date = None, None

# ============================================================================
# TOP NAV HEADER BAR
# ============================================================================
mode_label = "LIVE REAL-TIME INFERENCE (TODAY)" if "Live" in data_mode else "HISTORICAL DATASET SIMULATION"
st.markdown(f"""
<div class="top-nav">
    <div>
        <div class="top-nav-title">WORKSPACE / {mode_label} / {selected_city.upper() if selected_city else 'GLOBAL'}</div>
        <div style="font-size: 17px; font-weight: 700; color: #f8fafc; margin-top: 2px;">
            {selected_city if selected_city else 'Dashboard'} — {selected_date if selected_date else ''}
        </div>
    </div>
    <div class="top-nav-badge">
        <div class="live-dot"></div>
        {mode_label}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGES
# ============================================================================

# ----------------------------------------------------------------------------
# 1. OVERVIEW (HOME)
# ----------------------------------------------------------------------------
if selected_page == "Overview":
    if "Live" in data_mode:
        st.markdown(f"### 🔴 Live Real-Time Overview — {selected_city}")
        if predictor:
            live_data = predictor.fetch_live_air_quality(selected_city)
            c1, c2, c3, c4 = st.columns(4)
            metrics_data = [
                ("Live PM2.5", live_data['pm2_5'], "µg/m³", "#f43f5e"),
                ("Live PM10", live_data['pm10'], "µg/m³", "#f59e0b"),
                ("Live NO2", live_data['no2'], "ppb", "#8b5cf6"),
                ("Live CO", live_data['co'], "ppm", "#06b6d4")
            ]
            for col, (title, val, unit, color) in zip([c1, c2, c3, c4], metrics_data):
                with col:
                    st.markdown(f"""
                    <div class="bento-card">
                        <div class="kpi-title">{title}</div>
                        <div class="kpi-value" style="color: {color};">{val:.1f}</div>
                        <div style="font-size: 12px; color: #64748b;">Source: {live_data.get('source', 'Live Sensor')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
            
            # Predict live forecast
            res, _ = predictor.predict_live(selected_city)
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                st.markdown("### Today's Live Pollutant Footprint")
                pollutants = ['pm2_5', 'pm10', 'no2', 'nh3', 'so2', 'co', 'o3']
                vals = [float(live_data.get(p, 0)) for p in pollutants]
                labels = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']
                
                fig = go.Figure(go.Scatterpolar(
                    r=vals, theta=labels, fill='toself',
                    fillcolor='rgba(99, 102, 241, 0.2)',
                    line=dict(color='#6366f1', width=2)
                ))
                fig.update_layout(polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.05)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                ))
                fig = plotly_theme(fig, height=360)
                st.plotly_chart(fig, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_r:
                st.markdown('<div class="bento-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("### Tomorrow's Forecasted AQI")
                aqi_24 = res.get('Predicted_AQI_24h') if res else 0.0
                if aqi_24 is None: aqi_24 = 0.0
                cat_name, cat_color, _, advice = get_aqi_meta(aqi_24)
                st.markdown(f"""
                <div style="margin-top: 16px;">
                    <div style="font-size: 46px; font-weight: 800; font-family: 'JetBrains Mono'; color: {cat_color}; line-height: 1;">
                        {aqi_24:.0f}
                    </div>
                    <div style="font-size: 17px; font-weight: 700; color: #f8fafc; margin-top: 8px;">
                        {cat_name} Category
                    </div>
                    <div style="font-size: 13px; color: #94a3b8; margin-top: 12px; line-height: 1.6;">
                        {advice}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Historical mode
        if not df.empty and selected_city and selected_date:
            city_df = df[df['city'] == selected_city].sort_values('date')
            row_mask = city_df['date'].dt.date == selected_date
            if row_mask.any():
                row = city_df[row_mask].iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                metrics_data = [
                    ("PM2.5 Particulates", row.get('pm2_5', 0), "µg/m³", "#f43f5e"),
                    ("PM10 Particulates", row.get('pm10', 0), "µg/m³", "#f59e0b"),
                    ("Nitrogen Dioxide (NO2)", row.get('no2', 0), "ppb", "#8b5cf6"),
                    ("Carbon Monoxide (CO)", row.get('co', 0), "ppm", "#06b6d4")
                ]
                for col, (title, val, unit, color) in zip([c1, c2, c3, c4], metrics_data):
                    with col:
                        st.markdown(f"""
                        <div class="bento-card">
                            <div class="kpi-title">{title}</div>
                            <div class="kpi-value" style="color: {color};">{val:.1f}</div>
                            <div style="font-size: 12px; color: #64748b; font-weight: 500;">{unit}</div>
                        </div>
                        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 2. AQI FORECAST
# ----------------------------------------------------------------------------
elif selected_page == "AQI Forecast":
    st.markdown("## Multi-Horizon AQI Forecast Engine")
    
    if "Live" in data_mode:
        st.markdown(f"Running **Real-Time Live Forecast** for **{selected_city}** starting **Today ({selected_date})**.")
        
        with st.expander("🛠️ Customize Live Pollutant Reading (What-If Simulation)"):
            st.markdown("Adjust today's pollutant levels below to see how the AI model recalculates future 24h, 48h, and 72h forecasts:")
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                custom_pm25 = st.slider("Live PM2.5 (µg/m³)", 0.0, 500.0, 60.0)
                custom_pm10 = st.slider("Live PM10 (µg/m³)", 0.0, 600.0, 120.0)
            with c_s2:
                custom_no2 = st.slider("Live NO2 (ppb)", 0.0, 300.0, 35.0)
                custom_co = st.slider("Live CO (ppm)", 0.0, 50.0, 1.2)
            with c_s3:
                custom_so2 = st.slider("Live SO2 (ppb)", 0.0, 200.0, 15.0)
                custom_o3 = st.slider("Live O3 (ppb)", 0.0, 200.0, 40.0)
                
            use_custom = st.checkbox("Use these custom values instead of cloud API readings")
            
        if st.button("RUN LIVE AI INFERENCE"):
            if predictor and selected_city:
                with st.spinner("Fetching today's live readings & executing LightGBM models..."):
                    if use_custom:
                        cust_dict = {'pm2_5': custom_pm25, 'pm10': custom_pm10, 'no2': custom_no2, 'co': custom_co, 'so2': custom_so2, 'o3': custom_o3, 'nh3': 10.0, 'source': 'Custom Simulator'}
                        res, live_info = predictor.predict_live(selected_city, custom_pollutants=cust_dict)
                    else:
                        res, live_info = predictor.predict_live(selected_city)
                
                st.success(f"Live Inference Complete! Data Source: **{live_info.get('source', 'Live Sensor')}**")
                
                horizons = [
                    ("Tomorrow (+24h)", res.get('Predicted_AQI_24h')),
                    ("Day After (+48h)", res.get('Predicted_AQI_48h')),
                    ("3 Days Ahead (+72h)", res.get('Predicted_AQI_72h'))
                ]
                
                c1, c2, c3 = st.columns(3)
                for col, (label, val) in zip([c1, c2, c3], horizons):
                    val_disp = float(val) if val is not None else 0.0
                    cat_name, color1, _, advice = get_aqi_meta(val_disp)
                    with col:
                        st.markdown(f"""
                        <div class="bento-card" style="border-top: 3px solid {color1};">
                            <div class="kpi-title">{label}</div>
                            <div class="kpi-value" style="color: {color1};">{val_disp:.1f}</div>
                            <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">{cat_name}</div>
                            <div style="font-size: 12px; color: #94a3b8; line-height: 1.5;">{advice}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
                
                col_g1, col_g2 = st.columns([3, 2])
                with col_g1:
                    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                    st.markdown("### Environmental Risk Score Gauge")
                    risk_val = res.get('Risk_Score_24h', 0)
                    if risk_val is None: risk_val = 0.0
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=float(risk_val),
                        number={'suffix': ' / 10', 'font': {'size': 32, 'color': '#f8fafc', 'family': 'JetBrains Mono'}},
                        gauge={
                            'axis': {'range': [0, 10], 'tickcolor': "#475569"},
                            'bar': {'color': "#6366f1", 'thickness': 0.25},
                            'bgcolor': "rgba(0,0,0,0)",
                            'steps': [
                                {'range': [0, 3], 'color': "rgba(16, 185, 129, 0.2)"},
                                {'range': [3, 6], 'color': "rgba(245, 158, 11, 0.2)"},
                                {'range': [6, 8], 'color': "rgba(244, 63, 94, 0.2)"},
                                {'range': [8, 10], 'color': "rgba(225, 29, 72, 0.3)"}
                            ]
                        }
                    ))
                    fig = plotly_theme(fig, height=300)
                    st.plotly_chart(fig, width="stretch")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_g2:
                    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                    st.markdown("### Export Forecast Data")
                    csv_bytes = pd.DataFrame([res]).to_csv(index=False).encode('utf-8')
                    st.download_button("Export Forecast Package (.csv)", data=csv_bytes, file_name=f"live_aqi_forecast_{selected_city}_{selected_date}.csv", mime="text/csv")
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Historical dataset forecast
        st.markdown(f"Running predictions for **{selected_city}** on historical date **{selected_date}**.")
        if st.button("RUN PREDICTIVE INFERENCE"):
            if predictor and selected_city and selected_date:
                res = predictor.predict(selected_city, str(selected_date), save_csv=False)
                if "error" not in res:
                    c1, c2, c3 = st.columns(3)
                    horizons = [("24-Hour Horizon", res.get('Predicted_AQI_24h')), ("48-Hour Horizon", res.get('Predicted_AQI_48h')), ("72-Hour Horizon", res.get('Predicted_AQI_72h'))]
                    for col, (label, val) in zip([c1, c2, c3], horizons):
                        val_disp = float(val) if val is not None else 0.0
                        cat_name, color1, _, advice = get_aqi_meta(val_disp)
                        with col:
                            st.markdown(f"""
                            <div class="bento-card" style="border-top: 3px solid {color1};">
                                <div class="kpi-title">{label}</div>
                                <div class="kpi-value" style="color: {color1};">{val_disp:.1f}</div>
                                <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">{cat_name}</div>
                            </div>
                            """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 3. POLLUTION ANALYTICS
# ----------------------------------------------------------------------------
elif selected_page == "Pollution Analytics":
    st.markdown("## Chemical & Particulate Composition Analytics")
    if not df.empty and selected_city:
        city_df = df[df['city'] == selected_city].sort_values('date')
        row = city_df.iloc[-1]
        pollutants = ['pm2_5', 'pm10', 'no2', 'nh3', 'so2', 'co', 'o3']
        labels = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']
        vals = [float(row.get(p, 0)) for p in pollutants]
        
        c_bar, c_pie = st.columns(2)
        with c_bar:
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            fig = px.bar(x=labels, y=vals, color=labels, title=f"Concentration Breakdown — {selected_city}", labels={'x': 'Species', 'y': 'Concentration'})
            fig = plotly_theme(fig, height=380)
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with c_pie:
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            fig2 = px.pie(values=vals, names=labels, title="Proportional Mass Share", hole=0.5)
            fig2 = plotly_theme(fig2, height=380)
            st.plotly_chart(fig2, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 4. TREND ANALYSIS
# ----------------------------------------------------------------------------
elif selected_page == "Trend Analysis":
    st.markdown("## Longitudinal Time-Series Trends")
    if not df.empty and selected_city:
        city_df = df[df['city'] == selected_city].sort_values('date')
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        fig = px.line(city_df, x='date', y='aqi_24', title=f"Historical AQI Timeline — {selected_city}", color_discrete_sequence=['#6366f1'])
        fig = plotly_theme(fig, height=400)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 5. RISK ASSESSMENT
# ----------------------------------------------------------------------------
elif selected_page == "Risk Assessment":
    st.markdown("## Environmental Risk Matrix")
    if predictor and selected_city:
        if "Live" in data_mode:
            res, _ = predictor.predict_live(selected_city)
        else:
            res = predictor.predict(selected_city, str(selected_date), save_csv=False)
            
        if isinstance(res, dict) and "error" not in res:
            aqi = res.get('Predicted_AQI_24h', 100.0)
            if aqi is None: aqi = 100.0
            cat, color, _, advice = get_aqi_meta(aqi)
            risk = res.get('Risk_Score_24h', 2.0)
            if risk is None: risk = 2.0
            st.markdown(f"""
            <div class="bento-card" style="border-left: 4px solid {color};">
                <div style="font-size: 14px; font-weight: 700; color: {color};">RISK LEVEL: {cat.upper()}</div>
                <div style="font-size: 28px; font-weight: 800; color: #f8fafc; margin-top: 4px;">Calculated Risk Index: {risk} / 10.0</div>
                <div style="font-size: 14px; color: #cbd5e1; margin-top: 8px;">{advice}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No risk assessment data available for {selected_city} on {selected_date}.")

# ----------------------------------------------------------------------------
# 6. ALERT SYSTEM
# ----------------------------------------------------------------------------
elif selected_page == "Alert System":
    st.markdown("## Automated Health Advisory & Alert Dispatcher")
    if predictor and selected_city:
        if "Live" in data_mode:
            res, _ = predictor.predict_live(selected_city)
        else:
            res = predictor.predict(selected_city, str(selected_date), save_csv=False)
            
        if isinstance(res, dict) and "error" not in res:
            cat_val = res.get('AQI_Category_24h', 'Moderate')
            if cat_val is None: cat_val = 'Moderate'
            alert = generate_alert(category=cat_val)
            st.markdown(f"""
            <div class="bento-card">
                <div style="font-size: 11.5px; color: #64748b; font-weight: 700;">System Advisory</div>
                <div style="font-size: 20px; font-weight: 700; color: #f8fafc; margin-top: 4px;">Status Level: {alert.get('Alert Level', 'Normal')}</div>
                <div style="font-size: 14px; color: #94a3b8; margin-top: 12px;"><strong>Recommendation:</strong> {alert.get('Health Recommendation', 'No action needed.')}</div>
                <div style="font-size: 14px; color: #f43f5e; margin-top: 8px;"><strong>Warning:</strong> {alert.get('Warning Message', 'None.')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No advisory data available for {selected_city} on {selected_date}.")

# ----------------------------------------------------------------------------
# 7. MODEL DIAGNOSTICS
# ----------------------------------------------------------------------------
elif selected_page == "Model Diagnostics":
    st.markdown("## LightGBM Feature Importance Diagnostics")
    if predictor and '24h' in predictor.models:
        model = predictor.models['24h']
        if hasattr(model, 'feature_importances_'):
            drop_cols = ['city', 'date', 'split', 'aqi_24', 'aqi_48', 'aqi_72', 'aqi_bucket']
            features = [c for c in df.columns if c not in drop_cols]
            importances = model.feature_importances_
            if len(features) == len(importances):
                imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                imp_df = imp_df.sort_values(by='Importance', ascending=False).head(15)
                st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top 15 Feature Attribution Weights", color='Importance', color_continuous_scale=[[0, '#1e1b4b'], [0.5, '#6366f1'], [1, '#a78bfa']])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                fig = plotly_theme(fig, height=450)
                st.plotly_chart(fig, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 8. SYSTEM INFO
# ----------------------------------------------------------------------------
elif selected_page == "System Info":
    st.markdown("## Platform Architecture & Technical Stack")
    st.markdown("""
    <div class="bento-card">
        <div style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">Pipeline Overview</div>
        <div style="font-size: 14px; color: #94a3b8; line-height: 1.7;">
            AirVista Pro is an industrial-grade machine learning system engineering pipeline for city-level AQI forecasting across India.<br>
            • <strong>Data Pipeline:</strong> Missing value handling, lag variables, rolling means.<br>
            • <strong>Inference Engine:</strong> LightGBM multi-horizon decision tree regressors.<br>
            • <strong>Live Ingestion:</strong> Open-Meteo Cloud Air Quality API integration.<br>
            • <strong>Dashboard:</strong> Vercel/Linear styled responsive glassmorphic UI.
        </div>
    </div>
    """, unsafe_allow_html=True)
