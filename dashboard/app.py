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

# --- PAGE CONFIG ---
st.set_page_config(page_title="AQI Forecast & Analytics", page_icon="🌤️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
    color: #4CAF50;
}
.metric-label {
    font-size: 14px;
    color: #AAAAAA;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA & PREDICTOR ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/featured_air_quality.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_predictor():
    try:
        return AQIPredictor(data_path='data/processed/featured_air_quality.csv', models_dir='models')
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

df = load_data()
predictor = get_predictor()

# --- SIDEBAR ---
st.sidebar.title("🌬️ AQI Dashboard")
st.sidebar.markdown("---")

# Navigation
pages = [
    "Home", 
    "AQI Forecast", 
    "Pollution Analytics", 
    "Trend Analysis", 
    "Risk Analysis", 
    "Alerts", 
    "Model Performance", 
    "About"
]
selected_page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")

# Global Filters
st.sidebar.subheader("Filters")
if not df.empty:
    cities = sorted(df['city'].unique())
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    # Filter dates for selected city
    city_df = df[df['city'] == selected_city].sort_values('date')
    dates = city_df['date'].dt.date.unique()
    
    # Default to the most recent date available for the city
    default_date_idx = len(dates) - 1 if len(dates) > 0 else 0
    selected_date = st.sidebar.selectbox("Select Date", dates, index=default_date_idx)
else:
    selected_city = None
    selected_date = None
    st.sidebar.warning("No data available.")

st.sidebar.markdown("---")
st.sidebar.info("Using LightGBM for robust tabular forecasting.")

# --- HELPERS ---
def create_metric_card(label, value, color="#4CAF50"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def get_color_for_aqi(aqi):
    if aqi <= 50: return "#00E400"
    elif aqi <= 100: return "#FFFF00"
    elif aqi <= 200: return "#FF7E00"
    elif aqi <= 300: return "#FF0000"
    elif aqi <= 400: return "#8F3F97"
    else: return "#7E0023"

# --- PAGES ---

if selected_page == "Home":
    st.title("Air Quality Intelligence System")
    st.markdown("""
    Welcome to the modern Air Quality Forecasting and Analytics platform. 
    Use the sidebar to navigate through detailed predictions, historical trends, and risk analysis for major cities.
    """)
    
    if not df.empty:
        st.subheader(f"Snapshot for {selected_city} on {selected_date}")
        row = city_df[city_df['date'].dt.date == selected_date].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: create_metric_card("PM2.5", round(row['pm2_5'], 2))
        with c2: create_metric_card("PM10", round(row['pm10'], 2))
        with c3: create_metric_card("NO2", round(row['no2'], 2))
        with c4: create_metric_card("CO", round(row['co'], 2))
        
        st.image("https://images.unsplash.com/photo-1534274988757-a28bf1a57c17?auto=format&fit=crop&w=1200&q=80", use_column_width=True, caption="Urban Atmosphere")

elif selected_page == "AQI Forecast":
    st.title(f"🌤️ AQI Forecast for {selected_city}")
    st.markdown(f"**Base Date:** {selected_date}")
    
    # AQI Scale Legend so users understand what the numbers mean
    with st.expander("📊 What do AQI numbers mean? (Click to expand)"):
        st.markdown("""
        | AQI Range | Category | Health Impact | Color |
        |-----------|----------|---------------|-------|
        | 0 – 50 | **Good** | Minimal impact | 🟢 |
        | 51 – 100 | **Satisfactory** | Minor breathing discomfort to sensitive people | 🟡 |
        | 101 – 200 | **Moderate** | Breathing discomfort to people with lung/heart disease | 🟠 |
        | 201 – 300 | **Poor** | Breathing discomfort to most people on prolonged exposure | 🔴 |
        | 301 – 400 | **Very Poor** | Respiratory illness on prolonged exposure | 🟣 |
        | 401 – 500+ | **Severe** | Affects healthy people. Serious impact on those with existing diseases | ⚫ |
        """)
    
    if st.button("🔮 Generate Forecast"):
        if predictor and selected_city and selected_date:
            with st.spinner("Running LightGBM Models..."):
                res = predictor.predict(selected_city, str(selected_date), save_csv=False)
                
            if "error" in res:
                st.error(res["error"])
            else:
                st.success("Forecast Generated Successfully!")
                
                # Helper to get category from AQI
                def _cat(aqi):
                    if aqi <= 50: return "Good"
                    elif aqi <= 100: return "Satisfactory"
                    elif aqi <= 200: return "Moderate"
                    elif aqi <= 300: return "Poor"
                    elif aqi <= 400: return "Very Poor"
                    else: return "Severe"
                
                def _health_msg(cat):
                    msgs = {
                        "Good": "✅ Air quality is ideal. Enjoy outdoor activities!",
                        "Satisfactory": "🙂 Air is acceptable. Sensitive individuals should be cautious.",
                        "Moderate": "😷 People with respiratory/heart conditions should limit outdoor exertion.",
                        "Poor": "⚠️ Everyone should reduce prolonged outdoor exertion. Avoid heavy exercise.",
                        "Very Poor": "🚨 Avoid all outdoor physical activity. Keep windows closed.",
                        "Severe": "🆘 HEALTH EMERGENCY! Stay indoors. Use air purifiers if available."
                    }
                    return msgs.get(cat, "")
                
                # Display forecasts with category and color
                horizons = [
                    ("Next 24 Hours", res.get('Predicted_AQI_24h')),
                    ("Next 48 Hours", res.get('Predicted_AQI_48h')),
                    ("Next 72 Hours", res.get('Predicted_AQI_72h'))
                ]
                
                c1, c2, c3 = st.columns(3)
                for col, (label, aqi_val) in zip([c1, c2, c3], horizons):
                    if aqi_val is not None:
                        cat = _cat(aqi_val)
                        color = get_color_for_aqi(aqi_val)
                        with col:
                            create_metric_card(label, f"{aqi_val}", color)
                            st.markdown(f"<div style='text-align:center; font-size:18px; font-weight:bold; color:{color};'>{cat}</div>", unsafe_allow_html=True)
                            st.caption(_health_msg(cat))
                
                st.markdown("---")
                
                # Risk Gauge
                st.markdown("### 🎯 Overall Risk Assessment")
                risk_val = res.get('Risk_Score_24h', 0)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_val,
                    number = {'suffix': ' / 10'},
                    title = {'text': f"Risk Score — {res.get('AQI_Category_24h', 'N/A')}"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 10], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 2], 'color': "#00E400"},
                            {'range': [2, 4], 'color': "#FFFF00"},
                            {'range': [4, 6], 'color': "#FF7E00"},
                            {'range': [6, 8], 'color': "#FF0000"},
                            {'range': [8, 10], 'color': "#7E0023"}],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_val
                        }
                    }
                ))
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Box
                aqi_24 = res.get('Predicted_AQI_24h', 0)
                cat_24 = _cat(aqi_24)
                if cat_24 in ['Severe', 'Very Poor']:
                    st.error(f"🚨 **DANGER:** Air quality in {selected_city} is expected to be **{cat_24}** in the next 24 hours. {_health_msg(cat_24)}")
                elif cat_24 in ['Poor', 'Moderate']:
                    st.warning(f"⚠️ **CAUTION:** Air quality in {selected_city} is expected to be **{cat_24}** in the next 24 hours. {_health_msg(cat_24)}")
                else:
                    st.success(f"✅ **ALL CLEAR:** Air quality in {selected_city} is expected to be **{cat_24}** in the next 24 hours. {_health_msg(cat_24)}")
                
                st.markdown("---")
                
                # Download CSV
                csv_data = pd.DataFrame([res]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Forecast (CSV)",
                    data=csv_data,
                    file_name=f'forecast_{selected_city}_{selected_date}.csv',
                    mime='text/csv'
                )

elif selected_page == "Pollution Analytics":
    st.title("Pollution Analytics")
    if not df.empty and selected_city and selected_date:
        row = city_df[city_df['date'].dt.date == selected_date]
        if not row.empty:
            row = row.iloc[0]
            pollutants = ['pm2_5', 'pm10', 'no2', 'nh3', 'so2', 'co', 'o3']
            vals = [row.get(p, 0) for p in pollutants]
            
            fig = px.bar(x=pollutants, y=vals, color=pollutants, title=f"Pollutant Breakdown for {selected_city} ({selected_date})",
                         labels={'x': 'Pollutant', 'y': 'Concentration'})
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.pie(values=vals, names=pollutants, title="Relative Contribution", hole=0.4)
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

elif selected_page == "Trend Analysis":
    st.title("Historical Trend Analysis")
    if not df.empty and selected_city:
        st.markdown(f"Viewing historical trends for **{selected_city}**.")
        fig = px.line(city_df, x='date', y='aqi_24', title="Historical AQI Trend (24h Target)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        selected_pollutant = st.selectbox("Compare Pollutant", ['pm2_5', 'pm10', 'no2', 'co', 'so2', 'o3'])
        fig2 = px.line(city_df, x='date', y=selected_pollutant, title=f"Historical {selected_pollutant.upper()} Trend", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

elif selected_page == "Risk Analysis":
    st.title("Risk Analysis")
    st.markdown("Understand the health implications and risks associated with the forecasted air quality.")
    if predictor and selected_city and selected_date:
        res = predictor.predict(selected_city, str(selected_date), save_csv=False)
        if "error" not in res:
            aqi = res['Predicted_AQI_24h']
            cat = res['AQI_Category_24h']
            risk = res['Risk_Score_24h']
            
            st.subheader(f"Current Forecast: {aqi} AQI ({cat})")
            st.progress(min(int((aqi/500)*100), 100))
            st.markdown(f"**Calculated Risk Score:** {risk} / 10.0")
            
            st.info("The Risk Score is a normalized 0-10 metric scaling alongside the AQI. A score above 6.0 generally represents dangerous levels of prolonged exposure.")
        else:
            st.error("No data available to calculate risk.")

elif selected_page == "Alerts":
    st.title("Active Alerts")
    if predictor and selected_city and selected_date:
        res = predictor.predict(selected_city, str(selected_date), save_csv=False)
        if "error" not in res:
            alert = generate_alert(category=res['AQI_Category_24h'])
            
            level = alert['Alert Level']
            if level in ['Critical', 'Severe', 'High']:
                st.error(f"🚨 **ALERT LEVEL:** {level}")
            elif level == 'Moderate':
                st.warning(f"⚠️ **ALERT LEVEL:** {level}")
            else:
                st.success(f"✅ **ALERT LEVEL:** {level}")
                
            st.markdown("### Health Recommendation")
            st.info(alert['Health Recommendation'])
            
            st.markdown("### Warning Message")
            st.warning(alert['Warning Message'])
        else:
            st.warning("Cannot generate alerts without a valid forecast.")

elif selected_page == "Model Performance":
    st.title("Model Performance & Diagnostics")
    st.markdown("The backend uses a finely tuned LightGBM Regressor.")
    
    if predictor and '24h' in predictor.models:
        model = predictor.models['24h']
        if hasattr(model, 'feature_importances_'):
            # Reconstruct feature names from the dataset since lightgbm drops them sometimes
            drop_cols = ['city', 'date', 'split', 'aqi_24', 'aqi_48', 'aqi_72', 'aqi_bucket']
            features = [c for c in df.columns if c not in drop_cols]
            
            importances = model.feature_importances_
            if len(features) == len(importances):
                imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
                
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top 20 Feature Importances (24h Model)", template="plotly_dark")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature count mismatch.")
        else:
            st.warning("Model does not expose feature importances.")
    else:
        st.error("24h Model not loaded.")

elif selected_page == "About":
    st.title("About This Project")
    st.markdown("""
    ## 🌤️ Air Quality Forecasting System
    This project is a complete end-to-end Machine Learning pipeline that predicts Air Quality Index (AQI) across major cities in India.
    
    ### Architecture
    1. **Data Processing:** Robust imputation, outlier handling, and rolling/lag feature engineering.
    2. **Modeling:** LightGBM (Gradient Boosting) used for tabular time-series forecasting.
    3. **Application:** Streamlit Dashboard for UI, fully decoupled from the prediction modules.
    
    ### Developer
    Built with Python, Pandas, LightGBM, and Plotly.
    """)
