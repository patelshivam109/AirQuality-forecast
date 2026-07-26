# AI-Powered Air Quality Forecasting & Health Alert System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-0080FF?style=for-the-badge&logo=scikit-learn&logoColor=white)

An AI-driven environmental decision support system that helps citizens and authorities evaluate current air quality, forecast future AQI, and make intelligent health safety decisions.

The system uses machine learning-based regression algorithms (LightGBM), real-time web scraping integration for live CPCB ground station data, and health risk optimization techniques to identify pollution trends, reduce exposure risks, and support better public health decisions.

## Features

- **Live Real-Time Data Scraping**: Fetches live pollutant levels directly from active CPCB ground stations via `aqi.in`.
- **Multi-Horizon AI Forecasting**: Uses a LightGBM regression model to predict the Air Quality Index (AQI) 24 hours, 48 hours, and 72 hours into the future.
- **Environmental Risk Engine**: Calculates a dynamic health risk score out of 10 based on predicted pollutants (specifically penalizing high PM2.5 and PM10).
- **Health Advisories**: Dispatches automated, targeted safety recommendations (e.g., "Stay Indoors", "Wear N95 Mask", "Normal Activity").
- **Premium Bento Grid Dashboard**: Features a modern, highly responsive Streamlit user interface inspired by premium web design standards.
- **Explainable AI (SHAP)**: Provides feature importance charts to explain which specific pollutants are driving the model's AQI predictions.
- **Interactive EDA**: Visualize historical trends, pollutant correlation heatmaps, and city-wise AQI distribution boxplots using Plotly..

## Project Structure

```text
├── dashboard/
│   └── app.py                  # Main Streamlit dashboard application
├── data/
│   └── processed/
│       └── featured_air_quality.csv # Cleaned & engineered dataset
├── models/
│   ├── lightgbm_aqi_24h.pkl    # 24h Horizon Model
│   ├── lightgbm_aqi_48h.pkl    # 48h Horizon Model
│   └── lightgbm_aqi_72h.pkl    # 72h Horizon Model
├── notebooks/                  # Jupyter notebooks for EDA and training
├── src/
│   ├── predict.py              # AI Prediction logic and Live Scraper
│   └── alerts.py               # Health Risk Scoring & Alert Logic
├── test_live.py                # Script to test the live scraper
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd AirQuality
```

**2. Create a virtual environment**
```bash
python -m venv venv
```

**3. Activate the virtual environment**
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Application

To launch the interactive dashboard, run:

```bash
streamlit run dashboard/app.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

## Data Pipeline

1. **Training Data**: Historical Air Quality Data (15,000+ records across 26 Indian cities).
2. **Feature Engineering**: Created 1-day, 2-day, and 3-day lag features, as well as 3-day and 7-day rolling averages to capture temporal sequences.
3. **Modeling**: Selected LightGBM over XGBoost and Random Forest for its superior accuracy (R² Score: 0.941) in handling complex time-series regression.
4. **Live Inference**: Uses `BeautifulSoup` to scrape exact ground-station readouts, injecting live sensor data into the model's lag footprint for true multi-horizon predictions.

## Contributing

Feel free to submit Pull Requests or open Issues if you want to enhance the dashboard's design, add new features, or improve the scraper logic!

## License

This project is submitted as an Internship Project Submission.
