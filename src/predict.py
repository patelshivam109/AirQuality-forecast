import os
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime

CITY_COORDINATES = {
    'Ahmedabad': (23.0225, 72.5714),
    'Aizawl': (23.7271, 92.7176),
    'Amaravati': (16.5131, 80.5165),
    'Amritsar': (31.6340, 74.8723),
    'Bengaluru': (12.9716, 77.5946),
    'Bhopal': (23.2599, 77.4126),
    'Brajrajnagar': (21.8215, 83.9213),
    'Chandigarh': (30.7333, 76.7794),
    'Chennai': (13.0827, 80.2707),
    'Coimbatore': (11.0168, 76.9558),
    'Delhi': (28.6139, 77.2090),
    'Ernakulam': (9.9816, 76.2999),
    'Gurugram': (28.4595, 77.0266),
    'Guwahati': (26.1445, 91.7362),
    'Hyderabad': (17.3850, 78.4867),
    'Jaipur': (26.9124, 75.7873),
    'Jorapokhar': (23.7088, 86.4103),
    'Kochi': (9.9312, 76.2673),
    'Kolkata': (22.5726, 88.3639),
    'Lucknow': (26.8467, 80.9462),
    'Mumbai': (19.0760, 72.8777),
    'Patna': (25.5941, 85.1376),
    'Shillong': (25.5788, 91.8933),
    'Talcher': (20.9500, 85.2333),
    'Thiruvananthapuram': (8.5241, 76.9366),
    'Visakhapatnam': (17.6868, 83.2185)
}

class AQIPredictor:
    def __init__(self, data_path='data/processed/featured_air_quality.csv', models_dir='models'):
        """
        Initializes the predictor by loading the dataset (for feature lookup) and models.
        """
        os.makedirs('outputs/predictions', exist_ok=True)
        self.data_path = data_path
        self.models_dir = models_dir
        self.df = None
        self.models = {}
        
        self._load_data()
        self._load_models()
        
    def _load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path, parse_dates=['date'])
        else:
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
    def _load_models(self):
        model_paths = {
            '24h': os.path.join(self.models_dir, 'best_model.pkl'),
            '48h': os.path.join(self.models_dir, 'best_model_48.pkl'),
            '72h': os.path.join(self.models_dir, 'best_model_72.pkl')
        }
        
        for horizon, path in model_paths.items():
            if os.path.exists(path):
                self.models[horizon] = joblib.load(path)
            else:
                print(f"Warning: Model {path} not found.")
                
    def _get_category(self, aqi):
        if aqi <= 50: return 'Good'
        elif aqi <= 100: return 'Satisfactory'
        elif aqi <= 200: return 'Moderate'
        elif aqi <= 300: return 'Poor'
        elif aqi <= 400: return 'Very Poor'
        else: return 'Severe'
        
    def _get_risk_score(self, aqi):
        risk = (aqi / 500) * 10
        return round(min(max(risk, 0), 10), 1)

    def fetch_live_air_quality(self, city):
        """
        Fetches current real-time pollutant measurements for a city via Open-Meteo API.
        """
        coords = CITY_COORDINATES.get(city, (28.6139, 77.2090)) # Default to Delhi if unknown
        lat, lon = coords
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,ammonia"
        
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get('current', {})
                def safe_float(val, default):
                    return float(val) if val is not None else default
                    
                return {
                    'pm2_5': safe_float(data.get('pm2_5'), 45.0),
                    'pm10': safe_float(data.get('pm10'), 95.0),
                    'no2': safe_float(data.get('nitrogen_dioxide'), 25.0),
                    'so2': safe_float(data.get('sulphur_dioxide'), 12.0),
                    'co': safe_float(data.get('carbon_monoxide'), 0.8),
                    'o3': safe_float(data.get('ozone'), 30.0),
                    'nh3': safe_float(data.get('ammonia'), 10.0),
                    'source': 'Live Open-Meteo Cloud API'
                }
        except Exception as e:
            print(f"Live API lookup failed: {e}")
            
        # Fallback to latest dataset values if offline
        city_rows = self.df[self.df['city'] == city]
        if not city_rows.empty:
            latest = city_rows.sort_values('date').iloc[-1]
            return {
                'pm2_5': float(latest.get('pm2_5', 50.0)),
                'pm10': float(latest.get('pm10', 100.0)),
                'no2': float(latest.get('no2', 30.0)),
                'so2': float(latest.get('so2', 15.0)),
                'co': float(latest.get('co', 1.0)),
                'o3': float(latest.get('o3', 35.0)),
                'nh3': float(latest.get('nh3', 10.0)),
                'source': 'Latest Station Baseline'
            }
        return {'pm2_5': 50.0, 'pm10': 100.0, 'no2': 30.0, 'so2': 15.0, 'co': 1.0, 'o3': 35.0, 'nh3': 10.0, 'source': 'Default Baseline'}

    def predict_live(self, city, custom_pollutants=None, save_csv=True, output_path='outputs/predictions/forecasts.csv'):
        """
        Takes real-time (API or custom) pollutant readings for today and runs trained LightGBM models
        to forecast 24h, 48h, and 72h future AQI.
        """
        # Fetch live readings if not provided
        if custom_pollutants is None:
            live_data = self.fetch_live_air_quality(city)
        else:
            live_data = custom_pollutants
            live_data['source'] = custom_pollutants.get('source', 'Custom Simulation Input')
            
        # Get baseline row for feature schema and lag structures
        city_df = self.df[self.df['city'] == city]
        if city_df.empty:
            record = self.df.iloc[[-1]].copy()
        else:
            record = city_df.sort_values('date').iloc[[-1]].copy()
            
        # Update baseline record with live pollutant measurements
        for pol in ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']:
            if pol in live_data and pol in record.columns:
                record[pol] = live_data[pol]
                
        # Drop non-feature target columns
        drop_cols = ['city', 'date', 'split', 'aqi_24', 'aqi_48', 'aqi_72', 'aqi_bucket']
        features = [c for c in self.df.columns if c not in drop_cols]
        
        X = record[features].fillna(self.df[features].median())
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        results = {
            'City': city,
            'Date': today_str,
            'Mode': 'Real-Time Live Forecast',
            'Data_Source': live_data.get('source', 'Unknown'),
            'Current_PM2.5': round(live_data.get('pm2_5', 0), 1),
            'Current_PM10': round(live_data.get('pm10', 0), 1),
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for horizon in ['24h', '48h', '72h']:
            if horizon in self.models:
                pred = float(self.models[horizon].predict(X)[0])
                results[f'Predicted_AQI_{horizon}'] = round(pred, 1)
            else:
                results[f'Predicted_AQI_{horizon}'] = None
                
        if results.get('Predicted_AQI_24h') is not None:
            results['AQI_Category_24h'] = self._get_category(results['Predicted_AQI_24h'])
            results['Risk_Score_24h'] = self._get_risk_score(results['Predicted_AQI_24h'])
            
        if save_csv:
            res_df = pd.DataFrame([results])
            if os.path.exists(output_path):
                res_df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                res_df.to_csv(output_path, mode='w', header=True, index=False)
                
        return results, live_data

    def predict(self, city, date, save_csv=True, output_path='outputs/predictions/forecasts.csv'):
        target_date = pd.to_datetime(date)
        record = self.df[(self.df['city'] == city) & (self.df['date'] == target_date)]
        
        if record.empty:
            return {"error": f"No data found for {city} on {date}."}
            
        drop_cols = ['city', 'date', 'split', 'aqi_24', 'aqi_48', 'aqi_72', 'aqi_bucket']
        features = [c for c in self.df.columns if c not in drop_cols]
        
        X = record[features].fillna(self.df[features].median())
        
        results = {
            'City': city,
            'Date': date,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for horizon in ['24h', '48h', '72h']:
            if horizon in self.models:
                pred = float(self.models[horizon].predict(X)[0])
                results[f'Predicted_AQI_{horizon}'] = round(pred, 1)
            else:
                results[f'Predicted_AQI_{horizon}'] = None
                
        if results.get('Predicted_AQI_24h') is not None:
            results['AQI_Category_24h'] = self._get_category(results['Predicted_AQI_24h'])
            results['Risk_Score_24h'] = self._get_risk_score(results['Predicted_AQI_24h'])
            
        if save_csv:
            res_df = pd.DataFrame([results])
            if os.path.exists(output_path):
                res_df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                res_df.to_csv(output_path, mode='w', header=True, index=False)
            
        return results

if __name__ == "__main__":
    predictor = AQIPredictor(data_path='data/processed/featured_air_quality.csv', models_dir='models')
    print("Testing Live API Real-Time Forecast for Delhi...")
    res, live_data = predictor.predict_live("Delhi")
    import pprint
    pprint.pprint(res)
