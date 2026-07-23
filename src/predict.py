import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

class AQIPredictor:
    def __init__(self, data_path='data/processed/featured_air_quality.csv', models_dir='models'):
        """
        Initializes the predictor by loading the dataset (for feature lookup) and models.
        """
        # Ensure outputs directory exists
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
        # We assume 3 models exist: best_model.pkl (24h), best_model_48.pkl, best_model_72.pkl
        # If 48/72 are missing, we fall back to 24h model as a safety catch, but they should be trained.
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
        """
        Risk score out of 10. 
        AQI 500+ is max risk (10).
        """
        risk = (aqi / 500) * 10
        return round(min(max(risk, 0), 10), 1)

    def predict(self, city, date, save_csv=True, output_path='outputs/predictions/forecasts.csv'):
        """
        Predicts 24h, 48h, and 72h AQI for a given city and date.
        
        :param city: str, Name of the city
        :param date: str, Date in 'YYYY-MM-DD' format
        :param save_csv: bool, Whether to append the result to a CSV
        :param output_path: str, Path to save the CSV
        :return: dict, Contains predictions and risk metrics
        """
        # Parse date
        target_date = pd.to_datetime(date)
        
        # Filter dataset for the specific city and date
        record = self.df[(self.df['city'] == city) & (self.df['date'] == target_date)]
        
        if record.empty:
            return {"error": f"No data found for {city} on {date}."}
            
        # Prepare features (same as training)
        drop_cols = ['city', 'date', 'split', 'aqi_24', 'aqi_48', 'aqi_72', 'aqi_bucket']
        features = [c for c in self.df.columns if c not in drop_cols]
        
        X = record[features]
        # Impute missing with median of the training data (approx by median of df)
        X = X.fillna(self.df[features].median())
        
        results = {
            'City': city,
            'Date': date,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Make predictions
        for horizon in ['24h', '48h', '72h']:
            if horizon in self.models:
                pred = float(self.models[horizon].predict(X)[0])
                results[f'Predicted_AQI_{horizon}'] = round(pred, 1)
            else:
                results[f'Predicted_AQI_{horizon}'] = None
                
        # Calculate Category and Risk based on the primary 24h forecast
        if results.get('Predicted_AQI_24h') is not None:
            results['AQI_Category_24h'] = self._get_category(results['Predicted_AQI_24h'])
            results['Risk_Score_24h'] = self._get_risk_score(results['Predicted_AQI_24h'])
            
        # Save to CSV
        if save_csv:
            res_df = pd.DataFrame([results])
            # Append if file exists, else write new
            if os.path.exists(output_path):
                res_df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                res_df.to_csv(output_path, mode='w', header=True, index=False)
            print(f"Prediction saved to {output_path}")
            
        return results

# Example usage (for testing)
if __name__ == "__main__":
    predictor = AQIPredictor(data_path='data/processed/featured_air_quality.csv', models_dir='models')
    # Use a sample city and date known to exist in the dataset
    sample_city = "Delhi"
    sample_date = "2020-03-21"  # Valid date from dataset
    
    print(f"Generating forecast for {sample_city} on {sample_date}...")
    res = predictor.predict(sample_city, sample_date, save_csv=True, output_path='outputs/predictions/forecasts.csv')
    
    import pprint
    pprint.pprint(res)
