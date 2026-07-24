import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, '.')

# Force fresh module load
import importlib
if 'src.predict' in sys.modules:
    del sys.modules['src.predict']
if 'src' in sys.modules:
    del sys.modules['src']

from src.predict import AQIPredictor
p = AQIPredictor()

print("=== OPEN-METEO LIVE TEST (MUMBAI) ===")
res, live = p.predict_live('Mumbai')
print("Source:", live['source'])
print("PM2.5:", live['pm2_5'], "| PM10:", live['pm10'], "| NO2:", live['no2'])
print("24h AQI:", res.get('Predicted_AQI_24h'))
print("48h AQI:", res.get('Predicted_AQI_48h'))
print("72h AQI:", res.get('Predicted_AQI_72h'))
