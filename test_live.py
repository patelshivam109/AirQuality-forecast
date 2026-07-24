import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, '.')

if 'src.predict' in sys.modules:
    del sys.modules['src.predict']
if 'src' in sys.modules:
    del sys.modules['src']

from src.predict import AQIPredictor
p = AQIPredictor()

# Test ALL 26 cities
all_cities = sorted(p.df['city'].unique())
print(f"Total cities: {len(all_cities)}\n")

for city in all_cities:
    res, live = p.predict_live(city)
    status = "LIVE" if "aqi.in" in live['source'] else "FALLBACK"
    print(f"[{status}] {city:25s} | Source: {live['source']:40s} | PM2.5: {live['pm2_5']}")
