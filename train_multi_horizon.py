import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/processed/featured_air_quality.csv')
drop_cols_24 = ['city', 'date', 'split', 'aqi_48', 'aqi_72', 'aqi_bucket']
drop_cols_48 = ['city', 'date', 'split', 'aqi_24', 'aqi_72', 'aqi_bucket']
drop_cols_72 = ['city', 'date', 'split', 'aqi_24', 'aqi_48', 'aqi_bucket']

train_df = df[df['split'] == 'train']

for horizon, target, drop_cols in [('48', 'aqi_48', drop_cols_48), ('72', 'aqi_72', drop_cols_72)]:
    print(f"Training model for {horizon}h...")
    features = [c for c in df.columns if c not in drop_cols and c != target]
    
    X_train = train_df[features].fillna(train_df[features].median())
    y_train = train_df[target]
    
    # Drop rows where target is NaN
    mask = ~y_train.isna()
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    model = LGBMRegressor(random_state=42, n_estimators=50, learning_rate=0.1, max_depth=5, verbose=-1)
    model.fit(X_train, y_train)
    
    joblib.dump(model, f'models/best_model_{horizon}.pkl')
    print(f"Saved models/best_model_{horizon}.pkl")
