import json
import os
import pandas as pd
import numpy as np
import joblib

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 5: Model Development\n",
                "\n",
                "In this phase, we train and compare multiple machine learning models to forecast `aqi_24` (AQI 24 hours/1 day ahead). We will evaluate Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import os\n",
                "import time\n",
                "import joblib\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from xgboost import XGBRegressor\n",
                "from lightgbm import LGBMRegressor\n",
                "from catboost import CatBoostRegressor\n",
                "\n",
                "from sklearn.model_selection import RandomizedSearchCV\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Data and Prepare Splits\n",
                "We use the pre-calculated `split` column to separate our data into Train, Validation, and Test sets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('../data/processed/featured_air_quality.csv')\n",
                "\n",
                "# Drop categorical and unnecessary columns for modeling\n",
                "drop_cols = ['city', 'date', 'split', 'aqi_48', 'aqi_72', 'aqi_bucket']\n",
                "features = [c for c in df.columns if c not in drop_cols and c != 'aqi_24']\n",
                "\n",
                "train_df = df[df['split'] == 'train']\n",
                "val_df = df[df['split'] == 'val']\n",
                "test_df = df[df['split'] == 'test']\n",
                "\n",
                "# We will combine train and val for Grid/Randomized search CV which handles its own CV,\n",
                "# or we can just use the provided split. For simplicity, we'll train on `train_df` \n",
                "# and test on `test_df` for final evaluation.\n",
                "X_train, y_train = train_df[features], train_df['aqi_24']\n",
                "X_test, y_test = test_df[features], test_df['aqi_24']\n",
                "\n",
                "# Handle any remaining NaNs in features by filling with median\n",
                "X_train = X_train.fillna(X_train.median())\n",
                "X_test = X_test.fillna(X_train.median())\n",
                "\n",
                "print(f\"Train shape: {X_train.shape}, Test shape: {X_test.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Model Definition and Hyperparameter Tuning\n",
                "We define the models and hyperparameter grids for tuning via `RandomizedSearchCV`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = {\n",
                "    'Linear Regression': {\n",
                "        'model': LinearRegression(),\n",
                "        'params': {}\n",
                "    },\n",
                "    'Random Forest': {\n",
                "        'model': RandomForestRegressor(random_state=42, n_jobs=-1),\n",
                "        'params': {\n",
                "            'n_estimators': [50, 100],\n",
                "            'max_depth': [None, 10, 20]\n",
                "        }\n",
                "    },\n",
                "    'XGBoost': {\n",
                "        'model': XGBRegressor(random_state=42, objective='reg:squarederror'),\n",
                "        'params': {\n",
                "            'n_estimators': [50, 100],\n",
                "            'learning_rate': [0.01, 0.1],\n",
                "            'max_depth': [3, 5, 7]\n",
                "        }\n",
                "    },\n",
                "    'LightGBM': {\n",
                "        'model': LGBMRegressor(random_state=42, verbose=-1),\n",
                "        'params': {\n",
                "            'n_estimators': [50, 100],\n",
                "            'learning_rate': [0.01, 0.1],\n",
                "            'max_depth': [-1, 5, 10]\n",
                "        }\n",
                "    },\n",
                "    'CatBoost': {\n",
                "        'model': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False),\n",
                "        'params': {\n",
                "            'iterations': [50, 100],\n",
                "            'learning_rate': [0.01, 0.1],\n",
                "            'depth': [4, 6, 8]\n",
                "        }\n",
                "    }\n",
                "}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Train and Evaluate Models\n",
                "Evaluate models using RMSE, MAE, MAPE, and R²."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def mean_absolute_percentage_error(y_true, y_pred): \n",
                "    y_true, y_pred = np.array(y_true), np.array(y_pred)\n",
                "    return np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-10, y_true))) * 100\n",
                "\n",
                "results = []\n",
                "best_models = {}\n",
                "\n",
                "for name, mp in models.items():\n",
                "    print(f\"Training {name}...\")\n",
                "    start_time = time.time()\n",
                "    \n",
                "    if not mp['params']:\n",
                "        model = mp['model']\n",
                "        model.fit(X_train, y_train)\n",
                "        best_models[name] = model\n",
                "    else:\n",
                "        # Use RandomizedSearchCV to speed up execution\n",
                "        clf = RandomizedSearchCV(mp['model'], mp['params'], n_iter=3, cv=3, \n",
                "                                 scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)\n",
                "        clf.fit(X_train, y_train)\n",
                "        best_models[name] = clf.best_estimator_\n",
                "        model = clf.best_estimator_\n",
                "        \n",
                "    preds = model.predict(X_test)\n",
                "    \n",
                "    rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
                "    mae = mean_absolute_error(y_test, preds)\n",
                "    mape = mean_absolute_percentage_error(y_test, preds)\n",
                "    r2 = r2_score(y_test, preds)\n",
                "    \n",
                "    time_taken = time.time() - start_time\n",
                "    \n",
                "    results.append({\n",
                "        'Model': name,\n",
                "        'RMSE': rmse,\n",
                "        'MAE': mae,\n",
                "        'MAPE (%)': mape,\n",
                "        'R²': r2,\n",
                "        'Training Time (s)': round(time_taken, 2)\n",
                "    })\n",
                "    print(f\"{name} evaluated in {round(time_taken, 2)}s\")\n",
                "\n",
                "results_df = pd.DataFrame(results)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Comparison\n",
                "Comparing all trained models based on performance metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(results_df.sort_values(by='RMSE'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Select Best Model and Save\n",
                "We select the model with the lowest RMSE and save it using `joblib`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_model_name = results_df.sort_values(by='RMSE').iloc[0]['Model']\n",
                "best_model = best_models[best_model_name]\n",
                "\n",
                "print(f\"Best Model Selected: {best_model_name}\")\n",
                "\n",
                "MODEL_DIR = '../models/'\n",
                "os.makedirs(MODEL_DIR, exist_ok=True)\n",
                "joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))\n",
                "print(\"Best model saved successfully to models/best_model.pkl\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 1. Write the notebook
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/05_Model_Training.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Model Training notebook generated.")

# 2. Execute the model training pipeline
print("Running model training pipeline...")
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('data/processed/featured_air_quality.csv')
drop_cols = ['city', 'date', 'split', 'aqi_48', 'aqi_72', 'aqi_bucket']
features = [c for c in df.columns if c not in drop_cols and c != 'aqi_24']

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

X_train, y_train = train_df[features], train_df['aqi_24']
X_test, y_test = test_df[features], test_df['aqi_24']

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

models = {
    'Linear Regression': {'model': LinearRegression(), 'params': {}},
    'Random Forest': {'model': RandomForestRegressor(random_state=42, n_jobs=-1), 'params': {'n_estimators': [50], 'max_depth': [10]}},
    'XGBoost': {'model': XGBRegressor(random_state=42, objective='reg:squarederror'), 'params': {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [5]}},
    'LightGBM': {'model': LGBMRegressor(random_state=42, verbose=-1), 'params': {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [5]}},
    'CatBoost': {'model': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False), 'params': {'iterations': [50], 'learning_rate': [0.1], 'depth': [6]}}
}

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-10, y_true))) * 100

results = []
best_models = {}

for name, mp in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    if not mp['params']:
        model = mp['model']
        model.fit(X_train, y_train)
        best_models[name] = model
    else:
        clf = RandomizedSearchCV(mp['model'], mp['params'], n_iter=1, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        best_models[name] = clf.best_estimator_
        model = clf.best_estimator_
        
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    time_taken = time.time() - start_time
    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R²': r2})

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.sort_values(by='RMSE').to_string(index=False))

best_model_name = results_df.sort_values(by='RMSE').iloc[0]['Model']
best_model = best_models[best_model_name]
print(f"\nBest Model Selected: {best_model_name}")

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.pkl')
print("Best model saved successfully to models/best_model.pkl")
