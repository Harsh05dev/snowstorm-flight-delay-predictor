# Snowstorm Flight Delay Predictor

Predicts flight delays at Newark Liberty International Airport (EWR) during winter weather conditions using machine learning.

## Live Demo

[Launch Dashboard on Streamlit Cloud](https://snowstorm-flight-delay-predictor-kms4suw7khir8j7gjkncjp.streamlit.app/)

## Overview

Trained on 31,121 winter flights (Dec–Feb) from EWR combined with NOAA weather data. The XGBoost model achieves ROC-AUC of **0.732**, identifying high-risk delay scenarios from snow, temperature, wind, and flight characteristics.

### Key Findings

| Condition | Delay Rate |
|-----------|-----------|
| No snow | 24.2% |
| Any snow | 37.6% |
| Heavy snow (>1") | 46.0% |
| High winds (>15 mph) | +9.8 pp |
| Snow on ground | +13.9 pp |

## Model Performance

| Model | ROC-AUC | Accuracy | Recall |
|-------|---------|----------|--------|
| **XGBoost** (best) | **0.732** | 70.2% | 62.9% |
| Random Forest | 0.711 | — | — |
| Logistic Regression | 0.664 | — | — |

Top predictors: airline historical delay rate → airline volume at EWR → heavy snow → snow on ground → high wind.

## Project Structure

```
snowstorm-flight-delay-predictor/
├── app/
│   └── app.py                    # Streamlit dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb # Data filtering, cleaning, EDA
│   └── 02_modeling.ipynb         # Feature engineering + model training
├── data/
│   ├── full_data_flightdelay.csv # Raw dataset (6.5M flights)
│   ├── ewr_winter_clean.csv      # Filtered Newark winter flights (31K)
│   └── model_results.csv         # Model comparison table
├── models/
│   ├── xgb_model.pkl             # Trained XGBoost model
│   ├── rf_model.pkl              # Trained Random Forest model
│   ├── feature_columns.pkl       # Feature order for inference
│   └── scaler.pkl                # StandardScaler
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
```

Run notebooks in order to generate data + models:

```bash
jupyter notebook notebooks/
# Run 01_data_exploration.ipynb → then 02_modeling.ipynb
```

Launch dashboard:

```bash
cd app
streamlit run app.py
```

## Feature Engineering

9 engineered features on top of 13 raw weather/flight inputs:

| Feature | Description |
|---------|-------------|
| `HEAVY_SNOW` | Snowfall > 1 inch |
| `BELOW_FREEZING` | TMAX < 32°F |
| `HIGH_WIND` | Wind > 15 mph |
| `HAS_PRECIP` | Any precipitation |
| `SNOW_ON_GROUND` | Snow depth > 0 |
| `PEAK_HOUR` | Departure 7–9 AM or 4–7 PM |
| `SEVERE_WEATHER` | Snow + cold + wind combo |
| `IS_WEEKEND` | Saturday or Sunday |
| `CARRIER_DELAY_RATE` | Historical airline delay rate |

## Dashboard Features

- **What-If Predictor** — adjust weather sliders + flight details → instant delay probability
- **Risk Classification** — HIGH (>70%), MODERATE (40–70%), LOW (<40%)
- **Model Comparison** — side-by-side metrics for all 3 models
- **EDA Visualizations** — delay rates by weather condition, airline, time of day

## Tech Stack

Python · scikit-learn · XGBoost · Streamlit · pandas · matplotlib · seaborn
