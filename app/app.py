"""
❄️ Snowstorm Flight Delay Predictor — Newark Liberty International Airport
Streamlit Dashboard (Day 3)

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="❄️ Newark Flight Delay Predictor",
    page_icon="❄️",
    layout="wide"
)

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_models():
    """Load saved models (cached so it only loads once)."""
    xgb_model = joblib.load('../models/xgb_model.pkl')
    rf_model = joblib.load('../models/rf_model.pkl')
    feature_cols = joblib.load('../models/feature_columns.pkl')
    return xgb_model, rf_model, feature_cols

@st.cache_data
def load_data():
    """Load the clean dataset for visualizations."""
    df = pd.read_csv('../data/ewr_winter_clean.csv')
    results = pd.read_csv('../data/model_results.csv')
    carrier_delay_rate = df.groupby('CARRIER_NAME')['DELAYED'].mean()
    return df, results, carrier_delay_rate

try:
    xgb_model, rf_model, feature_columns = load_models()
    df, model_results, carrier_delay_rate = load_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Error loading models/data: {e}")
    st.info("Make sure you ran Notebook 02 first and saved the models!")

# ============================================================
# HEADER
# ============================================================
st.title("❄️ Snowstorm Flight Delay Predictor")
st.markdown("### Newark Liberty International Airport (EWR)")
st.markdown("*Predict whether your flight will be delayed during winter weather conditions.*")
st.markdown("---")

if models_loaded:

    # ============================================================
    # SIDEBAR — WEATHER INPUT
    # ============================================================
    st.sidebar.header("🌨️ Enter Weather Conditions")
    st.sidebar.markdown("Adjust the sliders to simulate different weather scenarios.")

    snow = st.sidebar.slider("❄️ Snowfall (inches)", 0.0, 3.0, 0.5, 0.1)
    snwd = st.sidebar.slider("📏 Snow Depth on Ground (inches)", 0.0, 2.0, 0.0, 0.1)
    tmax = st.sidebar.slider("🌡️ Max Temperature (°F)", 15, 65, 32, 1)
    awnd = st.sidebar.slider("🌬️ Wind Speed (mph)", 0.0, 30.0, 10.0, 0.5)
    prcp = st.sidebar.slider("🌧️ Precipitation (inches)", 0.0, 2.0, 0.1, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.header("✈️ Flight Details")

    # Carrier selection
    top_carriers = df['CARRIER_NAME'].value_counts().head(10).index.tolist()
    carrier = st.sidebar.selectbox("Airline", top_carriers)

    # Time block
    time_blocks = sorted(df['DEP_TIME_BLK'].unique())
    _default_blk = '0800-0859' if '0800-0859' in time_blocks else time_blocks[0]
    dep_time = st.sidebar.selectbox("Departure Time", time_blocks, index=time_blocks.index(_default_blk))

    # Day of week
    day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday',
                 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    day = st.sidebar.selectbox("Day of Week", options=list(day_names.keys()),
                                format_func=lambda x: day_names[x])

    month_names = {1: 'January', 2: 'February', 12: 'December'}
    month = st.sidebar.selectbox("Month", options=[12, 1, 2],
                                  format_func=lambda x: month_names[x])

    # ============================================================
    # BUILD PREDICTION INPUT
    # ============================================================
    # Engineer features to match what the model expects
    peak_blocks = ['0700-0759', '0800-0859', '0900-0959',
                   '1600-1659', '1700-1759', '1800-1859', '1900-1959']

    # Get median values for columns we don't have sliders for
    input_data = pd.DataFrame([{
        'SNOW': snow,
        'SNWD': snwd,
        'TMAX': tmax,
        'AWND': awnd,
        'PRCP': prcp,
        'HEAVY_SNOW': int(snow > 1),
        'BELOW_FREEZING': int(tmax < 32),
        'HIGH_WIND': int(awnd > 15),
        'HAS_PRECIP': int(prcp > 0),
        'SNOW_ON_GROUND': int(snwd > 0),
        'PEAK_HOUR': int(dep_time in peak_blocks),
        'SEVERE_WEATHER': int((snow > 0) and (tmax < 35) and (awnd > 10)),
        'IS_WEEKEND': int(day in [6, 7]),
        'CARRIER_DELAY_RATE': carrier_delay_rate.get(carrier, 0.26),
        'MONTH': month,
        'DAY_OF_WEEK': day,
        'DISTANCE_GROUP': int(df['DISTANCE_GROUP'].median()),
        'PLANE_AGE': int(df['PLANE_AGE'].median()),
        'CONCURRENT_FLIGHTS': int(df['CONCURRENT_FLIGHTS'].median()),
        'NUMBER_OF_SEATS': int(df['NUMBER_OF_SEATS'].median()),
        'AIRPORT_FLIGHTS_MONTH': int(df['AIRPORT_FLIGHTS_MONTH'].median()),
        'AIRLINE_AIRPORT_FLIGHTS_MONTH': int(df[df['CARRIER_NAME'] == carrier]['AIRLINE_AIRPORT_FLIGHTS_MONTH'].median())
            if len(df[df['CARRIER_NAME'] == carrier]) > 0
            else int(df['AIRLINE_AIRPORT_FLIGHTS_MONTH'].median()),
    }])

    # Ensure column order matches training
    input_data = input_data[feature_columns]

    # ============================================================
    # MAKE PREDICTION
    # ============================================================
    delay_prob = xgb_model.predict_proba(input_data)[0][1]
    delay_prediction = "DELAYED" if delay_prob >= 0.5 else "ON TIME"

    # ============================================================
    # MAIN DISPLAY
    # ============================================================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌨️ Current Conditions")
        st.metric("Snowfall", f"{snow} in")
        st.metric("Temperature", f"{tmax}°F")
        st.metric("Wind Speed", f"{awnd} mph")

    with col2:
        st.markdown("### 🎯 Prediction")
        # Color the probability
        if delay_prob >= 0.7:
            color = "🔴"
            risk = "HIGH RISK"
        elif delay_prob >= 0.4:
            color = "🟡"
            risk = "MODERATE RISK"
        else:
            color = "🟢"
            risk = "LOW RISK"

        st.markdown(f"## {color} {delay_prob*100:.0f}% Delay Probability")
        st.markdown(f"**{risk}** of departure delay (>15 min)")
        st.markdown(f"Prediction: **{delay_prediction}**")

    with col3:
        st.markdown("### ✈️ Flight Info")
        st.metric("Airline", carrier.replace(" Co.", "").replace(" Inc.", ""))
        st.metric("Departure", dep_time)
        st.metric("Day", day_names[day])

    st.markdown("---")

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    st.markdown("### 🌟 Top Contributing Factors")

    importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Show top 10 as a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    top10 = importance.head(10).iloc[::-1]  # Reverse for horizontal bar
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top10)))
    ax.barh(top10['Feature'], top10['Importance'], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('What Drives Flight Delays?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ============================================================
    # TABS FOR MORE DETAILS
    # ============================================================
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📈 Snow Analysis",
                                        "🔮 What-If Scenarios", "ℹ️ About"])

    with tab1:
        st.markdown("### 🏆 Model Comparison")
        col1, col2 = st.columns(2)

        with col1:
            # Results table
            st.dataframe(
                model_results.style.format({
                    'Accuracy': '{:.3f}', 'Precision': '{:.3f}',
                    'Recall': '{:.3f}', 'F1 Score': '{:.3f}', 'ROC-AUC': '{:.3f}'
                }).highlight_max(subset=['ROC-AUC'], color='#90EE90'),
                use_container_width=True
            )

        with col2:
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
            x = np.arange(len(metrics))
            width = 0.25
            colors_list = ['#2196F3', '#4CAF50', '#FF5722']

            for i, (_, row) in enumerate(model_results.iterrows()):
                vals = [row[m] for m in metrics]
                ax.bar(x + i * width, vals, width, label=row['Model'], color=colors_list[i])

            ax.set_xticks(x + width)
            ax.set_xticklabels(metrics, rotation=30, ha='right')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=9)
            ax.set_title('Model Performance Comparison')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown("### ❄️ How Snow Affects Delays at Newark")
        col1, col2 = st.columns(2)

        with col1:
            # Snow vs delay rate
            fig, ax = plt.subplots(figsize=(8, 5))
            snow_bins = pd.cut(df['SNOW'], bins=[-0.1, 0, 0.5, 1, 2, 3],
                               labels=['None', '0-0.5"', '0.5-1"', '1-2"', '2"+'])
            snow_delay = df.groupby(snow_bins, observed=True)['DELAYED'].mean() * 100
            snow_delay.plot(kind='bar', ax=ax, color='#42A5F5', edgecolor='white')
            ax.set_ylabel('% Flights Delayed')
            ax.set_title('Delay Rate by Snowfall Amount')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            # Temperature vs delay rate
            fig, ax = plt.subplots(figsize=(8, 5))
            temp_bins = pd.cut(df['TMAX'], bins=5)
            temp_delay = df.groupby(temp_bins, observed=True)['DELAYED'].mean() * 100
            temp_delay.plot(kind='bar', ax=ax, color='#EF5350', edgecolor='white')
            ax.set_ylabel('% Flights Delayed')
            ax.set_title('Delay Rate by Temperature')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.markdown("### 🔮 What-If Scenario Explorer")
        st.markdown("See how delay probability changes as snowfall increases:")

        # Simulate increasing snowfall
        snow_range = np.arange(0, 3.1, 0.25)
        probs = []
        for s in snow_range:
            sim_input = input_data.copy()
            sim_input['SNOW'] = s
            sim_input['HEAVY_SNOW'] = int(s > 1)
            sim_input['SNOW_ON_GROUND'] = int(s > 0)
            sim_input['SEVERE_WEATHER'] = int((s > 0) and (tmax < 35) and (awnd > 10))
            prob = xgb_model.predict_proba(sim_input)[0][1]
            probs.append(prob * 100)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(snow_range, probs, 'o-', color='#1565C0', linewidth=2, markersize=6)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.axvline(x=snow, color='green', linestyle='--', alpha=0.5, label=f'Current ({snow}")')
        ax.fill_between(snow_range, probs, alpha=0.1, color='#1565C0')
        ax.set_xlabel('Snowfall (inches)', fontsize=12)
        ax.set_ylabel('Delay Probability (%)', fontsize=12)
        ax.set_title('How Does Snowfall Affect Your Delay Risk?', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown("### ℹ️ About This Project")
        st.markdown("""
        **❄️ Snowstorm Flight Delay Predictor**

        This machine learning model predicts whether a flight departing from
        Newark Liberty International Airport (EWR) will be delayed by more than
        15 minutes during winter weather conditions.

        **Data Sources:**
        - ✈️ Flight data: Bureau of Transportation Statistics (BTS)
        - 🌨️ Weather data: NOAA National Centers for Environmental Information

        **Models Used:**
        - Logistic Regression (baseline)
        - Random Forest (200 trees)
        - XGBoost (gradient boosting) ← Best performer

        **Features Engineered:**
        - Heavy snow flag (>1 inch)
        - Below freezing indicator
        - High wind flag (>15 mph)
        - Severe weather combo (snow + cold + wind)
        - Peak hour indicator
        - Airline historical delay rate
        - And more...

        **Dataset:** 31,121 winter flights at EWR (Dec, Jan, Feb)

        **Built with:** Python, scikit-learn, XGBoost, Streamlit
        """)

    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "❄️ Snowstorm Flight Delay Predictor | Newark Liberty International Airport | "
        "Built with Python, XGBoost & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )
