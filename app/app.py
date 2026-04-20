"""
Snowstorm Flight Delay Predictor — Newark Liberty International Airport
Streamlit Dashboard

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE      = Path(__file__).parent
MODEL_DIR = BASE / ".." / "models"
DATA_DIR  = BASE / ".." / "data"

# ── Palette ────────────────────────────────────────────────────────────────
NAVY      = "#1a2744"
NAVY_MID  = "#2c4a7c"
STEEL     = "#8b9dc3"
GOLD      = "#c8a951"
BG        = "#f5f5f5"
WHITE     = "#ffffff"
TEXT_DARK = "#1a1a2e"
TEXT_MID  = "#4a4a6a"
BORDER    = "#dde1ec"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EWR Winter Flight Delay Predictor",
    page_icon="✈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ---------- global ---------- */
  html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: {TEXT_DARK};
    background-color: {BG};
  }}

  /* ---------- top bar / toolbar ---------- */
  header[data-testid="stHeader"] {{
    background-color: {NAVY} !important;
    border-bottom: 3px solid {GOLD};
  }}

  /* ---------- sidebar ---------- */
  [data-testid="stSidebar"] {{
    background-color: {NAVY} !important;
    border-right: 1px solid {NAVY_MID};
  }}
  [data-testid="stSidebar"] * {{
    color: #e8ecf5 !important;
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label {{
    color: {STEEL} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
  }}
  [data-testid="stSidebar"] hr {{
    border-color: {NAVY_MID} !important;
  }}
  /* slider track accent */
  [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div[role="slider"] {{
    background-color: {GOLD} !important;
    border-color: {GOLD} !important;
  }}

  /* ---------- main content ---------- */
  .block-container {{
    padding-top: 2rem;
    max-width: 1100px;
  }}

  /* ---------- page title ---------- */
  .page-title {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: {NAVY};
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.2;
  }}
  .page-subtitle {{
    font-size: 0.9rem;
    color: {TEXT_MID};
    margin-top: 0.25rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .title-rule {{
    border: none;
    border-top: 2px solid {GOLD};
    width: 60px;
    margin: 0.6rem 0 1.4rem 0;
  }}

  /* ---------- section headers ---------- */
  .section-header {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: {NAVY};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
  }}

  /* ---------- metric cards ---------- */
  .metric-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-top: 3px solid {NAVY_MID};
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }}
  .metric-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT_MID};
    margin-bottom: 0.2rem;
  }}
  .metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1.2;
  }}

  /* ---------- prediction card ---------- */
  .prediction-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-left: 5px solid {NAVY};
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
  }}
  .probability-number {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 3rem;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
    margin-bottom: 0.15rem;
  }}
  .prediction-label {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_MID};
    margin-bottom: 0.75rem;
  }}

  /* ---------- risk badges ---------- */
  .badge {{
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 2px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  .badge-high    {{ background: #fdf0ef; color: #9b2c2c; border: 1px solid #f5c6c6; }}
  .badge-moderate{{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}
  .badge-low     {{ background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}

  /* ---------- tabs ---------- */
  [data-baseweb="tab-list"] {{
    background-color: {WHITE};
    border-bottom: 2px solid {BORDER};
    gap: 0;
  }}
  [data-baseweb="tab"] {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {TEXT_MID} !important;
    padding: 0.6rem 1.25rem;
    border-bottom: 3px solid transparent;
    font-weight: 600;
  }}
  [aria-selected="true"][data-baseweb="tab"] {{
    color: {NAVY} !important;
    border-bottom: 3px solid {GOLD};
    background: transparent;
  }}

  /* ---------- sidebar section label ---------- */
  .sidebar-section {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {GOLD};
    font-weight: 700;
    margin: 1.25rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid {NAVY_MID};
  }}

  /* ---------- data table ---------- */
  [data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 4px;
  }}

  /* ---------- footer ---------- */
  .footer {{
    text-align: center;
    font-size: 0.72rem;
    color: {TEXT_MID};
    letter-spacing: 0.06em;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid {BORDER};
    margin-top: 2rem;
    text-transform: uppercase;
  }}

  /* hide streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Load models & data ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model    = joblib.load(MODEL_DIR / "xgb_model.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    try:
        rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
    except FileNotFoundError:
        rf_model = None
    return xgb_model, rf_model, feature_cols

@st.cache_data
def load_data():
    df      = pd.read_csv(DATA_DIR / "ewr_winter_clean.csv")
    results = pd.read_csv(DATA_DIR / "model_results.csv")
    carrier_delay_rate = df.groupby("CARRIER_NAME")["DELAYED"].mean()
    return df, results, carrier_delay_rate

try:
    xgb_model, rf_model, feature_columns = load_models()
    df, model_results, carrier_delay_rate = load_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models/data: {e}")
    st.info("Run Notebook 02 first to generate model artifacts.")

# ── Page title ─────────────────────────────────────────────────────────────
st.markdown("""
<p class="page-title">Newark Flight Delay Predictor</p>
<p class="page-subtitle">Newark Liberty International Airport &nbsp;·&nbsp; Winter Operations</p>
<hr class="title-rule">
""", unsafe_allow_html=True)

if models_loaded:

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        f'<div style="padding:1.25rem 0 0.5rem; font-family: Georgia, serif; '
        f'font-size:1.1rem; font-weight:700; color:#e8ecf5; '
        f'border-bottom:1px solid {NAVY_MID};">Flight Conditions</div>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown('<div class="sidebar-section">Weather</div>', unsafe_allow_html=True)
    snow = st.sidebar.slider("Snowfall (inches)",             0.0, 3.0, 0.5, 0.1)
    snwd = st.sidebar.slider("Snow Depth on Ground (inches)", 0.0, 2.0, 0.0, 0.1)
    tmax = st.sidebar.slider("Max Temperature (°F)",          15,  65,  32,  1)
    awnd = st.sidebar.slider("Wind Speed (mph)",              0.0, 30.0, 10.0, 0.5)
    prcp = st.sidebar.slider("Precipitation (inches)",        0.0, 2.0, 0.1, 0.05)

    st.sidebar.markdown('<div class="sidebar-section">Flight</div>', unsafe_allow_html=True)

    top_carriers = df["CARRIER_NAME"].value_counts().head(10).index.tolist()
    carrier = st.sidebar.selectbox("Airline", top_carriers)

    time_blocks  = sorted(df["DEP_TIME_BLK"].unique())
    _default_blk = "0800-0859" if "0800-0859" in time_blocks else time_blocks[0]
    dep_time     = st.sidebar.selectbox("Departure Block", time_blocks,
                                         index=time_blocks.index(_default_blk))

    day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
                 5: "Friday", 6: "Saturday", 7: "Sunday"}
    day = st.sidebar.selectbox("Day of Week", options=list(day_names.keys()),
                                format_func=lambda x: day_names[x])

    month_names = {1: "January", 2: "February", 12: "December"}
    month = st.sidebar.selectbox("Month", options=[12, 1, 2],
                                  format_func=lambda x: month_names[x])

    # ── Build prediction input ─────────────────────────────────────────────
    peak_blocks = ["0700-0759", "0800-0859", "0900-0959",
                   "1600-1659", "1700-1759", "1800-1859", "1900-1959"]

    input_data = pd.DataFrame([{
        "SNOW": snow, "SNWD": snwd, "TMAX": tmax, "AWND": awnd, "PRCP": prcp,
        "HEAVY_SNOW":     int(snow > 1),
        "BELOW_FREEZING": int(tmax < 32),
        "HIGH_WIND":      int(awnd > 15),
        "HAS_PRECIP":     int(prcp > 0),
        "SNOW_ON_GROUND": int(snwd > 0),
        "PEAK_HOUR":      int(dep_time in peak_blocks),
        "SEVERE_WEATHER": int((snow > 0) and (tmax < 35) and (awnd > 10)),
        "IS_WEEKEND":     int(day in [6, 7]),
        "CARRIER_DELAY_RATE": carrier_delay_rate.get(carrier, 0.26),
        "MONTH": month, "DAY_OF_WEEK": day,
        "DISTANCE_GROUP":              int(df["DISTANCE_GROUP"].median()),
        "PLANE_AGE":                   int(df["PLANE_AGE"].median()),
        "CONCURRENT_FLIGHTS":          int(df["CONCURRENT_FLIGHTS"].median()),
        "NUMBER_OF_SEATS":             int(df["NUMBER_OF_SEATS"].median()),
        "AIRPORT_FLIGHTS_MONTH":       int(df["AIRPORT_FLIGHTS_MONTH"].median()),
        "AIRLINE_AIRPORT_FLIGHTS_MONTH": int(
            df[df["CARRIER_NAME"] == carrier]["AIRLINE_AIRPORT_FLIGHTS_MONTH"].median()
            if len(df[df["CARRIER_NAME"] == carrier]) > 0
            else df["AIRLINE_AIRPORT_FLIGHTS_MONTH"].median()
        ),
    }])
    input_data = input_data[feature_columns]

    # ── Prediction ─────────────────────────────────────────────────────────
    delay_prob       = xgb_model.predict_proba(input_data)[0][1]
    delay_prediction = "Delayed" if delay_prob >= 0.5 else "On Time"

    if delay_prob >= 0.7:
        badge_html = '<span class="badge badge-high">High Risk</span>'
    elif delay_prob >= 0.4:
        badge_html = '<span class="badge badge-moderate">Moderate Risk</span>'
    else:
        badge_html = '<span class="badge badge-low">Low Risk</span>'

    # ── Main columns ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col1:
        st.markdown('<p class="section-header">Current Conditions</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Snowfall</div>
          <div class="metric-value">{snow}" </div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Temperature</div>
          <div class="metric-value">{tmax}°F</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Wind Speed</div>
          <div class="metric-value">{awnd} mph</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Delay Prediction</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prediction-card">
          <div class="probability-number">{delay_prob*100:.0f}%</div>
          <div class="prediction-label">Probability of departure delay &gt;15 min</div>
          {badge_html}
          <div style="margin-top:0.75rem; font-size:0.85rem; color:{TEXT_MID};">
            Model verdict: <strong style="color:{NAVY};">{delay_prediction}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown('<p class="section-header">Flight Details</p>', unsafe_allow_html=True)
        carrier_display = carrier.replace(" Co.", "").replace(" Inc.", "")
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Airline</div>
          <div class="metric-value" style="font-size:1.1rem;">{carrier_display}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Departure Block</div>
          <div class="metric-value" style="font-size:1.3rem;">{dep_time}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Day</div>
          <div class="metric-value" style="font-size:1.3rem;">{day_names[day]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">Top Predictive Factors</p>', unsafe_allow_html=True)

    importance = pd.DataFrame({
        "Feature":    feature_columns,
        "Importance": xgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    top10 = importance.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(BG)

    bar_colors = [GOLD if i == len(top10) - 1 else NAVY_MID for i in range(len(top10))]
    ax.barh(top10["Feature"], top10["Importance"], color=bar_colors, height=0.6)
    ax.set_xlabel("Importance Score", fontsize=9, color=TEXT_MID)
    ax.set_title("Feature Importance — XGBoost Model",
                 fontsize=10, color=NAVY, fontweight="bold", loc="left", pad=10)
    ax.tick_params(colors=TEXT_MID, labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.xaxis.set_tick_params(color=BORDER)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Comparison", "Snow Analysis", "What-If Scenarios", "About"
    ])

    # shared chart style helper
    def style_ax(ax_, title=""):
        ax_.set_facecolor(BG)
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.spines[["left", "bottom"]].set_color(BORDER)
        ax_.tick_params(colors=TEXT_MID, labelsize=8)
        ax_.set_title(title, fontsize=10, color=NAVY, fontweight="bold", loc="left", pad=8)

    with tab1:
        st.markdown('<p class="section-header" style="margin-top:1rem;">Model Performance Comparison</p>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.dataframe(
                model_results.style.format({
                    "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                    "Recall": "{:.3f}", "F1 Score": "{:.3f}", "ROC-AUC": "{:.3f}",
                }).highlight_max(subset=["ROC-AUC"], color="#e8f0fe"),
                use_container_width=True,
            )

        with col_b:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor(WHITE)
            metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            x       = np.arange(len(metrics))
            width   = 0.25
            palette = [NAVY, NAVY_MID, STEEL]

            for i, (_, row) in enumerate(model_results.iterrows()):
                vals = [row[m] for m in metrics]
                ax.bar(x + i * width, vals, width, label=row["Model"],
                       color=palette[i], alpha=0.9)

            ax.set_xticks(x + width)
            ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, frameon=False)
            style_ax(ax, "Performance by Metric")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.markdown('<p class="section-header" style="margin-top:1rem;">Weather Impact on Departure Delays</p>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor(WHITE)
            snow_bins = pd.cut(df["SNOW"], bins=[-0.1, 0, 0.5, 1, 2, 3],
                               labels=["None", '0-0.5"', '0.5-1"', '1-2"', '2+"'])
            snow_delay = df.groupby(snow_bins, observed=True)["DELAYED"].mean() * 100
            snow_delay.plot(kind="bar", ax=ax, color=NAVY_MID, edgecolor=WHITE, width=0.6)
            ax.set_ylabel("Flights Delayed (%)", fontsize=8, color=TEXT_MID)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
            style_ax(ax, "Delay Rate by Snowfall")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor(WHITE)
            temp_bins  = pd.cut(df["TMAX"], bins=5)
            temp_delay = df.groupby(temp_bins, observed=True)["DELAYED"].mean() * 100
            temp_delay.plot(kind="bar", ax=ax, color=STEEL, edgecolor=WHITE, width=0.6)
            ax.set_ylabel("Flights Delayed (%)", fontsize=8, color=TEXT_MID)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
            style_ax(ax, "Delay Rate by Temperature")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.markdown('<p class="section-header" style="margin-top:1rem;">Snowfall vs. Delay Probability</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.84rem; color:{TEXT_MID}; margin-bottom:1rem;">'
            "Delay probability curve as snowfall increases — holding all other inputs constant."
            "</p>", unsafe_allow_html=True
        )

        snow_range = np.arange(0, 3.1, 0.25)
        probs = []
        for s in snow_range:
            sim = input_data.copy()
            sim["SNOW"]          = s
            sim["HEAVY_SNOW"]    = int(s > 1)
            sim["SNOW_ON_GROUND"]= int(s > 0)
            sim["SEVERE_WEATHER"]= int((s > 0) and (tmax < 35) and (awnd > 10))
            probs.append(xgb_model.predict_proba(sim)[0][1] * 100)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(WHITE)
        ax.plot(snow_range, probs, "o-", color=NAVY, linewidth=2, markersize=5)
        ax.axhline(y=50,   color="#9b2c2c", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
        ax.axvline(x=snow, color=GOLD,      linestyle="--", linewidth=1, alpha=0.8, label=f'Current ({snow}")')
        ax.fill_between(snow_range, probs, alpha=0.07, color=NAVY)
        ax.set_xlabel("Snowfall (inches)", fontsize=9, color=TEXT_MID)
        ax.set_ylabel("Delay Probability (%)", fontsize=9, color=TEXT_MID)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, frameon=False)
        style_ax(ax, "How Snowfall Shifts Delay Risk")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown('<p class="section-header" style="margin-top:1rem;">About This Project</p>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
            <div style="font-size:0.88rem; color:{TEXT_DARK}; line-height:1.7;">
              <p>Predicts departure delays of more than 15 minutes for flights out of
              Newark Liberty International Airport (EWR) during December, January, and February.</p>
              <p><strong>Dataset:</strong> 31,121 winter flights paired with NOAA weather observations.</p>
              <p><strong>Best model:</strong> XGBoost — ROC-AUC 0.732, Recall 62.9%</p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style="font-size:0.88rem; color:{TEXT_DARK}; line-height:1.7;">
              <p><strong>Data sources</strong><br>
                 Flight data: Bureau of Transportation Statistics<br>
                 Weather data: NOAA NCEI</p>
              <p><strong>Key engineered features</strong><br>
                 Heavy snow flag, below-freezing indicator, high-wind flag,
                 severe-weather composite, peak-hour indicator,
                 airline historical delay rate</p>
              <p><strong>Stack:</strong> Python · scikit-learn · XGBoost · Streamlit</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">Newark Liberty International Airport &nbsp;·&nbsp; '
        'Winter Delay Predictor &nbsp;·&nbsp; XGBoost · Streamlit</div>',
        unsafe_allow_html=True
    )
