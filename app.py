import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor · IDC 409",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.app-header {
    background: linear-gradient(135deg, #0f2d5a 0%, #1a5276 55%, #2471a3 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.app-header h1 { color: white; font-size: 2rem; font-weight: 800;
                  margin: 0 0 0.3rem 0; letter-spacing: -0.5px; }
.app-header .sub { color: #a8c8e8; font-size: 0.9rem; margin: 0; }
.app-header .badges { margin-top: 0.8rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge { background: rgba(255,255,255,0.15); color: white; padding: 0.2rem 0.75rem;
         border-radius: 20px; font-size: 0.72rem; font-weight: 600;
         border: 1px solid rgba(255,255,255,0.25); }

.price-card {
    background: linear-gradient(135deg, #eaf4fd 0%, #d6eaf8 100%);
    border: 2px solid #2471a3; border-radius: 16px;
    padding: 2rem; text-align: center;
}
.price-label { color: #555; font-size: 0.85rem; font-weight: 700;
               text-transform: uppercase; letter-spacing: 1.2px; margin: 0; }
.price-value { color: #0f2d5a; font-size: 3.6rem; font-weight: 800;
               margin: 0.4rem 0; letter-spacing: -1px; }
.price-range { color: #666; font-size: 0.85rem; margin: 0; }

.verdict { display: inline-block; padding: 0.25rem 1rem; border-radius: 20px;
           font-weight: 700; font-size: 0.85rem; margin-top: 0.6rem; }
.vb { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }
.vm { background:#e3f2fd; color:#1565c0; border:1px solid #90caf9; }
.va { background:#fff3e0; color:#e65100; border:1px solid #ffcc80; }
.vp { background:#fce4ec; color:#b71c1c; border:1px solid #f48fb1; }

.stat-card { background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
             border-left: 5px solid #2471a3;
             box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.stat-lbl { color: #888; font-size: 0.72rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.8px; margin: 0 0 0.3rem 0; }
.stat-val { color: #0f2d5a; font-size: 1.5rem; font-weight: 800; margin: 0; }
.stat-sub { color: #999; font-size: 0.74rem; margin: 0.2rem 0 0 0; }

.sec-head { background: #0f2d5a; color: white; padding: 0.35rem 0.8rem;
            border-radius: 6px; font-size: 0.72rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.8px;
            margin: 1rem 0 0.5rem 0; }

.welcome { background: #f8fafe; border: 2px dashed #b8d4f0;
           border-radius: 16px; padding: 3rem; text-align: center; }
.welcome h3 { color: #0f2d5a; font-size: 1.5rem; margin: 0 0 0.5rem 0; }
.welcome p  { color: #777; margin: 0; }

.info-note { background: #fff8e1; border-left: 4px solid #ffc107;
             padding: 0.65rem 1rem; border-radius: 0 8px 8px 0;
             font-size: 0.82rem; color: #5d4037; margin: 0.8rem 0; }

.footer { text-align: center; color: #aaa; font-size: 0.78rem;
          margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ──────────────────────────────────────────────────
# NEW — works both locally and on Streamlit Cloud
PROJECT_ROOT = Path(__file__).parent

@st.cache_resource
def load_model():
    m    = joblib.load(PROJECT_ROOT / "models" / "xgboost_house_price.pkl")
    cols = joblib.load(PROJECT_ROOT / "models" / "model_columns.pkl")
    with open(PROJECT_ROOT / "models" / "numeric_medians.json") as f:
        meds = json.load(f)
    return m, cols, meds

@st.cache_data
def load_prices():
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "ames_housing" / "train.csv")
    return df["SalePrice"].values

model, MODEL_COLUMNS, num_medians = load_model()
all_prices = load_prices()

CAT_COLS = [
    'MSZoning','Street','Alley','LotShape','LandContour','Utilities',
    'LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
    'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
    'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
    'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',
    'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature',
    'SaleType','SaleCondition'
]

# ── Prediction function ─────────────────────────────────────────────
def predict_price(user_inputs):
    row = pd.DataFrame(0.0, index=[0], columns=MODEL_COLUMNS)
    for col, val in num_medians.items():
        if col in MODEL_COLUMNS:
            row[col] = float(val)
    for col, val in user_inputs.items():
        if col in MODEL_COLUMNS:
            row[col] = float(val)
    for col, val in user_inputs.items():
        if col in CAT_COLS:
            enc = f"{col}_{val}"
            if enc in MODEL_COLUMNS:
                row[enc] = 1.0
    price = float(model.predict(row)[0])
    return {"price": round(price), "low": round(price * 0.90),
            "high": round(price * 1.10)}

# ── Gauge chart ──────────────────────────────────────────────────────
def make_gauge(predicted, low, high):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    zones = [
        (34900,  130000, "#66bb6a", "Budget\n< $130k"),
        (130000, 200000, "#42a5f5", "Mid-Range\n$130–200k"),
        (200000, 300000, "#ffa726", "Above Avg\n$200–300k"),
        (300000, 755000, "#ef5350", "Premium\n> $300k"),
    ]
    for start, end, color, label in zones:
        ax.barh(0, end - start, left=start, height=0.4,
                color=color, alpha=0.75, linewidth=0)
        ax.text((start + end) / 2, -0.33, label,
                ha="center", va="top", fontsize=7,
                color=color, fontweight="bold", multialignment="center")

    # Confidence band
    ax.barh(0, high - low, left=low, height=0.54,
            color="#0f2d5a", alpha=0.18, linewidth=0)

    # Predicted marker
    ax.plot(predicted, 0, "v", markersize=14,
            color="#0f2d5a", zorder=5, clip_on=False)
    ax.text(predicted, 0.34, f"${predicted:,}",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#0f2d5a")

    ax.set_xlim(34900, 755000)
    ax.set_ylim(-0.68, 0.68)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.tick_params(axis="x", labelsize=8, colors="#666")
    ax.set_title("Where this house sits in the Ames market",
                 fontsize=9, color="#555", pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout(pad=0.4)
    return fig

# ── Verdict ──────────────────────────────────────────────────────────
def verdict(price):
    if price < 130000:
        return "Budget", "vb"
    elif price < 200000:
        return "Mid-Range", "vm"
    elif price < 300000:
        return "Above Average", "va"
    else:
        return "Premium", "vp"

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>🏠 House Price Predictor</h1>
  <p class="sub">IDC 409 &nbsp;·&nbsp; Group 18 &nbsp;·&nbsp; IISER Mohali</p>
  <div class="badges">
    <span class="badge">XGBoost</span>
    <span class="badge">R² = 0.918</span>
    <span class="badge">MAE = $15,331</span>
    <span class="badge">Ames Housing Dataset</span>
    <span class="badge">1,460 training houses</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — ALL INPUTS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ House Parameters")
    st.caption("Adjust sliders and dropdowns, then click **Predict**.")

    # ── Size & Space ────────────────────────────────────────────────
    st.markdown('<div class="sec-head">📐 Size & Space</div>',
                unsafe_allow_html=True)

    gr_liv_area = st.slider(
        "Living Area (sq ft)", 300, 5000, 1500, 50,
        help="Above-ground living area"
    )
    first_flr_sf = st.slider("First Floor Area (sq ft)", 300, 4000, 1000, 50)
    garage_cars  = st.select_slider(
        "Garage Capacity",
        options=[0, 1, 2, 3, 4], value=2,
        format_func=lambda x: f"{x} car{'s' if x != 1 else ''}"
    )

    # ── Quality & Condition ─────────────────────────────────────────
    st.markdown('<div class="sec-head">⭐ Quality & Condition</div>',
                unsafe_allow_html=True)

    overall_qual = st.slider("Overall Quality  (1 = Poor → 10 = Excellent)",
                              1, 10, 6)
    qual_labels = {
        1:"Very Poor", 2:"Poor", 3:"Fair", 4:"Below Average",
        5:"Average",   6:"Above Average", 7:"Good",
        8:"Very Good", 9:"Excellent", 10:"Very Excellent"
    }
    st.caption(f"Selected: **{qual_labels[overall_qual]}**")

    kitchen_qual = st.select_slider(
        "Kitchen Quality",
        options=["Fa", "TA", "Gd", "Ex"], value="Gd",
        format_func=lambda x: {
            "Fa": "Fair", "TA": "Typical",
            "Gd": "Good", "Ex": "Excellent"
        }[x]
    )

    # ── Age & History ───────────────────────────────────────────────
    st.markdown('<div class="sec-head">🏗️ Age & History</div>',
                unsafe_allow_html=True)

    year_built = st.slider("Year Built", 1872, 2010, 1990)
    year_remod = st.slider(
        "Year Last Remodelled", 1950, 2010,
        value=max(1950, year_built),
        help="Same as Year Built if never remodelled"
    )

    # ── Basement ────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">🏚️ Basement</div>',
                unsafe_allow_html=True)

    bsmt_qual = st.selectbox(
        "Basement Quality",
        options=["Ex", "Gd", "TA", "Fa", "Unknown"],
        index=1,
        format_func=lambda x: {
            "Ex": "Excellent", "Gd": "Good", "TA": "Typical",
            "Fa": "Fair",      "Unknown": "No Basement"
        }[x]
    )
    # bsmt_qual is now defined — safe to use below
    if bsmt_qual == "Unknown":
        total_bsmt_sf = 0
        st.info("No basement → area set to 0 sq ft")
    else:
        total_bsmt_sf = st.slider("Basement Area (sq ft)", 0, 3000, 800, 50)

    # ── Other ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">🚿 Other</div>',
                unsafe_allow_html=True)

    full_bath = st.select_slider(
        "Full Bathrooms (above ground)",
        options=[1, 2, 3, 4], value=2
    )

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Price",
                             type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════
if not predict_btn:
    # Welcome screen
    st.markdown("""
    <div class="welcome">
      <h3>← Fill in the house details</h3>
      <p>Adjust the parameters in the sidebar and click <b>Predict Price</b>
         to get an estimated sale price with a confidence range.</p>
      <br>
      <p style="font-size:0.78rem;color:#bbb;">
        Ames Housing Dataset · Iowa, USA · 2006–2010 · Prices in USD
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### How does this model compare to doing nothing?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="stat-card">
            <p class="stat-lbl">Baseline Error (mean predictor)</p>
            <p class="stat-val">$62,576</p>
            <p class="stat-sub">Predicting avg price every time</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <p class="stat-lbl">XGBoost Error (our model)</p>
            <p class="stat-val">$15,331</p>
            <p class="stat-sub">75% lower than the baseline</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <p class="stat-lbl">R² Score</p>
            <p class="stat-val">0.918</p>
            <p class="stat-sub">Model explains 91.8% of price variation</p>
        </div>""", unsafe_allow_html=True)

else:
    # ── Run prediction ───────────────────────────────────────────────
    with st.spinner("Running XGBoost model..."):
        result = predict_price({
            "OverallQual"  : overall_qual,
            "GrLivArea"    : gr_liv_area,
            "YearBuilt"    : year_built,
            "GarageCars"   : garage_cars,
            "1stFlrSF"     : first_flr_sf,
            "TotalBsmtSF"  : total_bsmt_sf,
            "YearRemodAdd" : year_remod,
            "FullBath"     : full_bath,
            "KitchenQual"  : kitchen_qual,
            "BsmtQual"     : bsmt_qual,
        })

    price      = result["price"]
    low        = result["low"]
    high       = result["high"]
    pct        = round((all_prices < price).mean() * 100, 1)
    vlab, vcls = verdict(price)
    pct_diff   = round((price - np.mean(all_prices)) / np.mean(all_prices) * 100, 1)
    direction  = "above" if pct_diff >= 0 else "below"

    # ── Price card ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="price-card">
      <p class="price-label">Predicted Sale Price</p>
      <p class="price-value">${price:,}</p>
      <p class="price-range">
        Confidence range &nbsp; <b>${low:,}</b> – <b>${high:,}</b> &nbsp;(±10%)
      </p>
      <div style="margin-top:0.7rem;">
        <span class="verdict {vcls}">{vlab}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge ────────────────────────────────────────────────────────
    fig = make_gauge(price, low, high)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Four stat cards ──────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="stat-card">
            <p class="stat-lbl">Market Percentile</p>
            <p class="stat-val">{pct}%</p>
            <p class="stat-sub">Pricier than {pct}% of Ames houses</p>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="stat-card">
            <p class="stat-lbl">vs. Market Average</p>
            <p class="stat-val">{abs(pct_diff)}%</p>
            <p class="stat-sub">{direction} the avg (${int(np.mean(all_prices)):,})</p>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="stat-card">
            <p class="stat-lbl">Confidence Width</p>
            <p class="stat-val">${(high - low)//1000}k</p>
            <p class="stat-sub">Range of ±10% interval</p>
        </div>""", unsafe_allow_html=True)
    with s4:
        ppsf = round(price / max(gr_liv_area, 1))
        st.markdown(f"""<div class="stat-card">
            <p class="stat-lbl">Est. $/sq ft</p>
            <p class="stat-val">${ppsf}</p>
            <p class="stat-sub">Based on living area only</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input summary table ──────────────────────────────────────────
    st.markdown("#### Summary of inputs used")
    left, right = st.columns(2)

    kq_map  = {"Fa":"Fair","TA":"Typical","Gd":"Good","Ex":"Excellent"}
    bq_map  = {"Ex":"Excellent","Gd":"Good","TA":"Typical",
               "Fa":"Fair","Unknown":"No Basement"}
    bsf_str = f"{total_bsmt_sf:,} sq ft" if bsmt_qual != "Unknown" \
              else "No basement"

    with left:
        st.markdown(f"""
| Feature | Value |
|---|---|
| Overall Quality | {overall_qual}/10 — {qual_labels[overall_qual]} |
| Living Area | {gr_liv_area:,} sq ft |
| First Floor Area | {first_flr_sf:,} sq ft |
| Kitchen Quality | {kq_map[kitchen_qual]} |
""")
    with right:
        st.markdown(f"""
| Feature | Value |
|---|---|
| Year Built | {year_built} |
| Year Remodelled | {year_remod} |
| Garage | {garage_cars} car{'s' if garage_cars != 1 else ''} |
| Basement | {bq_map[bsmt_qual]} · {bsf_str} |
""")

    # ── Dataset note ─────────────────────────────────────────────────
    st.markdown("""
    <div class="info-note">
      ⚠️ <b>Context note:</b> Predictions are based on Ames, Iowa USA housing
      sales from 2006–2010. All prices are in 2006–2010 USD and do <b>not</b>
      represent Indian real estate or current market values.
    </div>
    """, unsafe_allow_html=True)

    # ── Model details (expandable) ───────────────────────────────────
    with st.expander("📊 How was this predicted? — Model & pipeline details"):
        st.markdown("""
**Algorithm:** XGBoost (eXtreme Gradient Boosting — 300 sequential trees)

**Training data:** 1,168 Ames, Iowa houses (80% of 1,460 total)

**Top price drivers (from SHAP analysis):**

| Rank | Feature | What it means |
|---|---|---|
| 1 | `OverallQual` | Overall material & finish quality (1–10) |
| 2 | `GrLivArea` | Above-ground living area (sq ft) |
| 3 | `GarageCars` | Garage capacity |
| 4 | `TotalBsmtSF` | Total basement area |
| 5 | `YearBuilt` | Year originally built |

**Full model comparison:**

| Model | R² | MAE |
|---|---|---|
| Baseline (mean predictor) | -0.001 | $62,576 |
| Ridge Regression | 0.818 | $19,332 |
| Lasso Regression | 0.736 | $19,346 |
| Decision Tree | 0.755 | $27,331 |
| Random Forest | 0.892 | $17,573 |
| **XGBoost ★ (this app)** | **0.918** | **$15,331** |

**What R² = 0.918 means:**
The model explains 91.8% of house-to-house price variation.
The remaining 8.2% is driven by factors not in the dataset
(exact street, interior finishes, buyer motivation, market timing).
        """)

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  IDC 409 · Data Science & Machine Learning · Project 15: House Price Prediction<br>
  Group 18 · IISER Mohali · 2026 &nbsp;|&nbsp;
  XGBoost · Ames Housing Dataset (Iowa, USA, 2006–2010)
</div>
""", unsafe_allow_html=True)
