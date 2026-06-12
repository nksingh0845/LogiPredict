import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogiPredict – Shipment Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('pipe.pkl', 'rb') as f:
        return pickle.load(f)

pipe = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚚 LogiPredict")
st.subheader("Shipment Delay Prediction System")
st.markdown("Predict how many days a shipment will be delayed based on logistics and weather conditions.")
st.markdown("---")

# ── Sidebar – Business Insights ───────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Key Business Insights")
    st.markdown("""
    **From EDA on 1,500 shipments:**

    | Factor | Impact |
    |---|---|
    | 🌩️ Storm vs Clear | **+2.32 days** |
    | ⭐ Rating 5 vs 1 | **22% faster** |
    | 🚢 Sea transport | Slowest (**5.98d**) |
    | 🌧️ Monsoon season | Highest delay (**5.81d**) |
    | 🏥 Medical cargo | Most at-risk (**5.80d**) |
    | 📦 FedEx | Best carrier (**5.29d**) |

    ---
    **Model:** Gradient Boosting Regressor  
    **Dataset:** 1,500 shipment records  
    **Top predictor:** Distance (corr = 0.70)
    """)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### Enter Shipment Details")

col1, col2, col3 = st.columns(3)

with col1:
    distance_km    = st.slider("Distance (km)", 100, 2000, 500, step=50)
    vendor_rating  = st.selectbox("Vendor Rating", [1, 2, 3, 4, 5], index=2)
    weight_kg      = st.slider("Weight (kg)", 1.0, 500.0, 100.0, step=5.0)
    is_fragile     = st.selectbox("Fragile Cargo?", ["No", "Yes"])

with col2:
    vehicle_type   = st.selectbox("Vehicle Type",   ["Truck", "Train", "Ship", "Van", "Bike"])
    transport_mode = st.selectbox("Transport Mode", ["Road", "Rail", "Sea"])
    carrier        = st.selectbox("Carrier",        ["Delhivery", "Blue Dart", "FedEx", "DTDC", "Ecom Express"])
    cargo_type     = st.selectbox("Cargo Type",     ["Electronics", "Apparel", "Furniture", "Medical", "Perishable", "Industrial"])

with col3:
    weather        = st.selectbox("Weather",        ["Clear", "Fog", "Rain", "Storm"])
    season         = st.selectbox("Season",         ["Spring", "Summer", "Monsoon", "Winter"])
    is_weekend     = st.selectbox("Weekend Shipment?", ["No", "Yes"])
    is_peak_season = st.selectbox("Peak Season?",   ["No", "Yes"])

# ── Prediction ────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔍 Predict Delay", use_container_width=True):

    weather_map = {"Clear": 0, "Fog": 1, "Rain": 2, "Storm": 3}

    input_data = pd.DataFrame([{
        "distance_km"     : distance_km,
        "vehicle_type"    : vehicle_type,
        "transport_mode"  : transport_mode,
        "carrier"         : carrier,
        "vendor_rating"   : vendor_rating,
        "season"          : season,
        "is_weekend"      : 1 if is_weekend == "Yes" else 0,
        "is_peak_season"  : 1 if is_peak_season == "Yes" else 0,
        "weight_kg"       : weight_kg,
        "cargo_type"      : cargo_type,
        "is_fragile"      : 1 if is_fragile == "Yes" else 0,
        "weather_severity": weather_map[weather]
    }])

    predicted_delay = pipe.predict(input_data)[0]

    # ── Result display ────────────────────────────────────────────────────────
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("📦 Predicted Delay", f"{predicted_delay:.2f} days")

    with res_col2:
        # Risk level
        if predicted_delay < 4:
            risk = "🟢 Low Risk"
        elif predicted_delay < 6.5:
            risk = "🟡 Medium Risk"
        else:
            risk = "🔴 High Risk"
        st.metric("Risk Level", risk)

    with res_col3:
        avg_delay = 5.50
        diff = predicted_delay - avg_delay
        label = f"+{diff:.2f} days above avg" if diff > 0 else f"{diff:.2f} days below avg"
        st.metric("vs Dataset Average (5.50d)", label)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("### 💡 Recommendations")
    recs = []

    if weather in ["Rain", "Storm"]:
        recs.append("⚠️ Adverse weather detected — consider rescheduling or pre-alerting the customer.")
    if vendor_rating <= 2:
        recs.append("⚠️ Low vendor rating — consider switching to a higher-rated vendor to reduce delay by ~1.4 days.")
    if transport_mode == "Sea":
        recs.append("🚢 Sea transport is the slowest mode — Rail is 0.72 days faster on average.")
    if season == "Monsoon":
        recs.append("🌧️ Monsoon season has the highest average delay — add buffer time to delivery estimates.")
    if cargo_type == "Medical":
        recs.append("🏥 Medical cargo has the highest delay risk — consider priority routing.")
    if is_peak_season == "Yes":
        recs.append("📈 Peak season adds ~0.5 days on average — plan capacity accordingly.")
    if not recs:
        recs.append("✅ Shipment conditions look good. No major delay risk factors detected.")

    for rec in recs:
        st.info(rec)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("LogiPredict | Built with Streamlit | Model: Gradient Boosting Regressor | Dataset: 1,500 shipment records")
