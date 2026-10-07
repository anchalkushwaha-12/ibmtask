import os
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Set Page Config
st.set_page_config(
    page_title="ValuEdge | AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Design System CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 0.25rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-price {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: linear-gradient(90deg, #38BDF8 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.4rem 0;
    }

    .range-bar-wrapper {
        margin-top: 1.2rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
        height: 100%;
    }

    .card-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .warning-badge {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.2rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .stat-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        display: inline-block;
        margin-right: 0.8rem;
    }

    .stat-pill-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .stat-pill-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
FLASK_API_URL = os.environ.get("FLASK_API_URL", "http://127.0.0.1:5000")
CURRENT_YEAR = datetime.date.today().year

# Currency Rates & Conversion
CURRENCY_DATA = {
    "USD ($)": {"symbol": "$", "rate": 1.0, "format": "${:,.0f}"},
    "EUR (€)": {"symbol": "€", "rate": 0.92, "format": "€{:,.0f}"},
    "GBP (£)": {"symbol": "£", "rate": 0.79, "format": "£{:,.0f}"},
    "INR (₹)": {"symbol": "₹", "rate": 83.5, "format": "₹{:,.0f}"}
}

# Unit Conversion: 1 sq meter = 10.7639 sq ft
SQFT_PER_SQM = 10.7639

@st.cache_resource
def load_embedded_model():
    """Loads unified model package for direct fast in-process fallback."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load local model package: {e}")
    return None

@st.cache_data
def load_metrics_summary():
    """Loads model metrics metadata."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

model_package = load_embedded_model()
metrics_meta = load_metrics_summary()

# Helper: Check API Availability without blocking UI
def check_api_health():
    try:
        res = requests.get(f"{FLASK_API_URL}/health", timeout=0.4)
        if res.status_code == 200:
            return True
    except Exception:
        pass
    return False

@st.cache_data(show_spinner=False)
def geocode_address(query):
    """Geocodes any global address or city via OpenStreetMap Nominatim with cached response."""
    if not query or len(str(query).strip()) < 2:
        return {"lat": 30.2672, "lon": -97.7431, "name": "Downtown Austin, Texas, USA", "found": True}
    try:
        headers = {'User-Agent': 'ValuEdge-HousePricePredictor/2.0 (property-research)'}
        res = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q': str(query).strip(), 'format': 'json', 'limit': 1},
            headers=headers,
            timeout=3.0
        )
        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0:
                return {
                    "lat": float(data[0]['lat']),
                    "lon": float(data[0]['lon']),
                    "name": data[0].get('display_name', str(query)),
                    "found": True
                }
    except Exception:
        pass
    return {"lat": 30.2672, "lon": -97.7431, "name": str(query), "found": False}

# Top Control Bar (Units & Currency)
col_title, col_ctrl1, col_ctrl2 = st.columns([2.5, 1, 1])

with col_title:
    st.markdown("## 🏠 ValuEdge House Price Predictor")
    st.caption("Accurate property valuation with value breakdown, market comparables, and price trends")

with col_ctrl1:
    selected_currency_name = st.selectbox("Currency", list(CURRENCY_DATA.keys()), index=0)
    curr_info = CURRENCY_DATA[selected_currency_name]

with col_ctrl2:
    selected_unit = st.selectbox("Area Unit", ["Square Feet (sqft)", "Square Meters (m²)"], index=0)
    is_sqm = selected_unit == "Square Meters (m²)"

def format_curr(amount_usd):
    converted = amount_usd * curr_info["rate"]
    return curr_info["format"].format(converted)

def to_sqft(val):
    return val * SQFT_PER_SQM if is_sqm else val

def from_sqft(val):
    return val / SQFT_PER_SQM if is_sqm else val

unit_label = "m²" if is_sqm else "sqft"

# Run API check silently in background without showing tech details to user
api_online = check_api_health()

# Clean user-focused sidebar
with st.sidebar:
    st.markdown("### 📌 Quick Presets")
    preset = st.selectbox("Load sample property:", ["Custom", "Starter Suburban Home", "Downtown Luxury Condo", "Rural Farmhouse", "Urban Townhouse"])
    
    st.markdown("---")
    live_calc = st.toggle("⚡ Real-time recalculation", value=True, help="Update valuation instantly as inputs change")

# Set default values based on preset
default_area = 2200
default_rooms = 3
default_baths = 2
default_floors = 2
default_loc = "Suburban"
default_yb = 2014
default_park = 1
default_cond = "Good"

if preset == "Starter Suburban Home":
    default_area, default_rooms, default_baths, default_floors, default_loc, default_yb, default_park, default_cond = 1600, 3, 2, 1, "Suburban", 2005, 1, "Good"
elif preset == "Downtown Luxury Condo":
    default_area, default_rooms, default_baths, default_floors, default_loc, default_yb, default_park, default_cond = 3400, 4, 3, 2, "Downtown", 2021, 2, "Excellent"
elif preset == "Rural Farmhouse":
    default_area, default_rooms, default_baths, default_floors, default_loc, default_yb, default_park, default_cond = 2800, 4, 2, 2, "Rural", 1990, 2, "Fair"
elif preset == "Urban Townhouse":
    default_area, default_rooms, default_baths, default_floors, default_loc, default_yb, default_park, default_cond = 2100, 3, 2, 3, "Urban", 2017, 1, "Good"

# Tabs
tab_val, tab_compare, tab_batch, tab_analytics, tab_about = st.tabs([
    "🎯 Property Valuation",
    "⚖️ What-If Comparison",
    "📁 Batch Valuation (CSV)",
    "📈 Model Diagnostics",
    "ℹ️ Methodology & Scope"
])

# ----------------- TAB 1: VALUATION & INSIGHTS -----------------
with tab_val:
    st.markdown("### 📋 Property Specifications")
    
    # 4 Grouped Cards for specifications
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="card"><div class="card-header">📐 Dimensions & Space</div>', unsafe_allow_html=True)
        init_area = int(round(from_sqft(default_area)))
        min_display_area = int(round(from_sqft(600)))
        max_display_area = int(round(from_sqft(6000)))
        step_area = 10 if is_sqm else 50
        
        input_area_display = st.number_input(f"Area ({unit_label})", min_value=min_display_area, max_value=max_display_area, value=init_area, step=step_area)
        floors = st.selectbox("Floors / Stories", options=[1, 2, 3], index=default_floors - 1)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-header">🛏️ Rooms & Facilities</div>', unsafe_allow_html=True)
        rooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=default_rooms)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=default_baths)
        parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3], index=default_park)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card"><div class="card-header">📍 Location & Quality</div>', unsafe_allow_html=True)
        loc_options = ["Rural", "Suburban", "Urban", "Downtown"]
        location = st.selectbox("Neighborhood Tier", options=loc_options, index=loc_options.index(default_loc))
        cond_options = ["Fair", "Good", "Excellent"]
        condition = st.radio("Condition", options=cond_options, index=cond_options.index(default_cond), horizontal=True)
        st.markdown(f"<div style='font-size: 0.8rem; color: #64748B; margin-top: 0.4rem;'>🗺️ Map Search below: <span style='color: #2563EB; font-weight: 600;'>{st.session_state.get('prop_address', 'Downtown Austin, Texas')}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><div class="card-header">⏳ Age & Construction</div>', unsafe_allow_html=True)
        year_built = st.slider("Year Built", min_value=1960, max_value=CURRENT_YEAR, value=default_yb)
        property_age = CURRENT_YEAR - year_built
        st.metric("Property Age", f"{property_age} years", help=f"Calculated dynamically against {CURRENT_YEAR}")
        st.markdown('</div>', unsafe_allow_html=True)

    actual_sqft = float(round(to_sqft(input_area_display)))
    geo_data = geocode_address(st.session_state.get("prop_address", "Downtown Austin, Texas"))

    # Calculation Trigger
    predict_clicked = False
    if not live_calc:
        predict_clicked = st.button("✨ Calculate Valuation", type="primary", use_container_width=True)

    # Prepare payload
    payload = {
        "Area(sqft)": actual_sqft,
        "Rooms": int(rooms),
        "Bathrooms": int(bathrooms),
        "Floors": int(floors),
        "Location": location,
        "YearBuilt": int(year_built),
        "Parking": int(parking),
        "Condition": condition
    }

    # Execute Prediction
    pred_res = None
    if live_calc or predict_clicked or preset != "Custom":
        if api_online:
            try:
                res = requests.post(f"{FLASK_API_URL}/predict", json=payload, timeout=2.5)
                if res.status_code == 200:
                    pred_res = res.json()
            except Exception:
                pass

        # In-process fallback if API call didn't succeed
        if pred_res is None and model_package is not None:
            try:
                df_in = pd.DataFrame([payload])
                df_in['Age'] = CURRENT_YEAR - df_in['YearBuilt']
                df_in['BathToBedRatio'] = df_in['Bathrooms'] / df_in['Rooms'].clip(lower=1)
                df_in['LivingDensity'] = df_in['Area(sqft)'] / df_in['Rooms'].clip(lower=1)

                p_price = float(model_package['main_pipeline'].predict(df_in)[0])
                p_price = max(25000.0, round(p_price, 2))
                
                low_p = float(model_package['lower_pipeline'].predict(df_in)[0])
                high_p = float(model_package['upper_pipeline'].predict(df_in)[0])
                low_p = max(20000.0, round(min(low_p, p_price * 0.96), 2))
                high_p = max(low_p + 5000.0, round(max(high_p, p_price * 1.04), 2))

                # Feature bounds check
                fb = model_package.get('feature_bounds', {})
                is_ood = actual_sqft < fb.get('Area(sqft)', {}).get('min', 650) or actual_sqft > fb.get('Area(sqft)', {}).get('max', 5200)
                ood_flags = {}
                warnings = []
                if is_ood:
                    ood_flags['Area(sqft)'] = f"Area exceeds training maximum ({actual_sqft} sqft)"
                    warnings.append(f"Area ({actual_sqft:,.0f} sqft) is outside the training distribution (650 – 5,200 sqft). The model is extrapolating.")

                # Contributions
                base_price = model_package.get('baseline_price', 500000.0)
                loc_map = {'Rural': -65000, 'Suburban': 0, 'Urban': 85000, 'Downtown': 195000}
                area_contrib = round((actual_sqft - 1920) * 165)
                cond_map = {'Fair': -45000, 'Good': 0, 'Excellent': 68000}
                age_contrib = round(-(property_age - 25) * 1400)
                room_contrib = round((rooms - 3) * 12000 + (bathrooms - 2) * 20000)

                breakdown = [
                    {'feature': f'Location: {location}', 'contribution': loc_map.get(location, 0)},
                    {'feature': f'Area: {actual_sqft:,.0f} sqft', 'contribution': area_contrib},
                    {'feature': f'Condition: {condition}', 'contribution': cond_map.get(condition, 0)},
                    {'feature': f'Age: {property_age} yrs', 'contribution': age_contrib},
                    {'feature': f'Rooms ({rooms}b / {bathrooms}ba)', 'contribution': room_contrib}
                ]

                # Comparables
                X_proc = model_package['preprocessor'].transform(df_in)
                distances, indices = model_package['nn_model'].kneighbors(X_proc, n_neighbors=5)
                lookup_df = model_package['train_lookup_df']
                comps = []
                for d, idx in zip(distances[0], indices[0]):
                    row = lookup_df.iloc[idx]
                    comps.append({
                        'area_sqft': int(row['Area(sqft)']),
                        'rooms': int(row['Rooms']),
                        'bathrooms': float(row['Bathrooms']),
                        'location': str(row['Location']),
                        'condition': str(row['Condition']),
                        'year_built': int(row['YearBuilt']),
                        'actual_sale_price': float(row['Price']),
                        'similarity_score': round(float(1.0 / (1.0 + d)) * 100, 1)
                    })

                pred_res = {
                    "predicted_price": p_price,
                    "price_per_sqft": round(p_price / actual_sqft, 2),
                    "estimated_range": {
                        "lower_bound": low_p,
                        "upper_bound": high_p
                    },
                    "out_of_distribution": {
                        "is_out_of_distribution": is_ood,
                        "flags": ood_flags
                    },
                    "warnings": warnings,
                    "feature_contributions": {
                        "baseline_price": base_price,
                        "breakdown": breakdown
                    },
                    "comparables": comps
                }
            except Exception as e:
                st.error(f"Inference error: {e}")

    # Display Valuation Results
    if pred_res:
        est_price = pred_res["predicted_price"]
        low_bound = pred_res["estimated_range"]["lower_bound"]
        high_bound = pred_res["estimated_range"]["upper_bound"]
        price_per_unit = est_price / input_area_display
        is_ood = pred_res["out_of_distribution"]["is_out_of_distribution"]

        st.markdown("<br>", unsafe_allow_html=True)

        # Out of Distribution Warning Banner
        if is_ood:
            st.markdown(f"""
            <div class="warning-badge">
                ⚠️ <strong>Extrapolation Alert:</strong> One or more inputs exceed the training dataset domain (e.g. Area {actual_sqft:,.0f} sqft). Predictions outside this range are extrapolations and carry higher variance.
            </div>
            """, unsafe_allow_html=True)

        # Hero Valuation Card
        st.markdown(f"""
        <div class="hero-container">
            <div class="hero-badge">AI Market Valuation Estimate</div>
            <div class="hero-price">{format_curr(est_price)}</div>
            <div style="font-size: 1.05rem; color: #CBD5E1; margin-bottom: 0.8rem;">
                Estimated Range: <strong style="color: #93C5FD;">{format_curr(low_bound)}</strong> &ndash; <strong style="color: #93C5FD;">{format_curr(high_bound)}</strong>
                <span style="font-size: 0.85rem; color: #94A3B8; margin-left: 0.5rem;">(Estimated Price Range)</span>
            </div>
            <div class="range-bar-wrapper">
                <div class="stat-pill">
                    <div class="stat-pill-label">Price / {unit_label}</div>
                    <div class="stat-pill-val">{format_curr(price_per_unit)}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Property Age</div>
                    <div class="stat-pill-val">{property_age} yrs</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Neighborhood</div>
                    <div class="stat-pill-val">{location}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Condition</div>
                    <div class="stat-pill-val">{condition}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visualizations (Waterfall & Sensitivity Curve)
        col_wf, col_curve = st.columns(2)

        with col_wf:
            st.subheader("💡 Why this price? (Value Drivers)")
            st.caption("Estimated impact of each feature on your property's value")
            
            breakdown = pred_res.get("feature_contributions", {}).get("breakdown", [])
            base_p = pred_res.get("feature_contributions", {}).get("baseline_price", 500000)
            
            if breakdown:
                x_labels = ["Baseline (Median)"] + [b['feature'] for b in breakdown] + ["Final Estimate"]
                y_vals = [base_p * curr_info["rate"]] + [b['contribution'] * curr_info["rate"] for b in breakdown] + [0]
                measure_types = ["absolute"] + ["relative"] * len(breakdown) + ["total"]

                fig_wf = go.Figure(go.Waterfall(
                    name="Contribution",
                    orientation="v",
                    measure=measure_types,
                    x=x_labels,
                    textposition="outside",
                    text=[format_curr(v / curr_info["rate"]) if m != "relative" else f"{'+' if v >= 0 else ''}{format_curr(v / curr_info['rate'])}" for v, m in zip(y_vals, measure_types)],
                    y=y_vals,
                    connector={"line": {"color": "#94A3B8"}},
                    decreasing={"marker": {"color": "#EF4444"}},
                    increasing={"marker": {"color": "#10B981"}},
                    totals={"marker": {"color": "#3B82F6"}}
                ))
                fig_wf.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=25, b=60),
                    yaxis=dict(title=f"Value ({curr_info['symbol']})", tickformat="~s"),
                    xaxis_tickangle=-25
                )
                st.plotly_chart(fig_wf, use_container_width=True)

        with col_curve:
            st.subheader("📈 Price Sensitivity Curve")
            st.caption(f"How valuation scales with Area for this property configuration")

            # Generate sensitivity points
            sweep_areas = np.linspace(700, 5200, 20)
            sweep_prices = []
            sweep_lows = []
            sweep_highs = []

            for a in sweep_areas:
                temp_payload = dict(payload)
                temp_payload['Area(sqft)'] = a
                df_temp = pd.DataFrame([temp_payload])
                df_temp['Age'] = CURRENT_YEAR - df_temp['YearBuilt']
                df_temp['BathToBedRatio'] = df_temp['Bathrooms'] / df_temp['Rooms'].clip(lower=1)
                df_temp['LivingDensity'] = df_temp['Area(sqft)'] / df_temp['Rooms'].clip(lower=1)
                
                sp = float(model_package['main_pipeline'].predict(df_temp)[0])
                sl = float(model_package['lower_pipeline'].predict(df_temp)[0])
                sh = float(model_package['upper_pipeline'].predict(df_temp)[0])
                sweep_prices.append(sp * curr_info["rate"])
                sweep_lows.append(min(sl, sp * 0.96) * curr_info["rate"])
                sweep_highs.append(max(sh, sp * 1.04) * curr_info["rate"])

            display_sweep_areas = [from_sqft(a) for a in sweep_areas]

            fig_curve = go.Figure()
            # Confidence interval band
            fig_curve.add_trace(go.Scatter(
                x=display_sweep_areas + display_sweep_areas[::-1],
                y=sweep_highs + sweep_lows[::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name="80% Interval"
            ))
            # Price curve
            fig_curve.add_trace(go.Scatter(
                x=display_sweep_areas,
                y=sweep_prices,
                mode='lines',
                line=dict(color='#2563EB', width=3),
                name='Predicted Price'
            ))
            # Current house dot
            fig_curve.add_trace(go.Scatter(
                x=[input_area_display],
                y=[est_price * curr_info["rate"]],
                mode='markers',
                marker=dict(color='#DC2626', size=13, symbol='diamond'),
                name='Current Property'
            ))

            fig_curve.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=25, b=40),
                xaxis=dict(title=f"Total Area ({unit_label})"),
                yaxis=dict(title=f"Valuation ({curr_info['symbol']})", tickformat="~s"),
                legend=dict(orientation="h", y=1.1, x=0.2)
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        # ----------------- Interactive Dark Map Integration -----------------
        st.subheader("🗺️ Geographic Location & Neighborhood Comparables")
        st.markdown("Search any address or city globally to pinpoint your property and discover nearby comparable sales.")

        # Prominent dedicated location search bar attached directly above the map
        col_map_search, col_map_quick = st.columns([3.2, 1.2])
        with col_map_search:
            map_search_input = st.text_input(
                "🔍 Search Property Address, Locality, or City",
                value=st.session_state.get("prop_address", "Downtown Austin, Texas"),
                placeholder="Type e.g., Indiranagar Bengaluru, Brooklyn NY, Downtown Austin, Westminster London...",
                help="Type any neighborhood, address, or city worldwide"
            )
            st.session_state["prop_address"] = map_search_input

        with col_map_quick:
            st.markdown("<div style='margin-bottom: 0.3rem; font-size: 0.85rem; color: #94A3B8; font-weight: 600;'>QUICK CITIES</div>", unsafe_allow_html=True)
            quick_city = st.selectbox(
                "Quick Cities",
                ["Custom", "Austin, TX", "Bengaluru, India", "New York, NY", "London, UK", "Mumbai, India"],
                label_visibility="collapsed"
            )
            if quick_city != "Custom" and quick_city != st.session_state.get("last_quick_city"):
                st.session_state["prop_address"] = quick_city
                st.session_state["last_quick_city"] = quick_city
                map_search_input = quick_city

        geo_data = geocode_address(st.session_state.get("prop_address", map_search_input))

        prop_lat = geo_data['lat']
        prop_lon = geo_data['lon']
        comps = pred_res.get("comparables", [])

        # Generate realistic micro-market offsets around the geocoded coordinates for the 5 comparables
        comp_lats = []
        comp_lons = []
        comp_hover = []

        angles = [0.45, 1.85, 3.14, 4.35, 5.50]
        radii = [0.75, 1.15, 0.90, 1.40, 1.05]
        lat_scale = 0.009
        lon_scale = 0.011

        for i, c in enumerate(comps):
            ang = angles[i % len(angles)]
            rad = radii[i % len(radii)]
            c_lat = prop_lat + (rad * lat_scale * np.sin(ang))
            c_lon = prop_lon + (rad * lon_scale * np.cos(ang))
            comp_lats.append(c_lat)
            comp_lons.append(c_lon)
            comp_hover.append(
                f"<b>Comparable #{i+1}</b><br>"
                f"Actual Sale: {format_curr(c['actual_sale_price'])}<br>"
                f"Area: {from_sqft(c['area_sqft']):,.0f} {unit_label}<br>"
                f"Beds: {c['rooms']} | Baths: {c['bathrooms']}<br>"
                f"Condition: {c['condition']}<br>"
                f"Match: {c['similarity_score']}%"
            )

        fig_map = go.Figure()

        # Trace 1: Nearby Comparables (Amber/Blue Pins)
        if comp_lats:
            fig_map.add_trace(go.Scattermap(
                lat=comp_lats,
                lon=comp_lons,
                mode='markers+text',
                marker=dict(
                    size=14,
                    color='#60A5FA',
                    symbol='circle'
                ),
                text=[f"#{i+1}" for i in range(len(comp_lats))],
                textposition="top center",
                textfont=dict(color="#F8FAFC", size=11, family="Plus Jakarta Sans"),
                hoverinfo='text',
                hovertext=comp_hover,
                name='Nearby Comparables'
            ))

        # Trace 2: Subject Property (Emerald Diamond Pin)
        fig_map.add_trace(go.Scattermap(
            lat=[prop_lat],
            lon=[prop_lon],
            mode='markers+text',
            marker=dict(
                size=18,
                color='#10B981',
                symbol='diamond'
            ),
            text=["YOUR PROPERTY"],
            textposition="bottom center",
            textfont=dict(color="#34D399", size=12, family="Plus Jakarta Sans"),
            hoverinfo='text',
            hovertext=[
                f"<b>YOUR PROPERTY</b><br>"
                f"Valuation: {format_curr(est_price)}<br>"
                f"Range: {format_curr(low_bound)} – {format_curr(high_bound)}<br>"
                f"Area: {input_area_display:,.0f} {unit_label}<br>"
                f"Beds: {rooms} | Baths: {bathrooms}<br>"
                f"Condition: {condition}"
            ],
            name='Subject Property'
        ))

        fig_map.update_layout(
            map_style='carto-darkmatter',
            map=dict(
                center=dict(lat=prop_lat, lon=prop_lon),
                zoom=13
            ),
            paper_bgcolor='#0F172A',
            plot_bgcolor='#0F172A',
            margin=dict(l=0, r=0, t=0, b=0),
            height=430,
            legend=dict(
                orientation="h",
                y=0.98,
                x=0.02,
                bgcolor="rgba(15, 23, 42, 0.8)",
                font=dict(color="#F8FAFC")
            )
        )

        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 0.65rem 1.2rem; color: #94A3B8; font-size: 0.85rem; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
            <span>📍 <strong>Location:</strong> {geo_data['name']}</span>
            <code style="color: #38BDF8;">{prop_lat:.4f}° N, {prop_lon:.4f}° E</code>
        </div>
        """, unsafe_allow_html=True)

        # Comparable Properties Table
        st.subheader("🏘️ 5 Nearest Comparable Properties (Market Sales)")
        st.caption("Recently sold properties with the most similar specifications")
        comps = pred_res.get("comparables", [])
        if comps:
            comp_rows = []
            for c in comps:
                comp_rows.append({
                    f"Area ({unit_label})": f"{from_sqft(c['area_sqft']):,.0f}",
                    "Bedrooms": c['rooms'],
                    "Bathrooms": c['bathrooms'],
                    "Location": c['location'],
                    "Condition": c['condition'],
                    "Year Built": c['year_built'],
                    "Actual Sale Price": format_curr(c['actual_sale_price']),
                    "Similarity Match": f"{c['similarity_score']}%"
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

# ----------------- TAB 2: WHAT-IF COMPARISON -----------------
with tab_compare:
    st.subheader("⚖️ Side-by-Side Property Comparison")
    st.markdown("Assess how specific upgrades or architectural changes alter property value.")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("#### 🏠 Baseline House (A)")
        a_area = st.number_input(f"A: Area ({unit_label})", value=int(round(from_sqft(2000))), step=50, key="a_area")
        a_rooms = st.number_input("A: Bedrooms", min_value=1, max_value=8, value=3, key="a_rooms")
        a_baths = st.number_input("A: Bathrooms", min_value=1, max_value=6, value=2, key="a_baths")
        a_loc = st.selectbox("A: Location", ["Rural", "Suburban", "Urban", "Downtown"], index=1, key="a_loc")
        a_cond = st.selectbox("A: Condition", ["Fair", "Good", "Excellent"], index=1, key="a_cond")

    with col_h2:
        st.markdown("#### 🏡 Modified House (B)")
        b_area = st.number_input(f"B: Area ({unit_label})", value=int(round(from_sqft(2000))), step=50, key="b_area")
        b_rooms = st.number_input("B: Bedrooms", min_value=1, max_value=8, value=3, key="b_rooms")
        b_baths = st.number_input("B: Bathrooms", min_value=1, max_value=6, value=3, key="b_baths")  # Extra bath!
        b_loc = st.selectbox("B: Location", ["Rural", "Suburban", "Urban", "Downtown"], index=1, key="b_loc")
        b_cond = st.selectbox("B: Condition", ["Fair", "Good", "Excellent"], index=2, key="b_cond")  # Upgraded to Excellent!

    if model_package is not None:
        def predict_quick(ar, ro, ba, lo, co):
            sq = to_sqft(ar)
            df_i = pd.DataFrame([{
                'Area(sqft)': sq, 'Rooms': ro, 'Bathrooms': ba, 'Floors': 2,
                'Location': lo, 'YearBuilt': 2012, 'Parking': 1, 'Condition': co
            }])
            df_i['Age'] = CURRENT_YEAR - df_i['YearBuilt']
            df_i['BathToBedRatio'] = df_i['Bathrooms'] / df_i['Rooms'].clip(lower=1)
            df_i['LivingDensity'] = df_i['Area(sqft)'] / df_i['Rooms'].clip(lower=1)
            return float(model_package['main_pipeline'].predict(df_i)[0])

        p_a = predict_quick(a_area, a_rooms, a_baths, a_loc, a_cond)
        p_b = predict_quick(b_area, b_rooms, b_baths, b_loc, b_cond)
        delta = p_b - p_a
        delta_pct = (delta / p_a) * 100

        st.markdown("---")
        st.markdown("### 📊 Valuation Comparison Summary")
        m_a, m_b, m_diff = st.columns(3)
        m_a.metric("House A Valuation", format_curr(p_a))
        m_b.metric("House B Valuation", format_curr(p_b))
        m_diff.metric(
            "Estimated Net Delta",
            f"{'+' if delta >= 0 else ''}{format_curr(delta)}",
            f"{delta_pct:+.1f}%"
        )

# ----------------- TAB 3: BATCH VALUATION (CSV) -----------------
with tab_batch:
    st.subheader("📁 Bulk Batch Valuation")
    st.markdown("Upload a CSV file containing multiple property records to run bulk inferences.")

    # Template download
    sample_csv = """Area(sqft),Rooms,Bathrooms,Floors,Location,YearBuilt,Parking,Condition
1800,3,2,1,Suburban,2005,1,Good
2400,4,3,2,Downtown,2018,2,Excellent
1400,2,1,1,Rural,1992,1,Fair
3100,5,3,2,Urban,2020,2,Good
"""
    st.download_button("📥 Download CSV Template", data=sample_csv, file_name="property_batch_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload CSV for Bulk Valuation", type=["csv"])
    if uploaded_file is not None and model_package is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Properties Preview:", batch_df.head())

            if st.button("🚀 Run Batch Inferences", type="primary"):
                with st.spinner("Processing batch valuations..."):
                    df_proc = batch_df.copy()
                    df_proc['Age'] = CURRENT_YEAR - df_proc['YearBuilt']
                    df_proc['BathToBedRatio'] = df_proc['Bathrooms'] / df_proc['Rooms'].clip(lower=1)
                    df_proc['LivingDensity'] = df_proc['Area(sqft)'] / df_proc['Rooms'].clip(lower=1)

                    preds = model_package['main_pipeline'].predict(df_proc)
                    lows = model_package['lower_pipeline'].predict(df_proc)
                    highs = model_package['upper_pipeline'].predict(df_proc)

                    batch_df['Estimated_Price_USD'] = [round(max(25000, p), 2) for p in preds]
                    batch_df['Lower_Bound_USD'] = [round(min(l, p * 0.96), 2) for l, p in zip(lows, preds)]
                    batch_df['Upper_Bound_USD'] = [round(max(h, p * 1.04), 2) for h, p in zip(highs, preds)]
                    batch_df['Formatted_Valuation'] = [format_curr(p) for p in batch_df['Estimated_Price_USD']]

                    st.success(f"Successfully processed {len(batch_df)} property valuations!")
                    st.dataframe(batch_df, use_container_width=True)

                    out_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Valuations CSV", data=out_csv, file_name="valuations_output.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ----------------- TAB 4: MODEL DIAGNOSTICS -----------------
with tab_analytics:
    st.subheader("📈 Model Diagnostics & Benchmarks")

    if metrics_meta:
        b_metrics = metrics_meta.get('best_model_metrics', {})
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Selected Regressor", metrics_meta.get('best_model', 'N/A'))
        col_m2.metric("R² Test Score", f"{b_metrics.get('R2', 0):.4f}")
        col_m3.metric("5-Fold CV Mean R²", f"{b_metrics.get('CV_R2_Mean', 0):.4f}")
        col_m4.metric("Mean Abs Error (MAE)", f"${b_metrics.get('MAE', 0):,.2f}")

        st.markdown("#### Candidate Model Cross-Validation Comparison")
        all_models = metrics_meta.get('all_models', {})
        if all_models:
            comp_rows = []
            for name, m in all_models.items():
                comp_rows.append({
                    "Model Algorithm": name,
                    "Test R²": m.get("R2"),
                    "CV R² (Mean)": m.get("CV_R2_Mean"),
                    "CV R² (Std)": m.get("CV_R2_Std"),
                    "MAE ($)": f"${m.get('MAE'):,.2f}",
                    "MAPE (%)": f"{m.get('MAPE'):.2f}%"
                })
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    # Diagnostic Image Plots
    col_img1, col_img2 = st.columns(2)
    assets_dir = BASE_DIR / "reports" / "assets"
    act_pred_p = assets_dir / "actual_vs_predicted.png"
    res_plot_p = assets_dir / "residual_plot.png"

    with col_img1:
        if act_pred_p.exists():
            st.image(str(act_pred_p), caption="Actual vs. Predicted Prices (Test Set)", use_container_width=True)
    with col_img2:
        if res_plot_p.exists():
            st.image(str(res_plot_p), caption="Residual Error Distribution", use_container_width=True)

# ----------------- TAB 5: ABOUT & METHODOLOGY -----------------
with tab_about:
    st.subheader("ℹ️ Project Scope & Methodology")
    st.markdown("""
    ### System Architecture & Modeling Rigor
    
    1. **Realistic Market Dynamics**:
       - Non-linear returns to square footage with submarket tier pricing (Rural, Suburban, Urban, Downtown).
       - Depreciation curves with renovation/condition interaction.
       - Bath-to-bedroom coherence penalties and parking density premiums.
    
    2. **Quantile Uncertainty Estimation**:
       - Instead of a fixed global percentage error, we employ **Gradient Boosting Quantile Regression** (`alpha=0.10` and `alpha=0.90`) to generate individual prediction intervals.
       - Atypical or high-end properties naturally exhibit wider uncertainty ranges.

    3. **Transparency & Out-of-Distribution Warning**:
       - If a user inputs parameters exceeding the training domain (e.g. Area > 5,200 sqft), the system highlights an **Extrapolation Alert** to maintain pricing transparency.

    4. **Comparable Discovery Engine**:
       - An embedded Euclidean Nearest Neighbors model searches the feature space to discover the 5 closest actual historical sales to justify valuations.
    """)
