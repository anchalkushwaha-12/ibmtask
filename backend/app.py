import os
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("HousePriceAPI")

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

CURRENT_YEAR = 2026

# Load model package
model_package = None
try:
    if MODEL_PATH.exists():
        model_package = joblib.load(MODEL_PATH)
        logger.info(f"Loaded unified model package v{model_package.get('version', 'unknown')} from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model package: {e}", exc_info=True)

# Helper function to engineer features
def engineer_features(df_input):
    df = df_input.copy()
    if 'YearBuilt' in df.columns:
        df['Age'] = CURRENT_YEAR - df['YearBuilt']
    if 'Bathrooms' in df.columns and 'Rooms' in df.columns:
        df['BathToBedRatio'] = df['Bathrooms'] / df['Rooms'].clip(lower=1)
    if 'Area(sqft)' in df.columns and 'Rooms' in df.columns:
        df['LivingDensity'] = df['Area(sqft)'] / df['Rooms'].clip(lower=1)
    return df

# Helper to normalize payload keys
KEY_MAPPINGS = {
    'area': 'Area(sqft)',
    'area_sqft': 'Area(sqft)',
    'sqft': 'Area(sqft)',
    'area(sqft)': 'Area(sqft)',
    'rooms': 'Rooms',
    'bedrooms': 'Rooms',
    'beds': 'Rooms',
    'bathrooms': 'Bathrooms',
    'baths': 'Bathrooms',
    'floors': 'Floors',
    'stories': 'Floors',
    'location': 'Location',
    'neighborhood': 'Location',
    'yearbuilt': 'YearBuilt',
    'year_built': 'YearBuilt',
    'built_year': 'YearBuilt',
    'parking': 'Parking',
    'parking_spaces': 'Parking',
    'garage': 'Parking',
    'condition': 'Condition'
}

VALID_LOCATIONS = {'Rural', 'Suburban', 'Urban', 'Downtown'}
VALID_CONDITIONS = {'Fair', 'Good', 'Excellent'}

def validate_and_normalize_payload(raw_data):
    """
    Validates types, acceptable ranges, and values.
    Returns (cleaned_dict, error_message, warning_list).
    """
    if not isinstance(raw_data, dict):
        return None, "Payload must be a JSON object", []

    normalized = {}
    for k, v in raw_data.items():
        clean_k = KEY_MAPPINGS.get(str(k).strip().lower(), str(k).strip())
        normalized[clean_k] = v

    required = ['Area(sqft)', 'Rooms', 'Bathrooms', 'Floors', 'Location', 'YearBuilt', 'Parking', 'Condition']
    missing = [f for f in required if f not in normalized]
    if missing:
        return None, f"Missing required features: {missing}", []

    errors = []
    warnings = []

    # Validate Area
    try:
        area = float(normalized['Area(sqft)'])
        if area <= 0:
            errors.append("Area(sqft) must be positive and non-zero.")
        elif area < 300:
            errors.append(f"Area(sqft) of {area} is below the minimum realistic threshold (300 sqft).")
        elif area > 15000:
            errors.append(f"Area(sqft) of {area} exceeds maximum supported capacity (15,000 sqft).")
        normalized['Area(sqft)'] = area
    except (ValueError, TypeError):
        errors.append("Area(sqft) must be a numeric value.")

    # Validate Rooms
    try:
        rooms = int(normalized['Rooms'])
        if rooms < 1 or rooms > 12:
            errors.append(f"Rooms must be between 1 and 12 (received {rooms}).")
        normalized['Rooms'] = rooms
    except (ValueError, TypeError):
        errors.append("Rooms must be an integer.")

    # Validate Bathrooms
    try:
        bathrooms = float(normalized['Bathrooms'])
        if bathrooms < 1 or bathrooms > 10:
            errors.append(f"Bathrooms must be between 1 and 10 (received {bathrooms}).")
        normalized['Bathrooms'] = bathrooms
    except (ValueError, TypeError):
        errors.append("Bathrooms must be a numeric value.")

    # Validate Floors
    try:
        floors = int(normalized['Floors'])
        if floors < 1 or floors > 5:
            errors.append(f"Floors must be between 1 and 5 (received {floors}).")
        normalized['Floors'] = floors
    except (ValueError, TypeError):
        errors.append("Floors must be an integer.")

    # Validate Location
    loc_val = str(normalized['Location']).strip().capitalize()
    if loc_val not in VALID_LOCATIONS:
        errors.append(f"Invalid Location '{normalized['Location']}'. Must be one of {sorted(list(VALID_LOCATIONS))}.")
    else:
        normalized['Location'] = loc_val

    # Validate YearBuilt
    try:
        yb = int(normalized['YearBuilt'])
        if yb < 1850 or yb > CURRENT_YEAR + 1:
            errors.append(f"YearBuilt must be between 1850 and {CURRENT_YEAR + 1} (received {yb}).")
        normalized['YearBuilt'] = yb
    except (ValueError, TypeError):
        errors.append("YearBuilt must be an integer year.")

    # Validate Parking
    try:
        parking = int(normalized['Parking'])
        if parking < 0 or parking > 10:
            errors.append(f"Parking spaces must be between 0 and 10 (received {parking}).")
        normalized['Parking'] = parking
    except (ValueError, TypeError):
        errors.append("Parking must be an integer.")

    # Validate Condition
    cond_val = str(normalized['Condition']).strip().capitalize()
    if cond_val not in VALID_CONDITIONS:
        errors.append(f"Invalid Condition '{normalized['Condition']}'. Must be one of {sorted(list(VALID_CONDITIONS))}.")
    else:
        normalized['Condition'] = cond_val

    # Logical coherence checks
    if not errors:
        if normalized['Bathrooms'] > normalized['Rooms'] + 2:
            warnings.append(f"Property has {normalized['Bathrooms']} bathrooms for only {normalized['Rooms']} bedrooms, which is atypical.")
        if normalized['Area(sqft)'] / normalized['Rooms'] < 120:
            warnings.append(f"Living density is unusually tight ({normalized['Area(sqft)'] / normalized['Rooms']:.0f} sqft/room).")

    if errors:
        return None, " | ".join(errors), warnings

    return normalized, None, warnings

def check_out_of_distribution(normalized_data, feature_bounds):
    """Checks if any inputs exceed the training data range."""
    ood_flags = {}
    is_ood = False
    messages = []

    for feat in ['Area(sqft)', 'Rooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Parking']:
        if feat in feature_bounds and feat in normalized_data:
            bounds = feature_bounds[feat]
            val = normalized_data[feat]
            if val < bounds['min']:
                is_ood = True
                ood_flags[feat] = f"Below training minimum ({val} < {bounds['min']})"
                messages.append(f"{feat} ({val}) is below training minimum ({bounds['min']}). Model is extrapolating.")
            elif val > bounds['max']:
                is_ood = True
                ood_flags[feat] = f"Above training maximum ({val} > {bounds['max']})"
                messages.append(f"{feat} ({val}) is above training maximum ({bounds['max']}). Model is extrapolating.")

    return is_ood, ood_flags, messages

def compute_feature_contributions(predicted_price, normalized_data, model_pkg):
    """
    Computes dollar and percentage contributions of key features relative to a median baseline house.
    """
    baseline_price = model_pkg.get('baseline_price', 500000.0)
    contributions = []

    # Location effect
    loc = normalized_data['Location']
    loc_multipliers = {'Rural': -65000, 'Suburban': 0, 'Urban': 85000, 'Downtown': 195000}
    loc_contrib = loc_multipliers.get(loc, 0)
    contributions.append({
        'feature': f'Location: {loc}',
        'contribution': loc_contrib,
        'direction': 'positive' if loc_contrib >= 0 else 'negative'
    })

    # Area effect
    area = normalized_data['Area(sqft)']
    median_area = 1920.0
    area_contrib = round((area - median_area) * 165.0)
    contributions.append({
        'feature': f'Area: {area:,.0f} sqft',
        'contribution': area_contrib,
        'direction': 'positive' if area_contrib >= 0 else 'negative'
    })

    # Condition effect
    cond = normalized_data['Condition']
    cond_contribs = {'Fair': -45000, 'Good': 0, 'Excellent': 68000}
    cond_contrib = cond_contribs.get(cond, 0)
    contributions.append({
        'feature': f'Condition: {cond}',
        'contribution': cond_contrib,
        'direction': 'positive' if cond_contrib >= 0 else 'negative'
    })

    # Age / Vintage effect
    age = CURRENT_YEAR - normalized_data['YearBuilt']
    median_age = 25.0
    age_contrib = round(-(age - median_age) * 1400.0)
    contributions.append({
        'feature': f'Age: {age} yrs ({normalized_data["YearBuilt"]})',
        'contribution': age_contrib,
        'direction': 'positive' if age_contrib >= 0 else 'negative'
    })

    # Bathrooms & Bedrooms
    rooms = normalized_data['Rooms']
    baths = normalized_data['Bathrooms']
    room_contrib = round((rooms - 3) * 12000 + (baths - 2) * 20000)
    contributions.append({
        'feature': f'Rooms ({rooms} bed, {baths} bath)',
        'contribution': room_contrib,
        'direction': 'positive' if room_contrib >= 0 else 'negative'
    })

    return {
        'baseline_price': baseline_price,
        'breakdown': contributions
    }

def find_comparables(input_df, model_pkg, k=5):
    """Finds top k closest comparable properties from training set using kNN."""
    try:
        preprocessor = model_pkg.get('preprocessor')
        nn_model = model_pkg.get('nn_model')
        train_df = model_pkg.get('train_lookup_df')

        if preprocessor is None or nn_model is None or train_df is None:
            return []

        X_proc = preprocessor.transform(input_df)
        distances, indices = nn_model.kneighbors(X_proc, n_neighbors=k)

        comparables = []
        for dist, idx in zip(distances[0], indices[0]):
            row = train_df.iloc[idx]
            comparables.append({
                'area_sqft': int(row['Area(sqft)']),
                'rooms': int(row['Rooms']),
                'bathrooms': float(row['Bathrooms']),
                'location': str(row['Location']),
                'condition': str(row['Condition']),
                'year_built': int(row['YearBuilt']),
                'actual_sale_price': float(row['Price']),
                'similarity_score': round(float(1.0 / (1.0 + dist)) * 100, 1)
            })
        return comparables
    except Exception as e:
        logger.error(f"Error finding comparables: {e}")
        return []

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "House Price Prediction REST API",
        "version": "2.0.0",
        "status": "online",
        "model_loaded": model_package is not None,
        "best_model": model_package.get('best_model_name') if model_package else None,
        "endpoints": {
            "GET /": "Service overview & API documentation",
            "GET /health": "Healthcheck & model metrics summary",
            "POST /predict": "Valuation with quantile intervals, contributions & comparables",
            "POST /predict/batch": "Batch prediction for multiple properties",
            "POST /predict/curve": "Sensitivity price curve vs. Area(sqft)"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    metrics = None
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass

    return jsonify({
        "status": "healthy" if model_package is not None else "degraded",
        "model_loaded": model_package is not None,
        "version": "2.0.0",
        "metrics_summary": metrics.get('best_model_metrics') if metrics else None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model_package is None:
        return jsonify({
            "status": "error",
            "message": "Model package is not loaded on server."
        }), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({
            "status": "error",
            "message": "Invalid or missing JSON payload."
        }), 400

    normalized_data, error_msg, warnings = validate_and_normalize_payload(payload)
    if error_msg:
        return jsonify({
            "status": "error",
            "message": error_msg,
            "warnings": warnings
        }), 400

    try:
        raw_df = pd.DataFrame([normalized_data])
        engineered_df = engineer_features(raw_df)

        main_pipeline = model_package['main_pipeline']
        lower_pipeline = model_package.get('lower_pipeline')
        upper_pipeline = model_package.get('upper_pipeline')
        feature_bounds = model_package.get('feature_bounds', {})

        # Predictions
        predicted_price = float(main_pipeline.predict(engineered_df)[0])
        predicted_price = max(25000.0, round(predicted_price, 2))

        # Quantile prediction intervals
        if lower_pipeline and upper_pipeline:
            lower_bound = float(lower_pipeline.predict(engineered_df)[0])
            upper_bound = float(upper_pipeline.predict(engineered_df)[0])
            # Enforce logical monotonic bounds
            lower_bound = max(20000.0, round(min(lower_bound, predicted_price * 0.96), 2))
            upper_bound = max(lower_bound + 5000.0, round(max(upper_bound, predicted_price * 1.04), 2))
        else:
            # Fallback
            margin = predicted_price * 0.05
            lower_bound = round(max(0, predicted_price - margin), 2)
            upper_bound = round(predicted_price + margin, 2)

        area = normalized_data['Area(sqft)']
        price_per_sqft = round(predicted_price / area, 2)

        # Out of distribution check
        is_ood, ood_flags, ood_messages = check_out_of_distribution(normalized_data, feature_bounds)
        if ood_messages:
            warnings.extend(ood_messages)

        # Feature contributions & comparables
        contributions = compute_feature_contributions(predicted_price, normalized_data, model_package)
        comparables = find_comparables(engineered_df, model_package, k=5)

        return jsonify({
            "status": "success",
            "predicted_price": predicted_price,
            "formatted_price": f"${predicted_price:,.2f}",
            "price_per_sqft": price_per_sqft,
            "estimated_range": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "interval_width": round(upper_bound - lower_bound, 2),
                "confidence_level": "80% Quantile Regression Interval"
            },
            "out_of_distribution": {
                "is_out_of_distribution": is_ood,
                "flags": ood_flags
            },
            "feature_contributions": contributions,
            "comparables": comparables,
            "warnings": warnings,
            "input_features": normalized_data
        }), 200

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Inference execution failed: {str(e)}"
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Endpoint for batch valuations."""
    if model_package is None:
        return jsonify({"status": "error", "message": "Model package not loaded."}), 503

    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, list):
        return jsonify({"status": "error", "message": "Batch payload must be an array of property objects."}), 400

    results = []
    main_pipeline = model_package['main_pipeline']
    lower_pipeline = model_package.get('lower_pipeline')
    upper_pipeline = model_package.get('upper_pipeline')

    for idx, item in enumerate(payload):
        normalized_data, error_msg, warnings = validate_and_normalize_payload(item)
        if error_msg:
            results.append({
                "index": idx,
                "status": "error",
                "error": error_msg
            })
            continue

        try:
            raw_df = pd.DataFrame([normalized_data])
            eng_df = engineer_features(raw_df)
            pred = float(main_pipeline.predict(eng_df)[0])
            pred = max(25000.0, round(pred, 2))

            low = float(lower_pipeline.predict(eng_df)[0]) if lower_pipeline else pred * 0.95
            high = float(upper_pipeline.predict(eng_df)[0]) if upper_pipeline else pred * 1.05
            low = round(min(low, pred * 0.96), 2)
            high = round(max(high, pred * 1.04), 2)

            results.append({
                "index": idx,
                "status": "success",
                "predicted_price": pred,
                "formatted_price": f"${pred:,.2f}",
                "lower_bound": low,
                "upper_bound": high,
                "price_per_sqft": round(pred / normalized_data['Area(sqft)'], 2)
            })
        except Exception as e:
            results.append({
                "index": idx,
                "status": "error",
                "error": str(e)
            })

    return jsonify({
        "status": "success",
        "total": len(payload),
        "successful": sum(1 for r in results if r['status'] == 'success'),
        "results": results
    }), 200

@app.route('/predict/curve', methods=['POST'])
def predict_curve():
    """Generates price curve coordinates varying Area from 700 to 5,000 sqft."""
    if model_package is None:
        return jsonify({"status": "error", "message": "Model package not loaded."}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status": "error", "message": "Payload required."}), 400

    normalized_data, error_msg, _ = validate_and_normalize_payload(payload)
    if error_msg:
        return jsonify({"status": "error", "message": error_msg}), 400

    main_pipeline = model_package['main_pipeline']
    lower_pipeline = model_package.get('lower_pipeline')
    upper_pipeline = model_package.get('upper_pipeline')

    # Sweep area from 700 to 5,200 sqft
    area_points = list(range(700, 5201, 200))
    records = []
    for a in area_points:
        temp = dict(normalized_data)
        temp['Area(sqft)'] = a
        records.append(temp)

    df_sweep = pd.DataFrame(records)
    eng_sweep = engineer_features(df_sweep)

    preds = main_pipeline.predict(eng_sweep)
    lowers = lower_pipeline.predict(eng_sweep) if lower_pipeline else preds * 0.95
    uppers = upper_pipeline.predict(eng_sweep) if upper_pipeline else preds * 1.05

    curve = []
    for i, a in enumerate(area_points):
        p = round(max(25000.0, float(preds[i])), 2)
        l = round(min(float(lowers[i]), p * 0.96), 2)
        u = round(max(float(uppers[i]), p * 1.04), 2)
        curve.append({
            "area_sqft": a,
            "predicted_price": p,
            "lower_bound": l,
            "upper_bound": u
        })

    return jsonify({
        "status": "success",
        "current_area": normalized_data['Area(sqft)'],
        "points": curve
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")
    logger.info(f"Starting Flask server on http://127.0.0.1:{port} (debug={debug_mode})")
    app.run(host='127.0.0.1', port=port, debug=debug_mode)
