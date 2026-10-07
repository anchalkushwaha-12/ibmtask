# 🏠 House Price Predictor (ValuEdge)

An end-to-end Machine Learning property valuation platform with honest uncertainty intervals, feature contributions (SHAP-style value drivers), kNN comparable discovery, and batch prediction. Built using Python, Scikit-Learn, Flask REST API, and Streamlit.

---

## 🌟 Key Features & Innovations

1. **Honest Quantile Uncertainty Estimation**:
   - Instead of a fixed global percentage error, the system trains Gradient Boosting Quantile Regressors ($\alpha=0.10$ and $\alpha=0.90$) to generate true, heteroscedastic 80% prediction intervals.
   - Distinct properties with high variance receive wider ranges; standard properties receive tighter estimates.

2. **Extrapolation Safeguard (OOD Detection)**:
   - Evaluates input features against training set domain percentiles (e.g. Area $650 - 5,200$ sqft).
   - Flags out-of-distribution inputs with a visible **Extrapolation Alert** to avoid misleading predictions outside the model's bounds.

3. **"Why this price?" Value Driver Breakdown**:
   - Interactive Plotly waterfall chart displaying feature-by-feature dollar contributions (Location tier, Area, Condition, Vintage, Rooms) relative to a market baseline.

4. **Dynamic Price Sensitivity Curve**:
   - Real-time 2D Plotly curve sweeping square footage ($700 - 5,200$ sqft) holding your exact property specs constant, with your current property plotted as a reference anchor.

5. **Interactive Global Dark Map Integration**:
   - Type-to-search any address, locality, or city worldwide with OpenStreetMap geocoding.
   - Built with Plotly Carto Dark Matter matching the deep slate `#0F172A` / `#1E293B` hero card theme.
   - Plots the **Subject Property** (glowing emerald diamond) and the **5 Nearest Comparable Sales** (amber/blue pins) with rich interactive hover cards.

6. **5 Nearest Comparables Engine**:
   - Searches historical sales using Euclidean $k$-Nearest Neighbors on the preprocessed feature space to surface comparable properties.

6. **What-If Scenario Comparator**:
   - Side-by-side comparison tool (House A vs. House B) to immediately quantify the financial impact of renovations or adding bathrooms.

7. **Batch Valuation**:
   - Upload any CSV of property specifications, evaluate in bulk, and export valuations with quantile bounds.

8. **Multi-Currency & Unit Switcher**:
   - Live conversion across USD ($), EUR (€), GBP (£), and INR (₹).
   - Toggle between Square Feet (sqft) and Square Meters ($\text{m}^2$).

9. **Zero-Latency Hybrid Architecture**:
   - Works seamlessly with the Flask REST API, with an instantaneous fallback to the in-process ML pipeline when deployed as a standalone app.

---

## 📁 Project Structure

```
house_price_prediction/
├── data/
│   ├── generate_dataset.py       # Realistic market dataset generator
│   └── house_prices.csv          # Multi-factor housing dataset (2,400 samples)
├── models/
│   ├── train_model.py            # 5-fold CV training pipeline + quantile regressors + kNN
│   ├── house_price_model.joblib  # Serialized model package (pipelines, bounds, kNN)
│   └── metrics.json              # Model evaluation metrics & domain bounds
├── backend/
│   ├── app.py                    # Production-ready Flask REST API (/predict, /batch, /curve)
│   └── test_api.py               # Unit test suite covering ranges, types, and edge cases
├── frontend/
│   └── app.py                    # Streamlit web dashboard with interactive Plotly UI
├── reports/
│   ├── generate_report.py        # Standalone HTML report builder
│   ├── final_report.html         # Comprehensive benchmark report
│   └── assets/                   # Feature importance & diagnostic residual plots
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 📊 Dataset Schema

| Feature Column | Type | Description / Range |
| :--- | :--- | :--- |
| `Area(sqft)` | Numeric | Property square footage (650 – 5,200 sqft) |
| `Rooms` | Numeric | Total number of bedrooms (1 – 6) |
| `Bathrooms` | Numeric | Total number of bathrooms (1 – 5) |
| `Floors` | Numeric | Number of floors (1 – 3) |
| `Location` | Categorical | Neighborhood (`Rural`, `Suburban`, `Urban`, `Downtown`) |
| `YearBuilt` | Numeric | Construction year (1960 – 2024) |
| `Parking` | Numeric | Vehicle parking spaces (0 – 3) |
| `Condition` | Categorical | Physical condition (`Fair`, `Good`, `Excellent`) |
| **`Price`** | Target | Property valuation in USD ($94,000 – $2,298,000) |

---

## 🤖 Model Performance (5-Fold Cross-Validation)

Evaluated across candidate models on an 80/20 train-test split:

| Model Algorithm | Test $R^2$ | 5-Fold CV $R^2$ (Mean $\pm$ Std) | MAE ($) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Hist Gradient Boosting (Selected)** | **0.9772** | **0.9747 $\pm$ 0.0051** | **$34,919.73** | **5.80%** |
| Gradient Boosting | 0.9749 | 0.9730 $\pm$ 0.0068 | $35,823.26 | 5.98% |
| Random Forest | 0.9601 | 0.9505 $\pm$ 0.0136 | $44,307.30 | 7.45% |
| Linear Regression | 0.9203 | 0.9154 $\pm$ 0.0077 | $66,247.37 | 13.52% |
| Ridge Regression | 0.9176 | 0.9116 $\pm$ 0.0097 | $65,951.34 | 12.78% |

*Prediction Interval*: 80% nominal quantile interval with an average width of ~$132,000 depending on property scale.

---

## 🚀 Quickstart & Local Execution

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Pipeline
```bash
python data/generate_dataset.py
python models/train_model.py
```

### 3. Run Automated Unit Tests
```bash
python backend/test_api.py
```

### 4. Start Flask REST API Backend
```bash
python backend/app.py
```
*API available at `http://127.0.0.1:5000`.*

### 5. Launch Streamlit Frontend Application
In a separate terminal:
```bash
streamlit run frontend/app.py
```
*UI opens at `http://localhost:8501`.*

### 6. Generate Standalone HTML Report
```bash
python reports/generate_report.py
```
*Report generated at `reports/final_report.html`.*

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that can access the repository.
3. Select **Create app** and configure:
   - **Repository:** `anchalkushwaha-12/ibmtask`
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`
4. Select **Deploy**. Community Cloud installs dependencies from the root `requirements.txt`.

The Streamlit app includes the saved model and can run predictions without a separate Flask server. The scikit-learn version is pinned to match the version used to save the model.
