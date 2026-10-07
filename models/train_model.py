import os
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import joblib

CURRENT_YEAR = 2026

def engineer_features(df_input):
    """Adds engineered features to dataset."""
    df = df_input.copy()
    if 'YearBuilt' in df.columns:
        df['Age'] = CURRENT_YEAR - df['YearBuilt']
    if 'Bathrooms' in df.columns and 'Rooms' in df.columns:
        df['BathToBedRatio'] = df['Bathrooms'] / df['Rooms'].clip(lower=1)
    if 'Area(sqft)' in df.columns and 'Rooms' in df.columns:
        df['LivingDensity'] = df['Area(sqft)'] / df['Rooms'].clip(lower=1)
    return df

def train_and_evaluate_models():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "house_prices.csv"
    models_dir = base_dir / "models"
    assets_dir = base_dir / "reports" / "assets"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df = engineer_features(df_raw)

    X = df.drop(columns=['Price'])
    y = df['Price']

    # Define feature groups
    num_features = ['Area(sqft)', 'Rooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Parking', 'Age', 'BathToBedRatio', 'LivingDensity']
    cat_features = ['Location', 'Condition']

    # Compute domain bounds for out-of-distribution detection
    feature_bounds = {}
    for col in ['Area(sqft)', 'Rooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Parking']:
        feature_bounds[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'p01': float(df[col].quantile(0.01)),
            'p99': float(df[col].quantile(0.99)),
            'mean': float(df[col].mean()),
            'median': float(df[col].median())
        }
    feature_bounds['Location'] = {
        'allowed': sorted(df['Location'].unique().tolist())
    }
    feature_bounds['Condition'] = {
        'allowed': sorted(df['Condition'].unique().tolist())
    }

    # 2. Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Define candidate models
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingRegressor(max_iter=180, learning_rate=0.08, max_depth=6, random_state=42)
    }

    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None

    print("\n--- Training and Evaluating Candidate Models with 5-Fold Cross-Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, regressor in candidate_models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        # 5-fold Cross-Validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')
        cv_r2_mean = float(cv_scores.mean())
        cv_r2_std = float(cv_scores.std())

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100)

        results[name] = {
            'R2': round(r2, 4),
            'CV_R2_Mean': round(cv_r2_mean, 4),
            'CV_R2_Std': round(cv_r2_std, 4),
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2)
        }

        print(f"[{name}] Test R2: {r2:.4f} | CV R2: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f}) | MAE: ${mae:,.2f} | MAPE: {mape:.2f}%")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n--> Selected Best Model: {best_model_name} (R2 = {best_r2:.4f})")

    # 4. Train Quantile Regressors for Honest Uncertainty Intervals (10th and 90th percentiles)
    print("\n--- Training Quantile Regressors for True Uncertainty Estimation ---")
    lower_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(loss='quantile', alpha=0.10, n_estimators=160, max_depth=4, random_state=42))
    ])
    upper_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(loss='quantile', alpha=0.90, n_estimators=160, max_depth=4, random_state=42))
    ])

    lower_pipeline.fit(X_train, y_train)
    upper_pipeline.fit(X_train, y_train)

    y_lower_test = lower_pipeline.predict(X_test)
    y_upper_test = upper_pipeline.predict(X_test)
    coverage = float(np.mean((y_test >= y_lower_test) & (y_test <= y_upper_test)) * 100)
    avg_interval_width = float(np.mean(y_upper_test - y_lower_test))
    print(f"80% Prediction Interval Empirical Test Coverage: {coverage:.2f}%")
    print(f"Average Interval Width: ${avg_interval_width:,.2f}")

    # 5. Fit Nearest Neighbors for Finding 5 Closest Comparables
    print("\n--- Fitting Nearest Neighbors for Comparable Properties Discovery ---")
    X_train_transformed = preprocessor.transform(X_train)
    nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_model.fit(X_train_transformed)

    # Keep a compact sample of training properties for fast comparable retrieval
    train_lookup_df = X_train.copy()
    train_lookup_df['Price'] = y_train.values

    # 6. Extract Feature Importances
    feature_names = []
    # Numeric features
    feature_names.extend(num_features)
    # Categorical features one-hot names
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    feature_names.extend(encoded_cat_names)

    importances_dict = {}
    if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
        raw_importances = best_pipeline.named_steps['regressor'].feature_importances_
        importances_dict = {feat: float(imp) for feat, imp in zip(feature_names, raw_importances)}
    elif hasattr(best_pipeline.named_steps['regressor'], 'coef_'):
        raw_coefs = np.abs(best_pipeline.named_steps['regressor'].coef_)
        total_coef = np.sum(raw_coefs) if np.sum(raw_coefs) > 0 else 1.0
        importances_dict = {feat: float(c / total_coef) for feat, imp in zip(feature_names, raw_coefs)}
    else:
        from sklearn.inspection import permutation_importance
        X_test_proc = preprocessor.transform(X_test)
        perm = permutation_importance(best_pipeline.named_steps['regressor'], X_test_proc, y_test, n_repeats=5, random_state=42)
        raw_imp = np.clip(perm.importances_mean, 0, None)
        total_imp = np.sum(raw_imp) if np.sum(raw_imp) > 0 else 1.0
        importances_dict = {feat: float(raw_imp[i] / total_imp) for i, feat in enumerate(feature_names)}

    # 7. Generate Diagnostic Charts
    print("\n--- Generating Diagnostic Visualizations ---")
    y_pred_best = best_pipeline.predict(X_test)
    residuals = y_test - y_pred_best

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Chart 1: Actual vs. Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.5, color='#2563EB', edgecolors='none', s=35)
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#DC2626', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.title(f'Actual vs. Predicted Prices ({best_model_name})', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Actual Sale Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.gca().xaxis.set_major_formatter('${x:,.0f}')
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')
    plt.legend(frameon=True)
    plt.tight_layout()
    act_pred_path = assets_dir / "actual_vs_predicted.png"
    plt.savefig(act_pred_path, dpi=200)
    plt.close()

    # Chart 2: Residual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_best, residuals, alpha=0.5, color='#059669', edgecolors='none', s=35)
    plt.axhline(0, color='#DC2626', linestyle='--', linewidth=2)
    plt.title(f'Residual Analysis ({best_model_name})', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Price ($)', fontsize=12)
    plt.ylabel('Residual / Error ($)', fontsize=12)
    plt.gca().xaxis.set_major_formatter('${x:,.0f}')
    plt.gca().yaxis.set_major_formatter('${x:,.0f}')
    plt.tight_layout()
    res_plot_path = assets_dir / "residual_plot.png"
    plt.savefig(res_plot_path, dpi=200)
    plt.close()

    # Chart 3: Feature Importance
    if importances_dict:
        sorted_imp = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        feats = [x[0] for x in sorted_imp][::-1]
        vals = [x[1] for x in sorted_imp][::-1]

        plt.figure(figsize=(9, 6))
        plt.barh(feats, vals, color='#4F46E5', alpha=0.85)
        plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold', pad=12)
        plt.xlabel('Normalized Importance Score', fontsize=12)
        plt.tight_layout()
        feat_imp_path = assets_dir / "feature_importance.png"
        plt.savefig(feat_imp_path, dpi=200)
        plt.close()

    # 8. Save Metrics JSON
    metrics_summary = {
        'version': '2.0.0',
        'training_timestamp': datetime.datetime.now().isoformat(),
        'dataset_records': len(df),
        'best_model': best_model_name,
        'best_model_metrics': results[best_model_name],
        'quantile_uncertainty': {
            'nominal_coverage_pct': 80.0,
            'empirical_coverage_pct': round(coverage, 2),
            'avg_interval_width': round(avg_interval_width, 2)
        },
        'all_models': results,
        'feature_bounds': feature_bounds,
        'feature_names': feature_names,
        'top_feature_importances': dict(sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:8])
    }

    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved successfully to: {metrics_path}")

    # 9. Save Unified Model Package
    model_package = {
        'version': '2.0.0',
        'created_at': datetime.datetime.now().isoformat(),
        'best_model_name': best_model_name,
        'main_pipeline': best_pipeline,
        'lower_pipeline': lower_pipeline,
        'upper_pipeline': upper_pipeline,
        'preprocessor': preprocessor,
        'nn_model': nn_model,
        'train_lookup_df': train_lookup_df.reset_index(drop=True),
        'feature_bounds': feature_bounds,
        'raw_numeric_features': ['Area(sqft)', 'Rooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Parking'],
        'raw_categorical_features': ['Location', 'Condition'],
        'feature_names': feature_names,
        'feature_importances': importances_dict,
        'baseline_price': float(y_train.median())
    }

    model_joblib_path = models_dir / "house_price_model.joblib"
    joblib.dump(model_package, model_joblib_path)
    print(f"Unified model package serialized to: {model_joblib_path}")

if __name__ == '__main__':
    train_and_evaluate_models()
