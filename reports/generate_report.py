import os
import json
import base64
import pandas as pd

def encode_image_to_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return ""

from pathlib import Path

def generate_final_report():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "house_prices.csv"
    metrics_path = base_dir / "models" / "metrics.json"
    assets_dir = base_dir / "reports" / "assets"
    output_html_path = base_dir / "reports" / "final_report.html"

    # Load metrics
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

    # Load data summary
    df_stats_html = ""
    total_samples = 0
    if data_path.exists():
        df = pd.read_csv(data_path)
        total_samples = len(df)
        stats_df = df.describe().round(2)
        df_stats_html = stats_df.to_html(classes="table table-striped table-bordered", border=0)

    # Base64 images for self-contained HTML
    act_pred_b64 = encode_image_to_base64(str(assets_dir / "actual_vs_predicted.png"))
    res_plot_b64 = encode_image_to_base64(str(assets_dir / "residual_plot.png"))
    feat_imp_b64 = encode_image_to_base64(str(assets_dir / "feature_importance.png"))

    best_model_name = metrics_data.get('best_model', 'Gradient Boosting')
    all_models = metrics_data.get('all_models', {})

    # Build Model Comparison Table HTML
    models_table_rows = ""
    for model_name, m in all_models.items():
        is_best = (model_name == best_model_name)
        highlight_style = "font-weight: bold; background-color: #e6f4ea;" if is_best else ""
        badge = " <span class='badge badge-success'>BEST</span>" if is_best else ""
        models_table_rows += f"""
        <tr style="{highlight_style}">
            <td>{model_name}{badge}</td>
            <td>{m.get('R2', 0):.4f}</td>
            <td>${m.get('MAE', 0):,.2f}</td>
            <td>${m.get('RMSE', 0):,.2f}</td>
            <td>{m.get('MAPE', 0):.2f}%</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction - Project Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1100px;
            margin: 0 auto;
            padding: 30px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 700; }}
        .header p {{ margin-top: 10px; font-size: 1.1em; opacity: 0.9; }}
        
        .card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        h2 {{ color: #1e3c72; border-bottom: 2px solid #eef2f5; padding-bottom: 8px; margin-top: 0; }}
        h3 {{ color: #2a5298; }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}
        .img-container {{
            text-align: center;
            margin: 15px 0;
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2a5298;
            color: white;
        }}
        tr:hover {{ background-color: #f1f5f9; }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            font-size: 0.75em;
            font-weight: bold;
            color: white;
            border-radius: 4px;
        }}
        .badge-success {{ background-color: #28a745; }}
        
        .metric-highlight {{
            display: flex;
            justify-content: space-around;
            text-align: center;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f0f4f8;
            padding: 15px 25px;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
        }}
        .metric-card .value {{ font-size: 1.8em; font-weight: bold; color: #1e3c72; }}
        .metric-card .label {{ font-size: 0.9em; color: #666; text-transform: uppercase; }}
        
        pre {{
            background: #272822;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        code {{ font-family: Consolas, Monaco, 'Courier New', monospace; }}
    </style>
</head>
<body>

    <div class="header">
        <h1>House Price Prediction Report</h1>
        <p>End-to-End Machine Learning System & Valuation Analytics</p>
    </div>

    <div class="card">
        <h2>1. Executive Summary</h2>
        <p>This report documents the design, evaluation, and deployment of a Machine Learning Regression system created to predict residential house prices. Using 8 key property attributes (Area, Rooms, Bathrooms, Floors, Location, Year Built, Parking, Condition), multiple algorithms were trained and benchmarked.</p>
        
        <div class="metric-highlight">
            <div class="metric-card">
                <div class="value">{total_samples:,}</div>
                <div class="label">Dataset Samples</div>
            </div>
            <div class="metric-card">
                <div class="value">{best_model_name}</div>
                <div class="label">Selected Model</div>
            </div>
            <div class="metric-card">
                <div class="value">{all_models.get(best_model_name, {}).get('R2', 0):.4f}</div>
                <div class="label">Best R² Score</div>
            </div>
            <div class="metric-card">
                <div class="value">${all_models.get(best_model_name, {}).get('MAE', 0):,.2f}</div>
                <div class="label">Mean Absolute Error</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>2. Dataset Overview & Summary Statistics</h2>
        <p>The dataset consists of {total_samples} samples across 8 features and 1 target variable (Price).</p>
        {df_stats_html}
    </div>

    <div class="card">
        <h2>3. Model Performance & Evaluation</h2>
        <p>Four regression models were evaluated on an 80/20 train/test split. The <strong>{best_model_name}</strong> regressor achieved the highest predictive accuracy.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Model Architecture</th>
                    <th>R² Score</th>
                    <th>MAE ($)</th>
                    <th>RMSE ($)</th>
                    <th>MAPE (%)</th>
                </tr>
            </thead>
            <tbody>
                {models_table_rows}
            </tbody>
        </table>

        <h3>Visual Diagnostics</h3>
        <div class="grid">
            <div class="img-container">
                <h4>Actual vs. Predicted Prices</h4>
                {"<img src='data:image/png;base64," + act_pred_b64 + "' alt='Actual vs Predicted'>" if act_pred_b64 else "<p>Image not available</p>"}
            </div>
            <div class="img-container">
                <h4>Feature Importance Breakdown</h4>
                {"<img src='data:image/png;base64," + feat_imp_b64 + "' alt='Feature Importance'>" if feat_imp_b64 else "<p>Image not available</p>"}
            </div>
        </div>
        {f"<div class='img-container'><h4>Prediction Error Distribution (Residuals)</h4><img src='data:image/png;base64,{res_plot_b64}' alt='Residual Plot' style='max-width: 600px;'></div>" if res_plot_b64 else ""}
    </div>

    <div class="card">
        <h2>4. System Architecture & Usage Guide</h2>
        <p>The solution is organized into modular directories:</p>
        <ul>
            <li><code>data/</code>: Dataset generation & CSV persistence</li>
            <li><code>models/</code>: Scikit-learn Pipeline training, feature scaling, & joblib export</li>
            <li><code>backend/</code>: Flask REST API serving <code>/predict</code> and <code>/health</code> endpoints</li>
            <li><code>frontend/</code>: Streamlit web interface with real-time prediction sliders</li>
            <li><code>reports/</code>: Standalone HTML analytical report generator</li>
        </ul>

        <h3>Execution Quickstart</h3>
        <pre><code># 1. Install Dependencies
pip install -r requirements.txt

# 2. Train Model Pipeline
python models/train_model.py

# 3. Start Backend Flask API (Port 5000)
python backend/app.py

# 4. Launch Streamlit Frontend Application
streamlit run frontend/app.py</code></pre>
    </div>

    <div style="text-align: center; color: #777; margin-top: 20px; font-size: 0.9em;">
        Generated automatically by House Price Prediction Report System
    </div>

</body>
</html>
"""

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Final project report generated successfully at {output_html_path}")

if __name__ == '__main__':
    generate_final_report()
