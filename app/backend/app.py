"""
Cryptocurrency Market Behaviour - Full-Stack Flask Backend API
Course: 23CSE351 - Foundations of Data Science Case Study

Endpoints:
- GET  /api/health
- GET  /api/dataset-summary
- GET  /api/cryptocurrencies
- GET  /api/historical-trends?asset=Bitcoin&start_date=2026-02-01&end_date=2026-05-01&limit=200
- GET  /api/eda
- GET  /api/analysis-questions
- GET  /api/analysis/<int:question_id>
- GET  /api/hypothesis-tests
- GET  /api/hypothesis-tests/<int:test_id>
- GET  /api/model-results
- POST /api/predict-volume
- GET  /api/export/<filename>
"""

import os
import sys
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, render_template_string
from flask_cors import CORS

# Ensure backend directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from analysis_engine import CryptoAnalysisEngine

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Locate CSV dataset
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

csv_candidates = [
    os.path.join(REPO_ROOT, "data", "cleaned_Cryptocurrency_Dataset.csv"),
    os.path.join(PROJECT_ROOT, "cleaned_crypto.csv"),
    os.path.join(CURRENT_DIR, "cleaned_crypto.csv"),
    "cleaned_crypto.csv"
]
csv_path = None
for p in csv_candidates:
    if os.path.exists(p):
        csv_path = p
        break

if not csv_path:
    raise FileNotFoundError("Could not find 'cleaned_crypto.csv' in project directory.")

export_dir = os.path.join(PROJECT_ROOT, "tableau_exports")
print(f"Initializing Analysis Engine with CSV: {csv_path}")
engine = CryptoAnalysisEngine(csv_path=csv_path, export_dir=export_dir)


FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# -------------------------------------------------------------------------
# Root / Dashboard / Health
# -------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    # If a browser requests HTML, serve the dashboard frontend
    accept = request.headers.get("Accept", "")
    if "text/html" in accept and os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")

    return jsonify({
        "project": "Cryptocurrency Market Behaviour Case Study API",
        "course": "23CSE351 - Foundations of Data Science",
        "status": "online",
        "dashboard_url": "/dashboard",
        "documentation": {
            "health": "/api/health",
            "dataset_summary": "/api/dataset-summary",
            "cryptocurrencies": "/api/cryptocurrencies",
            "historical_trends": "/api/historical-trends?asset=Bitcoin",
            "eda": "/api/eda",
            "analysis_questions": "/api/analysis-questions",
            "analysis_detail": "/api/analysis/1",
            "hypothesis_tests": "/api/hypothesis-tests",
            "model_results": "/api/model-results",
            "predict_volume": "/api/predict-volume (POST)",
            "export_csv": "/api/export/daily-panel"
        }
    })

@app.route("/dashboard", methods=["GET"])
def dashboard():
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return "Dashboard index.html not found.", 404

@app.route("/css/<path:filename>", methods=["GET"])
def serve_css(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "css"), filename)

@app.route("/js/<path:filename>", methods=["GET"])
def serve_js(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "js"), filename)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engine_initialized": engine.is_initialized,
        "dataset_records": engine.summary_stats["overview"]["cleaned_records"],
        "distinct_assets": engine.summary_stats["overview"]["distinct_assets"],
        "questions_count": len(engine.questions_data),
        "hypothesis_tests_count": len(engine.hypothesis_tests)
    })


# -------------------------------------------------------------------------
# Dataset Overview & Data Quality
# -------------------------------------------------------------------------
@app.route("/api/dataset-summary", methods=["GET"])
def dataset_summary():
    try:
        return jsonify({
            "status": "success",
            "data": engine.summary_stats["overview"],
            "hygiene_audit": engine.summary_stats["hygiene_audit"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# Cryptocurrency Asset Selection & Coverage
# -------------------------------------------------------------------------
@app.route("/api/cryptocurrencies", methods=["GET"])
def get_cryptocurrencies():
    try:
        asset_type_filter = request.args.get("type", None)
        assets = engine.get_cryptocurrencies_list()
        
        if asset_type_filter:
            assets = [a for a in assets if a["asset_type"].lower() == asset_type_filter.lower()]
            
        return jsonify({
            "status": "success",
            "total_count": len(assets),
            "data": assets
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# Historical Trends
# -------------------------------------------------------------------------
@app.route("/api/historical-trends", methods=["GET"])
def historical_trends():
    try:
        asset = request.args.get("asset", None)
        if asset and asset.strip().upper() == "ALL":
            asset = None
        start_date = request.args.get("start_date", None)
        end_date = request.args.get("end_date", None)
        limit = int(request.args.get("limit", 500))

        trends = engine.get_historical_trends(
            asset_name=asset,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        return jsonify({
            "status": "success",
            "asset_queried": asset or "ALL",
            "record_count": len(trends),
            "data": trends
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------------
@app.route("/api/eda", methods=["GET"])
def eda():
    try:
        return jsonify({
            "status": "success",
            "distributions": engine.summary_stats["distributions"],
            "correlation_matrix": engine.summary_stats["correlation_matrix"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# 15 Analysis Questions
# -------------------------------------------------------------------------
@app.route("/api/analysis-questions", methods=["GET"])
def list_analysis_questions():
    try:
        questions_summary = []
        for q_id, q_data in sorted(engine.questions_data.items()):
            questions_summary.append({
                "id": q_id,
                "title": q_data["title"],
                "motivation": q_data["motivation"],
                "variables": q_data["variables"],
                "method": q_data["method"],
                "conclusion_brief": q_data["conclusion"]
            })
        return jsonify({
            "status": "success",
            "total_questions": len(questions_summary),
            "data": questions_summary
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/analysis/<int:question_id>", methods=["GET"])
def get_analysis_question(question_id):
    try:
        if question_id not in engine.questions_data:
            return jsonify({
                "status": "error",
                "message": f"Question {question_id} not found. Must be between 1 and 15."
            }), 404

        return jsonify({
            "status": "success",
            "data": engine.questions_data[question_id]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# Statistical Hypothesis Testing
# -------------------------------------------------------------------------
@app.route("/api/hypothesis-tests", methods=["GET"])
def list_hypothesis_tests():
    try:
        return jsonify({
            "status": "success",
            "total_tests": len(engine.hypothesis_tests),
            "data": engine.hypothesis_tests
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/hypothesis-tests/<int:test_id>", methods=["GET"])
def get_hypothesis_test(test_id):
    try:
        test = next((t for t in engine.hypothesis_tests if t["test_id"] == test_id), None)
        if not test:
            return jsonify({
                "status": "error",
                "message": f"Hypothesis Test {test_id} not found. Must be between 1 and 7."
            }), 404

        return jsonify({
            "status": "success",
            "data": test
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# Machine Learning Regression & Volume Prediction
# -------------------------------------------------------------------------
@app.route("/api/model-results", methods=["GET"])
def model_results():
    try:
        return jsonify({
            "status": "success",
            "data": engine.model_artifacts
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict-volume", methods=["POST"])
def predict_volume():
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON body. Required fields: market_cap, chg_24h, chg_7d, asset_type, is_weekend."
            }), 400

        market_cap = payload.get("market_cap")
        if market_cap is None:
            return jsonify({"status": "error", "message": "Field 'market_cap' (in USD) is required."}), 400

        chg_24h = payload.get("chg_24h", 0.0)
        chg_7d = payload.get("chg_7d", 0.0)
        asset_type = payload.get("asset_type", "Volatile")
        is_weekend = payload.get("is_weekend", False)

        result = engine.predict_volume(
            market_cap=market_cap,
            chg_24h=chg_24h,
            chg_7d=chg_7d,
            asset_type=asset_type,
            is_weekend=is_weekend
        )

        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# Tableau / Power BI Data Exports
# -------------------------------------------------------------------------
@app.route("/api/export/<dataset_name>", methods=["GET"])
def export_dataset(dataset_name):
    try:
        file_map = {
            "daily-panel": "tableau_crypto_daily_panel.csv",
            "summary-by-asset": "tableau_crypto_summary_by_asset.csv",
            "hypothesis-tests": "tableau_hypothesis_tests.csv",
            "ml-predictions": "tableau_ml_predictions.csv"
        }

        # Also support raw .csv filename lookup
        clean_name = dataset_name.lower().replace(".csv", "")
        filename = file_map.get(clean_name, None)
        
        if not filename:
            # Check if direct filename in export directory
            if os.path.exists(os.path.join(engine.export_dir, f"{dataset_name}.csv")):
                filename = f"{dataset_name}.csv"
            elif os.path.exists(os.path.join(engine.export_dir, dataset_name)):
                filename = dataset_name
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Export dataset '{dataset_name}' not found. Available exports: {list(file_map.keys())}"
                }), 404

        return send_from_directory(
            directory=engine.export_dir,
            path=filename,
            as_attachment=True,
            mimetype="text/csv"
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -------------------------------------------------------------------------
# Application Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================================")
    print(f" Cryptocurrency Market Behaviour Case Study API Server")
    print(f" Course: 23CSE351 - Foundations of Data Science")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f" Health check: http://127.0.0.1:{port}/api/health")
    print(f" Tableau Exports: {engine.export_dir}")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
