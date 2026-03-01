"""
Personal Loan Predictor - Flask API
"""

from flask import Flask, request, jsonify
import pickle
import json
import numpy as np
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Auto-train if model doesn't exist
model_path = os.path.join(BASE_DIR, "model.pkl")
meta_path = os.path.join(BASE_DIR, "model_metadata.json")

if not os.path.exists(model_path):
    print("model.pkl not found — training now...")
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_model.py")], check=True, cwd=BASE_DIR)

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(meta_path) as f:
    metadata = json.load(f)

EDUCATION_MAP = {"High School": 0, "Some College": 1, "Bachelor's": 2, "Graduate": 3}
HOME_MAP = {"Renting": 0, "Own": 1, "Mortgage": 2}

@app.route("/")
def index():
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html"}

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        features = np.array([[
            int(data["age"]),
            int(data["income"]),
            int(data["employment_years"]),
            int(data["credit_score"]),
            int(data["existing_loans"]),
            int(data["loan_amount"]),
            int(data["loan_term"]),
            float(data["debt_to_income"]),
            EDUCATION_MAP.get(data["education"], 2),
            HOME_MAP.get(data["home_ownership"], 0),
        ]])

        prob = model.predict_proba(features)[0][1]
        approved = bool(prob >= 0.5)

        if prob >= 0.80:
            risk, risk_color = "Low Risk", "green"
        elif prob >= 0.60:
            risk, risk_color = "Moderate Risk", "yellow"
        elif prob >= 0.40:
            risk, risk_color = "High Risk", "orange"
        else:
            risk, risk_color = "Very High Risk", "red"

        suggested_rate = round(5.0 + (1 - prob) * 20, 2)

        advice = []
        if int(data["credit_score"]) < 650:
            advice.append("Improve your credit score to boost approval chances.")
        if float(data["debt_to_income"]) > 0.4:
            advice.append("Reduce your debt-to-income ratio below 40%.")
        if int(data["employment_years"]) < 2:
            advice.append("Longer employment history strengthens your application.")
        if not advice:
            advice.append("Your profile looks strong — maintain your financial health!")

        return jsonify({
            "approved": approved,
            "probability": round(float(prob), 4),
            "risk": risk,
            "risk_color": risk_color,
            "suggested_rate": suggested_rate,
            "advice": advice,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/metadata")
def get_metadata():
    return jsonify(metadata)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
