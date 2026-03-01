"""
Personal Loan Predictor - Flask API
Run: python app.py
Then open http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle
import json
import numpy as np
import os

app = Flask(__name__, static_folder="static")

# Load model and metadata
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_metadata.json") as f:
    metadata = json.load(f)

EDUCATION_MAP = {"High School": 0, "Some College": 1, "Bachelor's": 2, "Graduate": 3}
HOME_MAP = {"Renting": 0, "Own": 1, "Mortgage": 2}

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

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

        # Risk band
        if prob >= 0.80:
            risk = "Low Risk"
            risk_color = "green"
        elif prob >= 0.60:
            risk = "Moderate Risk"
            risk_color = "yellow"
        elif prob >= 0.40:
            risk = "High Risk"
            risk_color = "orange"
        else:
            risk = "Very High Risk"
            risk_color = "red"

        # Suggested interest rate
        base_rate = 5.0
        risk_premium = (1 - prob) * 20
        suggested_rate = round(base_rate + risk_premium, 2)

        # Advice
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
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
