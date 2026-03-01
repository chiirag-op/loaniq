# 🏦 LoanIQ — Personal Loan Predictor

A full-stack ML web app that predicts personal loan approvals using a Random Forest classifier, with a polished interactive dashboard.

---

## 📁 Project Structure

```
loan_predictor/
├── train_model.py       # ML model training script
├── app.py               # Flask API server
├── requirements.txt     # Python dependencies
├── Procfile             # For Heroku/Render deployment
├── model.pkl            # Trained model (generated)
├── model_metadata.json  # Model stats (generated)
└── static/
    └── index.html       # Dashboard UI
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model.pkl + model_metadata.json)
python train_model.py

# 3. Run the app
python app.py

# 4. Open http://localhost:5000
```

---

## 🌐 Deploy to Render (Free)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New Web Service
3. Connect your repo
4. Set:
   - Build Command: `pip install -r requirements.txt && python train_model.py`
   - Start Command: `gunicorn app:app`
5. Deploy! 🎉

## 🌐 Deploy to Heroku

```bash
heroku create your-loaniq-app
git push heroku main
heroku open
```

## 🌐 Deploy to Railway

1. Go to https://railway.app
2. New Project → Deploy from GitHub
3. Add `pip install -r requirements.txt && python train_model.py` as build command
4. Done!

---

## 🤖 Model Details

- **Algorithm**: Random Forest (200 trees, max_depth=8)
- **Features**: Age, Income, Credit Score, Employment Years, Existing Loans, Loan Amount, Loan Term, Debt-to-Income Ratio, Education, Home Ownership
- **Performance**: ~82% accuracy, 0.878 AUC-ROC
- **Training Data**: 5,000 synthetic samples with realistic distributions

---

## 🔌 API Endpoints

### `POST /api/predict`
```json
{
  "age": 35,
  "income": 750000,
  "credit_score": 720,
  "employment_years": 5,
  "existing_loans": 1,
  "loan_amount": 500000,
  "loan_term": 36,
  "debt_to_income": 0.28,
  "education": "Bachelor's",
  "home_ownership": "Mortgage"
}
```

**Response:**
```json
{
  "approved": true,
  "probability": 0.82,
  "risk": "Low Risk",
  "risk_color": "green",
  "suggested_rate": 8.6,
  "advice": ["Your profile looks strong — maintain your financial health!"]
}
```

### `GET /api/metadata`
Returns model performance metrics and feature importances.
