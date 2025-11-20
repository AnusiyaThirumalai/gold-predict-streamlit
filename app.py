from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return "Gold Prediction API is running. Use POST /predict"

lr_model = joblib.load("gold_lr_model.joblib")
rf_model = joblib.load("gold_rf_model.joblib")

def prepare_features(data):
    df = pd.DataFrame([data])
    features = ['USD_INR','USD_lag1','Gold_lag1','Gold_lag2','Gold_roll3','Gold_roll5']
    return df[features]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X = prepare_features(data)
    return jsonify({
        "LinearRegression_Prediction": float(lr_model.predict(X)[0]),
        "RandomForest_Prediction": float(rf_model.predict(X)[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
