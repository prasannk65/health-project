# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import traceback
from health_project2 import get_models_and_scalers_from_run

app = Flask(__name__)

# Load (or train then load) models and scalers
diab_model, heart_model, scaler_diab, scaler_heart = get_models_and_scalers_from_run()
print("✅ Models loaded for Flask.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        disease = data.get("disease")
        features = data.get("features")  # list of numbers
        if not isinstance(features, list):
            return jsonify({"error":"features must be a list"}), 400
        x = np.array(features).reshape(1, -1)

        if disease == "diabetes":
            x_s = scaler_diab.transform(x)
            proba = float(diab_model.predict_proba(x_s)[:,1][0]) if hasattr(diab_model, "predict_proba") else float(diab_model.decision_function(x_s)[0])
            label = int(diab_model.predict(x_s)[0])
        elif disease == "heart":
            x_s = scaler_heart.transform(x)
            proba = float(heart_model.predict_proba(x_s)[:,1][0]) if hasattr(heart_model, "predict_proba") else float(heart_model.decision_function(x_s)[0])
            label = int(heart_model.predict(x_s)[0])
        else:
            return jsonify({"error":"invalid disease type"}), 400

        status = "✅ Healthy" if label == 0 else "⚠️ At Risk"
        return jsonify({"probability": proba, "label": label, "status": status})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
