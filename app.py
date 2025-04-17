from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("best_crop_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Expected features in the same order as training
expected_features = [
    "soil_type", "soil_ph", "N", "P", "K", "Fe", "Zn", "Cu", "Mn", "organic_matter",
    "moisture_capacity", "salinity", "drainage", "temperature", "season", "rainfall",
    "humidity", "sunlight_hours", "frost_risk", "wind_speed", "gdd", "altitude",
    "slope", "water_body", "flood_risk", "variety", "growth_duration", "water_need",
    "pest_risk", "rotation_crop", "irrigation", "fertilizer", "market_demand",
    "price", "labor"
]

@app.route('/')
def home():
    return "✅ Crop Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        input_df = pd.DataFrame([data])

        # Encode categorical features
        for col in label_encoders:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

        input_df = input_df[expected_features]

        prediction = model.predict(input_df)[0]
        predicted_crop = label_encoders['crop'].inverse_transform([prediction])[0]
        return jsonify({"predicted_crop": predicted_crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
