from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and encoders with proper paths
model = joblib.load(os.path.join(base_dir, "best_crop_model.pkl"))
label_encoders = joblib.load(os.path.join(base_dir, "label_encoders.pkl"))

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
    # Use PORT environment variable if available (for Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)