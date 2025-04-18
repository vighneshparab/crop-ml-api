from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and label encoders
try:
    model = joblib.load(os.path.join(base_dir, "best_crop_model.pkl"))
    label_encoders = joblib.load(os.path.join(base_dir, "label_encoders.pkl"))
except Exception as e:
    print(f"Error loading model or label encoders: {e}")
    exit(1)

# Updated expected features as per the model
expected_features = [
    'Soil_Type', 'Soil_pH', 'N_Value', 'P_Value', 'K_Value', 'Fe_Value', 'Zn_Value',
    'Cu_Value', 'Mn_Value', 'Organic_Matter', 'Soil_Moisture', 'Soil_Salinity',
    'Soil_Drainage', 'Temperature', 'Season', 'Rainfall', 'Humidity', 'Sunlight',
    'Frost_Risk', 'Wind_Speed', 'Altitude', 'Slope', 'Water_Proximity',
    'Flood_Risk', 'Crop_Variety', 'Growth_Duration', 'Water_Requirements',
    'Pest_Susceptibility', 'Irrigation_Method', 'Fertilizer_Use',
    'Market_Demand', 'Market_Price', 'Labor_Availability'
]

@app.route('/')
def home():
    return "✅ Crop Prediction API is running!"
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Validate input data
        if not all(feature in data for feature in expected_features):
            missing = [feature for feature in expected_features if feature not in data]
            return jsonify({"error": f"Missing features: {', '.join(missing)}"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure correct column order
        input_df = input_df[expected_features]

        # Predict
        prediction = model.predict(input_df)[0]

        return jsonify({"predicted_crop": prediction})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 400

    data = request.json
    try:
        # Validate input data
        if not all(feature in data for feature in expected_features):
            missing = [feature for feature in expected_features if feature not in data]
            return jsonify({"error": f"Missing features: {', '.join(missing)}"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Apply label encoding where needed
        for col in label_encoders:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Ensure correct order
        input_df = input_df[expected_features]

        # Predict
        prediction = model.predict(input_df)[0]

        # Decode prediction
        predicted_crop = label_encoders['crop'].inverse_transform([prediction])[0]

        return jsonify({"predicted_crop": predicted_crop})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
