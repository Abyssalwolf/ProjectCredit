from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('creditnetxai_model.h5')

# Load the scaler and feature columns from training
scaler = joblib.load('standard_scaler.pkl')
feature_columns = pd.read_csv('feature_columns.csv').iloc[:, 0].tolist()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as a dictionary
        data = request.json
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure categorical columns are strings (to match one-hot encoding)
        categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
        input_df[categorical_cols] = input_df[categorical_cols].astype(str)
        
        # One-hot encode categorical variables (replicate training preprocessing)
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Add missing dummy columns (if any)
        missing_cols = set(feature_columns) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0  # Add missing columns with 0s
        
        # Reorder columns to match training data order
        input_encoded = input_encoded[feature_columns]
        
        # Scale numerical features
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = float(prediction[0][0])
        result = 'Default' if probability > 0.5 else 'No Default'
        
        return jsonify({
            'prediction': result,
            'probability': probability,
            'shap_explanation': "..."  # Add SHAP values if needed
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)