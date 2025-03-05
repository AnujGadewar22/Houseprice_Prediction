import os
import logging
from flask import Flask, render_template, request, jsonify
import numpy as np
from model import HousePriceModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Initialize the model
model = HousePriceModel()
model.train()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        features = {
            'bedrooms': float(request.form['bedrooms']),
            'bathrooms': float(request.form['bathrooms']),
            'sqft': float(request.form['sqft']),
            'age': float(request.form['age']),
            'garage': float(request.form['garage'])
        }
        
        # Input validation
        for key, value in features.items():
            if value < 0:
                return jsonify({'error': f'Invalid value for {key}: must be positive'})
            if key in ['bedrooms', 'bathrooms', 'garage'] and value > 10:
                return jsonify({'error': f'Invalid value for {key}: must be less than 10'})
            if key == 'sqft' and value > 10000:
                return jsonify({'error': 'Square footage must be less than 10000'})
            if key == 'age' and value > 200:
                return jsonify({'error': 'House age must be less than 200 years'})

        # Make prediction
        prediction = model.predict(features)
        
        # Return formatted prediction and feature importance
        return jsonify({
            'prediction': f"${prediction:,.2f}",
            'features': model.get_feature_importance()
        })
    
    except ValueError as e:
        return jsonify({'error': 'Please enter valid numbers for all fields'})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'})
