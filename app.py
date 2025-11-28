from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ------------------- Load model and encoders -------------------
try:
    model = joblib.load("best_oa_drug_response_model.pkl")
    feature_names = joblib.load("model_features.pkl")
    
    # Load all label encoders
    encoders = {}
    encoder_columns = ["Gender", "Drug_Type", "Dosage_Level", "Activity_Level", 
                      "Smoking_Status", "Alcohol_Consumption", "Response"]
    
    for col in encoder_columns:
        encoders[col] = joblib.load(f"{col.lower()}_label_encoder.pkl")
    
    print("✓ Models and encoders loaded successfully!")
    
except Exception as e:
    print(f"✗ Error loading models: {e}")
    model = None
    encoders = {}
    feature_names = []

# ------------------- Routes -------------------

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/get_options', methods=['GET'])
def get_options():
    """Return encoder options for dropdowns"""
    try:
        options = {}
        
        # Get classes from each encoder
        for key in ["Gender", "Drug_Type", "Dosage_Level", "Activity_Level", 
                   "Smoking_Status", "Alcohol_Consumption"]:
            if key in encoders:
                options[key] = encoders[key].classes_.tolist()
        
        return jsonify(options)
    
    except Exception as e:
        print(f"Error getting options: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'bmi', 'severity', 'duration', 'crp', 
                          'esr', 'drug', 'dosage', 'treat_months', 'activity', 
                          'diet_score', 'smoke', 'alcohol']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Transform categorical variables using label encoders
        gender_encoded = encoders["Gender"].transform([data['gender']])[0]
        drug_encoded = encoders["Drug_Type"].transform([data['drug']])[0]
        dosage_encoded = encoders["Dosage_Level"].transform([data['dosage']])[0]
        activity_encoded = encoders["Activity_Level"].transform([data['activity']])[0]
        smoke_encoded = encoders["Smoking_Status"].transform([data['smoke']])[0]
        alcohol_encoded = encoders["Alcohol_Consumption"].transform([data['alcohol']])[0]
        
        # Prepare input dataframe with exact feature order
        input_df = pd.DataFrame([[
            data['age'],
            gender_encoded,
            data['bmi'],
            data['severity'],
            data['duration'],
            data['crp'],
            data['esr'],
            drug_encoded,
            dosage_encoded,
            data['treat_months'],
            activity_encoded,
            data['diet_score'],
            smoke_encoded,
            alcohol_encoded
        ]], columns=feature_names)
        
        # Make prediction
        pred_numeric = model.predict(input_df)[0]
        pred_label = encoders["Response"].inverse_transform([pred_numeric])[0]
        
        # Get probabilities
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            class_labels = encoders["Response"].inverse_transform(range(len(probs)))
            
            for label, prob in zip(class_labels, probs):
                probabilities[label] = float(prob)
        else:
            # If model doesn't support probability, return 100% for predicted class
            probabilities[pred_label] = 1.0
        
        # Prepare response
        response = {
            'predicted_response': pred_label,
            'probabilities': probabilities,
            'input_data': {
                'age': data['age'],
                'gender': data['gender'],
                'bmi': data['bmi'],
                'severity': data['severity'],
                'duration': data['duration'],
                'crp': data['crp'],
                'esr': data['esr'],
                'drug': data['drug'],
                'dosage': data['dosage'],
                'treat_months': data['treat_months'],
                'activity': data['activity'],
                'diet_score': data['diet_score'],
                'smoke': data['smoke'],
                'alcohol': data['alcohol']
            }
        }
        
        return jsonify(response)
    
    except KeyError as e:
        print(f"Key error in prediction: {e}")
        return jsonify({'error': f'Invalid value for field: {e}'}), 400
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    encoders_status = f"{len(encoders)} encoders loaded"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'encoders_status': encoders_status,
        'features_count': len(feature_names)
    })

# ------------------- Error Handlers -------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ------------------- Run App -------------------

if __name__ == '__main__':
    # Check if static and templates folders exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("✓ Created 'templates' folder")
    
    if not os.path.exists('static'):
        os.makedirs('static')
        print("✓ Created 'static' folder")
    
    print("\n" + "="*60)
    print("Starting Osteoarthritis Drug Response Prediction Server")
    print("="*60)
    print(f"Model Status: {'Loaded ✓' if model else 'Not Loaded ✗'}")
    print(f"Encoders: {len(encoders)} loaded")
    print(f"Features: {len(feature_names)}")
    print("="*60 + "\n")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)