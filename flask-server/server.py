from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from auth.routes import auth  # 👈 Auth blueprint
from auth.jwt_utils import token_required  # Import the token_required decorator
from dotenv import load_dotenv
import os
from ml_models.logistic_regression import predict_logreg, load_model

# ✅ Load environment variables from .env file (if present)
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# MongoDB connection from env variable 
app.config["MONGO_URI"] = os.getenv("MONGO_URI") 
mongo = PyMongo(app)
app.mongo = mongo

# Make MongoDB available globally 
db = mongo.db

# Register Auth Blueprint (handles /login and /register)
app.register_blueprint(auth, url_prefix='/auth')

# Default test route
@app.route('/')
def index():
    # db.patients.insert_one({"name": "John", "age": 30})  # Removed dummy insertion
    return "Hello World!"

# Diabetes prediction endpoint
@app.route('/api/predict-diabetes', methods=['POST'])
@token_required
def predict_diabetes():
    data = request.get_json()
    try:
        # Use the trained logistic regression model for prediction
        prediction, accuracy = predict_logreg(data)
        # Save to DB, linked to user
        db.predictions.insert_one({
            "user_id": request.user_id,
            "input": data,
            "prediction": prediction
        })
        return jsonify({'prediction': str(prediction), 'accuracy': accuracy})
    except Exception as e:
        return jsonify({'prediction': None, 'message': str(e)}), 400

# Add endpoint to delete user's prediction
def get_user_prediction_filter():
    return {"user_id": request.user_id}

@app.route('/api/predict-diabetes', methods=['DELETE'])
@token_required
def delete_prediction():
    result = db.predictions.delete_one(get_user_prediction_filter())
    if result.deleted_count == 1:
        return jsonify({'message': 'Prediction deleted.'}), 200
    else:
        return jsonify({'error': 'No prediction found to delete.'}), 404

@app.route('/api/predict-diabetes', methods=['GET'])
@token_required
def get_prediction():
    pred = db.predictions.find_one({"user_id": request.user_id})
    if pred:
        _, accuracy, _ = load_model()
        return jsonify({
            'prediction': pred.get('prediction'),
            'input': pred.get('input'),
            'accuracy': accuracy
        }), 200
    else:
        return jsonify({'error': 'No prediction found.'}), 404

if __name__ == "__main__":
    app.run(debug=True)