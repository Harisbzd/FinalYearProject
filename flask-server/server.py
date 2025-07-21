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
    app.logger.info(f"Received data for prediction: {data}")

    # Check if all required fields are present and have valid values
    required_fields = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]

    missing_fields = [field for field in required_fields if field not in data or data[field] is None or data[field] == '']
    if missing_fields:
        return jsonify({
            'prediction': None,
            'message': f'Missing or invalid values for fields: {", ".join(missing_fields)}'
        }), 400

    try:
        # Use the trained logistic regression model for prediction
        prediction, accuracy = predict_logreg(data)
        app.logger.info(f"Model prediction: {prediction}, Accuracy: {accuracy}")
        
        # Use update_one with upsert=True to create or update the prediction
        result = db.predictions.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "input": data,
                    "prediction": prediction
                }
            },
            upsert=True
        )

        app.logger.info(f"MongoDB update result: matched_count={result.matched_count}, modified_count={result.modified_count}, upserted_id={result.upserted_id}")
        
        return jsonify({'prediction': str(prediction), 'accuracy': accuracy})
    except Exception as e:
        app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
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