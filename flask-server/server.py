from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from auth.routes import auth
from auth.jwt_utils import token_required
from dotenv import load_dotenv
import os
from ml_models.xgboost_model import predict_xgboost, load_model as load_xgboost_model
from ml_models.randomforest_model import predict_randomforest, load_model as load_randomforest_model
from ml_models.knn_model import predict_knn, load_model as load_knn_model
from ml_models.logistic_regression import predict_logreg, load_model as load_logreg_model

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

app.config["MONGO_URI"] = os.getenv("MONGO_URI") 
mongo = PyMongo(app)
app.mongo = mongo
db = mongo.db
app.register_blueprint(auth, url_prefix='/auth')

@app.route('/')
def index():
    return "Hello World!"

@app.route('/api/predict-diabetes', methods=['POST'])
@token_required
def predict_diabetes():
    data = request.get_json()
    app.logger.info(f"Received data for prediction: {data}")

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
        prediction, accuracy, auc_score, sensitivity, specificity = predict_xgboost(data)
        app.logger.info(f"Model prediction: {prediction}, Accuracy: {accuracy}, AUC: {auc_score}, Sensitivity: {sensitivity}, Specificity: {specificity}")
        
        result = db.predictions.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "input": data,
                    "prediction": prediction,
                    "accuracy": accuracy,
                    "auc_score": auc_score,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                }
            },
            upsert=True
        )

        app.logger.info(f"MongoDB update result: matched_count={result.matched_count}, modified_count={result.modified_count}, upserted_id={result.upserted_id}")
        
        return jsonify({
            'prediction': str(prediction), 
            'accuracy': accuracy,
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        })
    except Exception as e:
        app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({'prediction': None, 'message': str(e)}), 400

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
        # Determine which model was used for the prediction and return its metrics
        if pred.get('logreg_prediction') is not None:
            # Logistic Regression was used
            _, accuracy, auc_score, sensitivity, specificity, _ = load_logreg_model()
        elif pred.get('xgboost_prediction') is not None:
            # XGBoost was used
            _, accuracy, auc_score, sensitivity, specificity, _, _ = load_xgboost_model()
        elif pred.get('randomforest_prediction') is not None:
            # Random Forest was used
            _, accuracy, auc_score, sensitivity, specificity, _, _ = load_randomforest_model()
        elif pred.get('knn_prediction') is not None:
            # KNN was used
            _, accuracy, auc_score, sensitivity, specificity, _ = load_knn_model()
        else:
            # Default to XGBoost for backward compatibility
            _, accuracy, auc_score, sensitivity, specificity, _, _ = load_xgboost_model()
        
        return jsonify({
            'prediction': pred.get('prediction'),
            'input': pred.get('input'),
            'accuracy': accuracy,
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        }), 200
    else:
        return jsonify({'error': 'No prediction found.'}), 404

@app.route('/api/predict-diabetes-xgboost', methods=['POST'])
@token_required
def predict_diabetes_xgboost():
    data = request.get_json()
    app.logger.info(f"Received data for XGBoost prediction: {data}")

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
        prediction, accuracy, auc_score, sensitivity, specificity = predict_xgboost(data)
        app.logger.info(f"XGBoost Model prediction: {prediction}, Accuracy: {accuracy}, AUC: {auc_score}, Sensitivity: {sensitivity}, Specificity: {specificity}")
        
        result = db.predictions.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "input": data,
                    "xgboost_prediction": prediction,
                    "xgboost_accuracy": accuracy,
                    "xgboost_auc": auc_score,
                    "xgboost_sensitivity": sensitivity,
                    "xgboost_specificity": specificity
                }
            },
            upsert=True
        )

        app.logger.info(f"MongoDB update result: matched_count={result.matched_count}, modified_count={result.modified_count}, upserted_id={result.upserted_id}")
        
        return jsonify({
            'prediction': str(prediction), 
            'accuracy': accuracy, 
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        })
    except Exception as e:
        app.logger.error(f"An error occurred during XGBoost prediction: {e}", exc_info=True)
        return jsonify({'prediction': None, 'message': str(e)}), 400

@app.route('/api/predict-diabetes-xgboost', methods=['GET'])
@token_required
def get_xgboost_prediction():
    pred = db.predictions.find_one({"user_id": request.user_id})
    if pred:
        _, accuracy, auc_score, sensitivity, specificity, _, _ = load_xgboost_model()
        return jsonify({
            'prediction': pred.get('xgboost_prediction'),
            'input': pred.get('input'),
            'accuracy': accuracy,
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        }), 200
    else:
        return jsonify({'error': 'No XGBoost prediction found.'}), 404

@app.route('/api/predict-diabetes-randomforest', methods=['POST'])
@token_required
def predict_diabetes_randomforest():
    data = request.get_json()
    app.logger.info(f"Received data for Random Forest prediction: {data}")

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
        prediction, accuracy, auc_score, sensitivity, specificity = predict_randomforest(data)
        app.logger.info(f"Random Forest Model prediction: {prediction}, Accuracy: {accuracy}, AUC: {auc_score}, Sensitivity: {sensitivity}, Specificity: {specificity}")
        
        result = db.predictions.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "input": data,
                    "randomforest_prediction": prediction,
                    "randomforest_accuracy": accuracy,
                    "randomforest_auc": auc_score,
                    "randomforest_sensitivity": sensitivity,
                    "randomforest_specificity": specificity
                }
            },
            upsert=True
        )

        app.logger.info(f"MongoDB update result: matched_count={result.matched_count}, modified_count={result.modified_count}, upserted_id={result.upserted_id}")
        
        return jsonify({
            'prediction': str(prediction), 
            'accuracy': accuracy, 
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        })
    except Exception as e:
        app.logger.error(f"An error occurred during Random Forest prediction: {e}", exc_info=True)
        return jsonify({'prediction': None, 'message': str(e)}), 400

@app.route('/api/predict-diabetes-randomforest', methods=['GET'])
@token_required
def get_randomforest_prediction():
    pred = db.predictions.find_one({"user_id": request.user_id})
    if pred:
        _, accuracy, auc_score, sensitivity, specificity, _, _ = load_randomforest_model()
        return jsonify({
            'prediction': pred.get('randomforest_prediction'),
            'input': pred.get('input'),
            'accuracy': accuracy,
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        }), 200
    else:
        return jsonify({'error': 'No Random Forest prediction found.'}), 404

@app.route('/api/predict-diabetes-knn', methods=['POST'])
@token_required
def predict_diabetes_knn():
    data = request.get_json()
    app.logger.info(f"Received data for KNN prediction: {data}")

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
        prediction, accuracy, auc_score, sensitivity, specificity = predict_knn(data)
        app.logger.info(f"KNN Model prediction: {prediction}, Accuracy: {accuracy}, AUC: {auc_score}, Sensitivity: {sensitivity}, Specificity: {specificity}")
        
        result = db.predictions.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "input": data,
                    "knn_prediction": prediction,
                    "knn_accuracy": accuracy,
                    "knn_auc": auc_score,
                    "knn_sensitivity": sensitivity,
                    "knn_specificity": specificity
                }
            },
            upsert=True
        )

        app.logger.info(f"MongoDB update result: matched_count={result.matched_count}, modified_count={result.modified_count}, upserted_id={result.upserted_id}")
        
        return jsonify({
            'prediction': str(prediction), 
            'accuracy': accuracy, 
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        })
    except Exception as e:
        app.logger.error(f"An error occurred during KNN prediction: {e}", exc_info=True)
        return jsonify({'prediction': None, 'message': str(e)}), 400

@app.route('/api/predict-diabetes-knn', methods=['GET'])
@token_required
def get_knn_prediction():
    pred = db.predictions.find_one({"user_id": request.user_id})
    if pred:
        _, accuracy, auc_score, sensitivity, specificity, _ = load_knn_model()
        return jsonify({
            'prediction': pred.get('knn_prediction'),
            'input': pred.get('input'),
            'accuracy': accuracy,
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        }), 200
    else:
        return jsonify({'error': 'No KNN prediction found.'}), 404

@app.route('/api/predict-diabetes-logreg', methods=['POST'])
@token_required
def predict_diabetes_logreg():
    data = request.get_json()
    app.logger.info(f"Received data for Logistic Regression prediction: {data}")

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
        prediction, accuracy, auc_score, sensitivity, specificity = predict_logreg(data)
        app.logger.info(f"Logistic Regression prediction: {prediction}, Accuracy: {accuracy}, AUC: {auc_score}, Sensitivity: {sensitivity}, Specificity: {specificity}")
        
        result = db.predictions.update_one(
            {"user_id": request.user_id},
            {
                "$set": {
                    "input": data,
                    "logreg_prediction": prediction,
                    "logreg_accuracy": accuracy,
                    "logreg_auc": auc_score,
                    "logreg_sensitivity": sensitivity,
                    "logreg_specificity": specificity
                }
            },
            upsert=True
        )
        
        return jsonify({
            'prediction': prediction,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'input': data
        }), 200
    except Exception as e:
        app.logger.error(f"Error in Logistic Regression prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict-diabetes-logreg', methods=['GET'])
@token_required
def get_logreg_prediction():
    pred = db.predictions.find_one({"user_id": request.user_id})
    if pred:
        _, accuracy, auc_score, sensitivity, specificity, _ = load_logreg_model()
        return jsonify({
            'prediction': pred.get('logreg_prediction'),
            'input': pred.get('input'),
            'accuracy': accuracy,
            'auc_score': auc_score,
            'sensitivity': sensitivity,
            'specificity': specificity
        }), 200
    else:
        return jsonify({'error': 'No Logistic Regression prediction found.'}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5001)