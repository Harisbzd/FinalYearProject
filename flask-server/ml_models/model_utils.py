
import os
import joblib
import numpy as np
from joblib import load

def load_model_metrics(model_name, base_path=None):
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(__file__))
    
    # Define file paths based on model name
    model_path = os.path.join(base_path, f"{model_name}_model.joblib")
    accuracy_path = os.path.join(base_path, f"{model_name}_accuracy.txt")
    auc_path = os.path.join(base_path, f"{model_name}_auc.txt")
    sensitivity_path = os.path.join(base_path, f"{model_name}_sensitivity.txt")
    specificity_path = os.path.join(base_path, f"{model_name}_specificity.txt")
    columns_path = os.path.join(base_path, f"{model_name}_columns.npy")
    scaler_path = os.path.join(base_path, f"{model_name}_scaler.joblib")
    
    # Load model
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            model = None
    
    # Load accuracy
    accuracy = None
    if os.path.exists(accuracy_path):
        try:
            with open(accuracy_path) as f:
                accuracy = float(f.read().strip())
        except Exception as e:
            print(f"Error loading {model_name} accuracy: {e}")
            accuracy = None
    
    # Load AUC
    auc = None
    if os.path.exists(auc_path):
        try:
            with open(auc_path) as f:
                auc = float(f.read().strip())
        except Exception as e:
            print(f"Error loading {model_name} AUC: {e}")
            auc = None
    
    # Load sensitivity
    sensitivity = None
    if os.path.exists(sensitivity_path):
        try:
            with open(sensitivity_path) as f:
                sensitivity = float(f.read().strip())
        except Exception as e:
            print(f"Error loading {model_name} sensitivity: {e}")
            sensitivity = None
    
    # Load specificity
    specificity = None
    if os.path.exists(specificity_path):
        try:
            with open(specificity_path) as f:
                specificity = float(f.read().strip())
        except Exception as e:
            print(f"Error loading {model_name} specificity: {e}")
            specificity = None
    
    # Load columns
    columns = None
    if os.path.exists(columns_path):
        try:
            columns = np.load(columns_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {model_name} columns: {e}")
            columns = None
    
    # Load scaler (only for models that use it)
    scaler = None
    if model_name in ['xgboost', 'randomforest']:
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Error loading {model_name} scaler: {e}")
                scaler = None
    
    return model, accuracy, auc, sensitivity, specificity, columns, scaler

def get_model_info(model_name):

    base_path = os.path.dirname(__file__)
    
    info = {
        'name': model_name,
        'model_path': os.path.join(base_path, f"{model_name}_model.joblib"),
        'accuracy_path': os.path.join(base_path, f"{model_name}_accuracy.txt"),
        'auc_path': os.path.join(base_path, f"{model_name}_auc.txt"),
        'sensitivity_path': os.path.join(base_path, f"{model_name}_sensitivity.txt"),
        'specificity_path': os.path.join(base_path, f"{model_name}_specificity.txt"),
        'columns_path': os.path.join(base_path, f"{model_name}_columns.npy"),
        'scaler_path': os.path.join(base_path, f"{model_name}_scaler.joblib"),
        'has_scaler': model_name in ['xgboost', 'randomforest']
    }
    
    return info
