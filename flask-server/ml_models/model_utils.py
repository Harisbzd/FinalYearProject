
import os
import joblib
import numpy as np
from joblib import load
from xgboost import XGBClassifier

class XGBoostCompatibilityWrapper:
    """Wrapper to handle XGBoost compatibility issues with older saved models"""
    
    def __init__(self, model):
        self.model = model
        # Fix XGBoost compatibility issues in the model
        self._fix_xgboost_compatibility()
    
    def _fix_xgboost_compatibility(self):
        """Fix XGBoost compatibility issues in the model"""
        # If it's a pipeline, fix the XGBClassifier in the last step
        if hasattr(self.model, 'steps'):
            for name, step in self.model.steps:
                if hasattr(step, '__class__') and 'XGBClassifier' in str(step.__class__):
                    self._fix_xgboost_classifier(step)
        # If it's directly an XGBClassifier, fix it
        elif hasattr(self.model, '__class__') and 'XGBClassifier' in str(self.model.__class__):
            self._fix_xgboost_classifier(self.model)
    
    def _fix_xgboost_classifier(self, classifier):
        """Fix a specific XGBClassifier instance"""
        # Remove deprecated attributes
        if hasattr(classifier, 'use_label_encoder'):
            delattr(classifier, 'use_label_encoder')
        
        # Override get_params to filter out deprecated parameters
        original_get_params = classifier.get_params
        
        def safe_get_params(deep=True):
            params = original_get_params(deep=deep)
            if 'use_label_encoder' in params:
                del params['use_label_encoder']
            return params
        
        classifier.get_params = safe_get_params
        
        # Override get_xgb_params if it exists
        if hasattr(classifier, 'get_xgb_params'):
            original_get_xgb_params = classifier.get_xgb_params
            
            def safe_get_xgb_params():
                params = original_get_xgb_params()
                if 'use_label_encoder' in params:
                    del params['use_label_encoder']
                return params
            
            classifier.get_xgb_params = safe_get_xgb_params
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped model
        return getattr(self.model, name)

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
            # Note: XGBoost compatibility wrapper removed as models are now retrained with current version
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
