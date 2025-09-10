import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve, balanced_accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Add current directory to path for feature_engineering import
sys.path.append(os.path.dirname(__file__))
from feature_engineering import apply_feature_engineering

# Model saving paths
MODEL_PATH = "xgboost_model.joblib"
ACCURACY_PATH = "xgboost_accuracy.txt"
AUC_PATH = "xgboost_auc.txt"
SENSITIVITY_PATH = "xgboost_sensitivity.txt"
SPECIFICITY_PATH = "xgboost_specificity.txt"
COLUMNS_PATH = "xgboost_columns.npy"

_loaded_model = None
_loaded_accuracy = None
_loaded_auc = None
_loaded_sensitivity = None
_loaded_specificity = None
_loaded_columns = None
_loaded_scaler = None

def load_model():
    global _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns, _loaded_scaler
    if _loaded_model is None:
        _loaded_model = joblib.load(MODEL_PATH)
    if _loaded_accuracy is None:
        if os.path.exists(ACCURACY_PATH):
            with open(ACCURACY_PATH) as f:
                _loaded_accuracy = float(f.read().strip())
        else:
            _loaded_accuracy = None
    if _loaded_auc is None:
        if os.path.exists(AUC_PATH):
            with open(AUC_PATH) as f:
                _loaded_auc = float(f.read().strip())
        else:
            _loaded_auc = None
    if _loaded_sensitivity is None:
        if os.path.exists(SENSITIVITY_PATH):
            with open(SENSITIVITY_PATH) as f:
                _loaded_sensitivity = float(f.read().strip())
        else:
            _loaded_sensitivity = None
    if _loaded_specificity is None:
        if os.path.exists(SPECIFICITY_PATH):
            with open(SPECIFICITY_PATH) as f:
                _loaded_specificity = float(f.read().strip())
        else:
            _loaded_specificity = None
    if _loaded_columns is None:
        if os.path.exists(COLUMNS_PATH):
            _loaded_columns = np.load(COLUMNS_PATH, allow_pickle=True)
        else:
            _loaded_columns = None
    if _loaded_scaler is None:
        if os.path.exists("xgboost_scaler.joblib"):
            _loaded_scaler = joblib.load("xgboost_scaler.joblib")
        else:
            _loaded_scaler = None
            
    return _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns, _loaded_scaler

def predict_xgboost(input_dict):
    model, acc, auc_score, sensitivity, specificity, columns, scaler = load_model()
    input_df = pd.DataFrame([input_dict])
    if columns is not None:
        input_df = input_df.reindex(columns=columns)
    
    # Scale the input
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values
    
    pred = model.predict(input_scaled)[0]
    return int(pred), acc, auc_score, sensitivity, specificity

if __name__ == "__main__":
    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    
    # -----------------------------
    # Load and prepare data
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]
    
    print(f"Original features: {X.shape[1]}")
    X = apply_feature_engineering(X)
    print(f"Total features after engineering: {X.shape[1]}")
    print(f"New features added: {X.shape[1] - 21}")

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------
    # Scale ALL features with StandardScaler 
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -----------------------------
    # Handle class imbalance with SMOTE 
    # -----------------------------
    
    # Use SMOTE to oversample minority class instead of undersampling
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # -----------------------------
    # Define XGBoost model with enhanced parameters
    # -----------------------------
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=1.0,  # Balanced since we use SMOTE
        tree_method='hist',  # Faster training
        enable_categorical=False
    )

    # -----------------------------
    # Fast hyperparameter distribution for quick testing
    # -----------------------------
    param_dist = {
        'n_estimators': [200, 400, 600],  
        'max_depth': [4, 6, 8, 10, 12],
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'min_child_weight': [1, 3, 5, 7, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1, 1.5, 2, 3]
    }

    # -----------------------------
    # Enhanced randomized search with balanced scoring
    # -----------------------------
    # Create custom scorer that balances recall and accuracy
    def balanced_recall_accuracy_scorer(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return 0.6 * recall + 0.4 * accuracy  # 60% recall, 40% accuracy
    
    custom_scorer = make_scorer(balanced_recall_accuracy_scorer, greater_is_better=True)
    
    grid = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=30,  # Reduced for faster execution on MacBook Air
        scoring=custom_scorer,  # Balanced scoring for both sensitivity and accuracy
        cv=2,  # Reduced CV folds for faster execution
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    grid.fit(X_train_res, y_train_res)

    # -----------------------------
    # Results
    # -----------------------------

    # -----------------------------
    # Evaluate on validation set with threshold optimization
    # -----------------------------
    best_model = grid.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    
    # Optimize threshold for better sensitivity
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    # Use optimized threshold for predictions
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    print(f"Optimal threshold for F1 score: {optimal_threshold:.4f}")
    print(f"Default threshold (0.5) vs Optimal threshold ({optimal_threshold:.4f})")

    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    
    # Calculate Sensitivity and Specificity from confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate (Recall for positive class)
    specificity = tn / (tn + fp)  # True Negative Rate
    
    # Print comprehensive classification evaluation metrics
    print("=" * 60)
    print("ENHANCED XGBOOST CLASSIFICATION EVALUATION METRICS")
    print("=" * 60)
    print(f"Validation Accuracy:     {accuracy:.4f}")
    print(f"Balanced Accuracy:      {balanced_acc:.4f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall (Sensitivity):   {recall:.4f}")
    print(f"Specificity:            {specificity:.4f}")
    print(f"F1-Score:               {f1:.4f}")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"Matthews Correlation:   {mcc:.4f}")
    print("=" * 60)
    
    # Confusion Matrix details
    print("\nCONFUSION MATRIX:")
    print(f"True Negatives (TN):    {tn}")
    print(f"False Positives (FP):   {fp}")
    print(f"False Negatives (FN):   {fn}")
    print(f"True Positives (TP):    {tp}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Additional metrics
    print(f"\nADDITIONAL METRICS:")
    print(f"False Positive Rate:    {fp/(fp+tn):.4f}")
    print(f"False Negative Rate:    {fn/(fn+tp):.4f}")
    print(f"Positive Predictive Value: {tp/(tp+fp):.4f}")
    print(f"Negative Predictive Value: {tn/(tn+fn):.4f}")
    
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT:")
    print("=" * 60)
    print(classification_report(y_val, y_pred))
    print("=" * 60)


    # -----------------------------
    # Feature Importance Plot
    # -----------------------------
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - XGBoost')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig('../client/public/xgboost_feature_importance.png')
    plt.close()

    # -----------------------------
    # Calibration Curve
    # -----------------------------
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve (Reliability Diagram) - XGBoost')
    plt.legend()
    plt.grid(True)
    plt.savefig('../client/public/xgboost_calibration_curve.png')
    plt.close()


    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - XGBoost (Validation Set)')
    plt.savefig('../client/public/xgboost_confusion_matrix.png')
    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost')
    plt.legend(loc="lower right")
    plt.savefig('../client/public/xgboost_roc_curve.png')
    plt.close()

    # -----------------------------
    # Save model, columns, and metrics
    # -----------------------------
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, "xgboost_scaler.joblib")
    np.save(COLUMNS_PATH, X.columns.values)
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(accuracy))
    with open(AUC_PATH, "w") as f:
        f.write(str(roc_auc))
    with open(SENSITIVITY_PATH, "w") as f:
        f.write(str(sensitivity))
    with open(SPECIFICITY_PATH, "w") as f:
        f.write(str(specificity))


    # -----------------------------
    # Cross-validation scores
    # -----------------------------
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    
    # Cross-validation with multiple metrics
    cv_accuracy = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='precision')
    cv_recall = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='recall')
    cv_f1 = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='f1')
    cv_roc_auc = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='roc_auc')
    cv_balanced_acc = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='balanced_accuracy')
    
    print(f"CV Accuracy scores:     {cv_accuracy}")
    print(f"Mean CV Accuracy:       {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"CV Balanced Accuracy:   {cv_balanced_acc}")
    print(f"Mean CV Balanced Acc:   {cv_balanced_acc.mean():.4f} (+/- {cv_balanced_acc.std() * 2:.4f})")
    print(f"CV Precision scores:    {cv_precision}")
    print(f"Mean CV Precision:      {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    print(f"CV Recall scores:       {cv_recall}")
    print(f"Mean CV Recall:         {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    print(f"CV F1 scores:           {cv_f1}")
    print(f"Mean CV F1:             {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    print(f"CV ROC-AUC scores:      {cv_roc_auc}")
    print(f"Mean CV ROC-AUC:        {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
    print("=" * 60)
    
    # Best hyperparameters
    print("\nBEST HYPERPARAMETERS:")
    print("=" * 60)
    for param, value in grid.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best CV Score: {grid.best_score_:.4f}")
    print("=" * 60)
