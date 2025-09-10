import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve, balanced_accuracy_score, make_scorer, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add current directory to path for feature_engineering import
sys.path.append(os.path.dirname(__file__))
from feature_engineering import apply_feature_engineering

# Model saving paths
MODEL_PATH = "logreg_model.joblib"
ACCURACY_PATH = "logreg_accuracy.txt"
AUC_PATH = "logreg_auc.txt"
SENSITIVITY_PATH = "logreg_sensitivity.txt"
SPECIFICITY_PATH = "logreg_specificity.txt"
COLUMNS_PATH = "logreg_columns.npy"

_loaded_model = None
_loaded_accuracy = None
_loaded_auc = None
_loaded_sensitivity = None
_loaded_specificity = None
_loaded_columns = None

def load_model():
    global _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns
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
    return _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns

def predict_logreg(input_dict):
    model, acc, auc_score, sensitivity, specificity, columns = load_model()
    input_df = pd.DataFrame([input_dict])
    if columns is not None:
        input_df = input_df.reindex(columns=columns)
    
    # Apply the same feature engineering as training
    input_df = apply_feature_engineering(input_df)
    
    # Pipeline handles scaling automatically
    pred = model.predict(input_df)[0]
    return int(pred), acc, auc_score, sensitivity, specificity

if __name__ == "__main__":
    # Load dataset
    DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(DATA_PATH)

    # Load and prepare data
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    
    print(f"Original features: {X.shape[1]}")
    X = apply_feature_engineering(X)
    print(f"Total features after engineering: {X.shape[1]}")
    print(f"New features added: {X.shape[1] - 21}")

    # -----------------------------
    # Train-validation split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Create pipeline with StandardScaler and LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, solver='saga', max_iter=3000))
    ])

    # Hyperparameter tuning with proper solver-penalty combinations
    param_dist = [
        {
            'classifier__penalty': ['l1'],
            'classifier__solver': ['saga'],
            'classifier__C': [0.01, 0.1, 1, 10],  # Reduced from 10 to 4 values
            'classifier__class_weight': [None, 'balanced', {0: 1, 1: 2}],  # Reduced from 4 to 3
            'classifier__max_iter': [2000]  # Single value for faster training
        },
        {
            'classifier__penalty': ['l2'],
            'classifier__solver': ['saga'],
            'classifier__C': [0.01, 0.1, 1, 10],  # Reduced from 10 to 4 values
            'classifier__class_weight': [None, 'balanced', {0: 1, 1: 2}],  # Reduced from 4 to 3
            'classifier__max_iter': [2000]  # Single value for faster training
        },
        {
            'classifier__penalty': ['elasticnet'],
            'classifier__solver': ['saga'],
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__l1_ratio': [0.3, 0.5, 0.7],
            'classifier__class_weight': [None, 'balanced', {0: 1, 1: 2}],
            'classifier__max_iter': [2000]
        }
    ]

    # Create custom scorer that balances recall and accuracy
    def balanced_recall_accuracy_scorer(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return 0.6 * recall + 0.4 * accuracy
    
    custom_scorer = make_scorer(balanced_recall_accuracy_scorer, greater_is_better=True)

    grid = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring=custom_scorer,
        cv=2,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    grid.fit(X_train_res, y_train_res)

    # Apply probability calibration
    print("\nApplying probability calibration...")
    best_base_model = grid.best_estimator_
    
    calibrated_model = CalibratedClassifierCV(
        best_base_model, 
        method='sigmoid',
        cv=3,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train_res, y_train_res)
    print("Calibration completed!")

    # Evaluate with threshold optimization
    best_model = grid.best_estimator_
    
    # Get probabilities from both uncalibrated and calibrated models
    y_pred_proba_uncalibrated = best_model.predict_proba(X_val)[:, 1]
    y_pred_proba_calibrated = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Use calibrated probabilities for evaluation
    y_pred_proba = y_pred_proba_calibrated
    
    # Optimize threshold for better sensitivity
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    # Use optimized threshold for predictions
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Calculate Brier Score for calibration comparison
    brier_uncalibrated = brier_score_loss(y_val, y_pred_proba_uncalibrated)
    brier_calibrated = brier_score_loss(y_val, y_pred_proba_calibrated)
    
    print(f"Brier Score (Uncalibrated): {brier_uncalibrated:.4f}")
    print(f"Brier Score (Calibrated):   {brier_calibrated:.4f}")
    print(f"Improvement: {((brier_uncalibrated - brier_calibrated) / brier_uncalibrated * 100):.2f}%")

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
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Print evaluation metrics
    print("\nLOGISTIC REGRESSION CLASSIFICATION EVALUATION METRICS")
    print("=" * 50)
    print(f"Validation Accuracy:     {accuracy:.4f}")
    print(f"Balanced Accuracy:      {balanced_acc:.4f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall (Sensitivity):   {recall:.4f}")
    print(f"Specificity:            {specificity:.4f}")
    print(f"F1-Score:               {f1:.4f}")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"Matthews Correlation:   {mcc:.4f}")
    print("=" * 50)
    
    # Confusion Matrix details
    print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Positive Rate:    {fp/(fp+tn):.4f}")
    print(f"False Negative Rate:    {fn/(fn+tp):.4f}")
    print(f"Positive Predictive Value: {tp/(tp+fp):.4f}")
    print(f"Negative Predictive Value: {tn/(tn+fn):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Feature Importance Plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.named_steps['classifier'].coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - Logistic Regression')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig('../client/public/logreg_feature_importance.png')
    plt.close()

    # Calibration Curve Comparison
    plt.figure(figsize=(12, 8))
    
    # Uncalibrated model
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_val, y_pred_proba_uncalibrated, n_bins=10)
    plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated Logistic Regression', linewidth=2)
    
    # Calibrated model
    prob_true_cal, prob_pred_cal = calibration_curve(y_val, y_pred_proba_calibrated, n_bins=10)
    plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Calibrated Logistic Regression', linewidth=2)
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfectly Calibrated', linewidth=2)
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve Comparison - Logistic Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Add Brier scores to the plot
    plt.text(0.05, 0.95, f'Uncalibrated Brier Score: {brier_uncalibrated:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(0.05, 0.90, f'Calibrated Brier Score: {brier_calibrated:.4f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../client/public/logreg_calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.savefig('../client/public/logreg_confusion_matrix.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig('../client/public/logreg_roc_curve.png')
    plt.close()

    # Save model and metrics
    joblib.dump(calibrated_model, MODEL_PATH)
    np.save(COLUMNS_PATH, X.columns.values)
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(accuracy))
    with open(AUC_PATH, "w") as f:
        f.write(str(roc_auc))
    with open(SENSITIVITY_PATH, "w") as f:
        f.write(str(sensitivity))
    with open(SPECIFICITY_PATH, "w") as f:
        f.write(str(specificity))

    # Cross-validation scores
    print("\nCROSS-VALIDATION RESULTS")
    print("=" * 50)
    
    cv_accuracy = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='precision')
    cv_recall = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='recall')
    cv_f1 = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='f1')
    cv_roc_auc = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='roc_auc')
    cv_balanced_acc = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='balanced_accuracy')
    
    print(f"CV Accuracy:       {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"CV Balanced Acc:   {cv_balanced_acc.mean():.4f} (+/- {cv_balanced_acc.std() * 2:.4f})")
    print(f"CV Precision:      {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    print(f"CV Recall:         {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    print(f"CV F1:             {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    print(f"CV ROC-AUC:        {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
    print("=" * 50)
    
    # Best hyperparameters
    print("\nBEST HYPERPARAMETERS:")
    print("=" * 50)
    for param, value in grid.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best CV Score: {grid.best_score_:.4f}")
    print("=" * 50)
