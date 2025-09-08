import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        if os.path.exists("logreg_scaler.joblib"):
            _loaded_scaler = joblib.load("logreg_scaler.joblib")
        else:
            _loaded_scaler = None
            
    return _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns, _loaded_scaler

def predict_logreg(input_dict):
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
    DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    # -----------------------------
    # Train-validation split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Handle class imbalance with SMOTE
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # -----------------------------
    # Scale features
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)

    # -----------------------------
    # Hyperparameter tuning with GridSearchCV
    # -----------------------------
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train_res)

    # -----------------------------
    # Results
    # -----------------------------

    # -----------------------------
    # Evaluate on validation set
    # -----------------------------
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val_scaled)
    y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Calculate Sensitivity and Specificity from confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate (Recall for positive class)
    specificity = tn / (tn + fp)  # True Negative Rate
    
    print("Validation Accuracy:", accuracy)
    print("Sensitivity (Recall):", f"{sensitivity:.4f}")
    print("Specificity:", f"{specificity:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print("\nConfusion Matrix:\n", cm)
    print("\nROC-AUC Score:", roc_auc)

    # -----------------------------
    # Feature Importance Plot
    # -----------------------------
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - Logistic Regression')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig('../client/public/logreg_feature_importance.png')
    plt.close()

    # -----------------------------
    # Calibration Curve
    # -----------------------------
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve (Reliability Diagram) - Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig('../client/public/logreg_calibration_curve.png')
    plt.close()

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Logistic Regression (Validation Set)')
    plt.savefig('../client/public/logreg_confusion_matrix.png')
    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig('../client/public/logreg_roc_curve.png')
    plt.close()

    # -----------------------------
    # Save model, columns, and metrics
    # -----------------------------
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, "logreg_scaler.joblib")
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
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train_res, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")
