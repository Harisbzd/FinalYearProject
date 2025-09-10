import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Resolve important paths
THIS_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, os.pardir))
CLIENT_PUBLIC_DIR = os.path.join(REPO_ROOT, "client", "public")

# Model saving paths
MODEL_PATH = os.path.join(SERVER_DIR, "randomforest_model.joblib")
ACCURACY_PATH = os.path.join(SERVER_DIR, "randomforest_accuracy.txt")
AUC_PATH = os.path.join(SERVER_DIR, "randomforest_auc.txt")
SENSITIVITY_PATH = os.path.join(SERVER_DIR, "randomforest_sensitivity.txt")
SPECIFICITY_PATH = os.path.join(SERVER_DIR, "randomforest_specificity.txt")
COLUMNS_PATH = os.path.join(SERVER_DIR, "randomforest_columns.npy")
SCALER_PATH = os.path.join(SERVER_DIR, "randomforest_scaler.joblib")

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
        if os.path.exists(MODEL_PATH):
            _loaded_model = joblib.load(MODEL_PATH)
        else:
            _loaded_model = None
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
        if os.path.exists(SCALER_PATH):
            _loaded_scaler = joblib.load(SCALER_PATH)
        else:
            _loaded_scaler = None

    return _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns, _loaded_scaler


def predict_randomforest(input_dict):
    model, acc, auc_score, sensitivity, specificity, columns, scaler = load_model()
    if model is None:
        raise ValueError("Random Forest model not found. Please train the model first.")
    
    input_df = pd.DataFrame([input_dict])
    
    # Reindex to match expected columns
    if columns is not None:
        input_df = input_df.reindex(columns=columns, fill_value=0)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Get prediction
    pred = model.predict(input_scaled)[0]
    
    return int(pred), acc, auc_score, sensitivity, specificity


if __name__ == "__main__":
    # Load dataset
    DATA_PATH = os.path.join(SERVER_DIR, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train_res)
    best_model = grid_search.best_estimator_

    # Predict on validation set
    y_val_pred = best_model.predict(X_val_scaled)
    y_val_prob = best_model.predict_proba(X_val_scaled)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_prob)
    
    # Calculate Sensitivity and Specificity from confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate

    print("Random Forest Model Evaluation Results:")
    print("=" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, digits=4))
    print("\nConfusion Matrix:")
    print(cm)


    # Feature Importance
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    feature_importance_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_feature_importance.png")
    plt.savefig(feature_importance_path, dpi=200)
    print("Feature importance saved to randomforest_feature_importance.png")
    plt.close()

    # Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_val, y_val_prob, n_bins=10
    )

    plt.figure(figsize=(10, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Random Forest")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve - Random Forest')
    plt.legend()
    plt.grid(True)
    calibration_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_calibration_curve.png")
    plt.savefig(calibration_path, dpi=200)
    print("Calibration curve saved to randomforest_calibration_curve.png")
    plt.close()


    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
    plt.yticks([0, 1], ["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    cm_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    from sklearn.metrics import auc as auc_func
    roc_auc_score_val = auc_func(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train_res, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Save classification report
    classification_report_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_classification_report.txt")
    with open(classification_report_path, 'w') as f:
        f.write("Random Forest Model Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(classification_report(y_val, y_val_pred, digits=4))
        f.write(f"\nAccuracy: {acc:.4f}\n")
        f.write(f"AUC Score: {auc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Save model, columns, and metrics
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    np.save(COLUMNS_PATH, X.columns.values)
    with open(ACCURACY_PATH, "w") as f:
        f.write(f"{acc:.6f}")
    with open(AUC_PATH, "w") as f:
        f.write(f"{auc:.6f}")
    with open(SENSITIVITY_PATH, "w") as f:
        f.write(f"{sensitivity:.6f}")
    with open(SPECIFICITY_PATH, "w") as f:
        f.write(f"{specificity:.6f}")

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Saved columns to: {COLUMNS_PATH}")
    print(f"Saved accuracy to: {ACCURACY_PATH}")
    print(f"Saved AUC to: {AUC_PATH}")
    print(f"Saved sensitivity to: {SENSITIVITY_PATH}")
    print(f"Saved specificity to: {SPECIFICITY_PATH}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved ROC curve to: {roc_path}")
    print(f"Saved feature importance to: {feature_importance_path}")
    print(f"Saved calibration curve to: {calibration_path}")
    print(f"Saved classification report to: {classification_report_path}") 