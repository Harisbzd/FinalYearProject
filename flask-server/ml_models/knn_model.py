import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    balanced_accuracy_score,
    make_scorer,
    brier_score_loss
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import sys

# Add current directory to path for feature_engineering import
sys.path.append(os.path.dirname(__file__))
from feature_engineering import apply_feature_engineering

# Resolve important paths
THIS_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, os.pardir))
CLIENT_PUBLIC_DIR = os.path.join(REPO_ROOT, "client", "public")

# Model saving paths
MODEL_PATH = os.path.join(SERVER_DIR, "knn_model.joblib")
ACCURACY_PATH = os.path.join(SERVER_DIR, "knn_accuracy.txt")
AUC_PATH = os.path.join(SERVER_DIR, "knn_auc.txt")
SENSITIVITY_PATH = os.path.join(SERVER_DIR, "knn_sensitivity.txt")
SPECIFICITY_PATH = os.path.join(SERVER_DIR, "knn_specificity.txt")
COLUMNS_PATH = os.path.join(SERVER_DIR, "knn_columns.npy")

_loaded_model = None
_loaded_accuracy = None
_loaded_auc = None
_loaded_sensitivity = None
_loaded_specificity = None
_loaded_columns = None

# Feature engineering is now imported from shared module

def load_model():
    global _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns
    if _loaded_model is None:
        if os.path.exists(MODEL_PATH):
            _loaded_model = load(MODEL_PATH)
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
    return _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_sensitivity, _loaded_specificity, _loaded_columns

def predict_knn(input_dict):
    model, acc, auc_score, sensitivity, specificity, columns = load_model()
    if model is None:
        raise ValueError("KNN model not found. Please train the model first.")
    
    input_df = pd.DataFrame([input_dict])
    
    # Reindex to match expected columns
    if columns is not None:
        input_df = input_df.reindex(columns=columns, fill_value=0)
    
    # Apply the same feature engineering as training
    input_df = apply_feature_engineering(input_df)
    
    # Pipeline handles scaling automatically
    pred_proba = model.predict_proba(input_df)[:, 1]
    
    # Apply threshold (0.5 for balanced classification)
    pred = int(pred_proba >= 0.5)
    
    return pred, acc, auc_score, sensitivity, specificity

if __name__ == "__main__":
    # Load dataset
    DATA_PATH = os.path.join(SERVER_DIR, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(DATA_PATH)

    # Load and prepare data
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    
    print(f"Original features: {X.shape[1]}")
    X = apply_feature_engineering(X)
    print(f"Total features after engineering: {X.shape[1]}")
    print(f"New features added: {X.shape[1] - 21}")

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Create pipeline with StandardScaler and KNeighborsClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])
    param_dist = {
        'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
        'classifier__p': [1, 2],  # For minkowski distance
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Custom scorer that balances recall and accuracy
    def balanced_recall_accuracy_scorer(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return 0.6 * recall + 0.4 * accuracy
    
    custom_scorer = make_scorer(balanced_recall_accuracy_scorer, greater_is_better=True)

    # Hyperparameter tuning with RandomizedSearchCV
    grid = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring=custom_scorer,
        cv=3,
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
        method='isotonic',
        cv=3,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train_res, y_train_res)
    print("Calibration completed!")

    # Evaluate with threshold optimization
    best_model = grid.best_estimator_
    
    y_pred_proba_uncalibrated = best_model.predict_proba(X_val)[:, 1]
    y_pred_proba_calibrated = calibrated_model.predict_proba(X_val)[:, 1]
    y_pred_proba = y_pred_proba_calibrated
    
    # Optimize threshold for better sensitivity
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    y_val_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Calculate Brier Score for calibration comparison
    brier_uncalibrated = brier_score_loss(y_val, y_pred_proba_uncalibrated)
    brier_calibrated = brier_score_loss(y_val, y_pred_proba_calibrated)
    
    print(f"Brier Score (Uncalibrated): {brier_uncalibrated:.4f}")
    print(f"Brier Score (Calibrated):   {brier_calibrated:.4f}")
    print(f"Improvement: {((brier_uncalibrated - brier_calibrated) / brier_uncalibrated * 100):.2f}%")

    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    mcc = matthews_corrcoef(y_val, y_val_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
    
    # Calculate Sensitivity and Specificity from confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Print evaluation metrics
    print("\nKNN CLASSIFICATION EVALUATION METRICS")
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
    print(classification_report(y_val, y_val_pred))

    # Confusion matrix figure
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("KNN Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
    plt.yticks([0, 1], ["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    cm_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC curve figure
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("KNN ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()


    # Feature Importance Plot (variance-based for KNN)
    X_train_scaled = best_model.named_steps['scaler'].transform(X_train_res)
    feature_variance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.var(X_train_scaled, axis=0)
    })
    feature_variance = feature_variance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_variance, x='importance', y='feature')
    plt.title('Feature Variance - KNN')
    plt.xlabel('Feature Variance Score')
    plt.tight_layout()
    feature_importance_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_feature_importance.png")
    plt.savefig(feature_importance_path)
    print("Feature importance plot saved")
    plt.close()

    # Calibration Curve
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='KNN')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve - KNN')
    plt.legend()
    plt.grid(True)
    calibration_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_calibration_curve.png")
    plt.savefig(calibration_path)
    print("Calibration curve saved")
    plt.close()


    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train_res, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Save classification report
    classification_report_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_classification_report.txt")
    with open(classification_report_path, 'w') as f:
        f.write("KNN Model Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(classification_report(y_val, y_val_pred, digits=4))
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"AUC Score: {roc_auc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Enhanced Cross-validation scores
    print("\nCROSS-VALIDATION RESULTS")
    print("=" * 50)
    
    # Cross-validation with multiple metrics
    cv_accuracy = cross_val_score(calibrated_model, X_train_res, y_train_res, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(calibrated_model, X_train_res, y_train_res, cv=5, scoring='precision')
    cv_recall = cross_val_score(calibrated_model, X_train_res, y_train_res, cv=5, scoring='recall')
    cv_f1 = cross_val_score(calibrated_model, X_train_res, y_train_res, cv=5, scoring='f1')
    cv_roc_auc = cross_val_score(calibrated_model, X_train_res, y_train_res, cv=5, scoring='roc_auc')
    cv_balanced_acc = cross_val_score(calibrated_model, X_train_res, y_train_res, cv=5, scoring='balanced_accuracy')
    
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

    # Save model and metrics
    dump(calibrated_model, MODEL_PATH)
    np.save(COLUMNS_PATH, X.columns.to_numpy())
    with open(ACCURACY_PATH, "w") as f:
        f.write(f"{accuracy:.6f}")
    with open(AUC_PATH, "w") as f:
        f.write(f"{roc_auc:.6f}")
    with open(SENSITIVITY_PATH, "w") as f:
        f.write(f"{sensitivity:.6f}")
    with open(SPECIFICITY_PATH, "w") as f:
        f.write(f"{specificity:.6f}")

    print(f"\nModel saved successfully!")
    print(f"Final Metrics - Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
