import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
    balanced_accuracy_score,
    make_scorer,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump, load
import sys
import xgboost as xgb

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from feature_engineering import apply_feature_engineering
from model_utils import load_model_metrics

# Resolve important paths
THIS_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, os.pardir))
CLIENT_PUBLIC_DIR = os.path.join(REPO_ROOT, "client", "public")

# Model saving paths
MODEL_PATH = os.path.join(SERVER_DIR, "ensemble_model.joblib")
ACCURACY_PATH = os.path.join(SERVER_DIR, "ensemble_accuracy.txt")
AUC_PATH = os.path.join(SERVER_DIR, "ensemble_auc.txt")
SENSITIVITY_PATH = os.path.join(SERVER_DIR, "ensemble_sensitivity.txt")
SPECIFICITY_PATH = os.path.join(SERVER_DIR, "ensemble_specificity.txt")
COLUMNS_PATH = os.path.join(SERVER_DIR, "ensemble_columns.npy")

def load_model():
    """Load the ensemble model and metrics"""
    try:
        model = load(MODEL_PATH)
        accuracy = float(open(ACCURACY_PATH).read().strip())
        auc = float(open(AUC_PATH).read().strip())
        sensitivity = float(open(SENSITIVITY_PATH).read().strip())
        specificity = float(open(SPECIFICITY_PATH).read().strip())
        columns = np.load(COLUMNS_PATH, allow_pickle=True)
        return model, accuracy, auc, sensitivity, specificity, columns
    except FileNotFoundError:
        return None, None, None, None, None, None

def predict_ensemble(input_dict):
    """Make prediction using the ensemble model"""
    model, acc, auc_score, sensitivity, specificity, columns = load_model()
    if model is None:
        raise ValueError("Ensemble model not found. Please train the model first.")
    
    input_df = pd.DataFrame([input_dict])
    
    # Apply the same feature engineering as training
    input_df = apply_feature_engineering(input_df)
    
    # Reindex to match expected columns (engineered features)
    if columns is not None:
        input_df = input_df.reindex(columns=columns, fill_value=0)
    
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
    X, y = apply_feature_engineering(X, target=y)
    print(f"Total features after engineering: {X.shape[1]}")
    print(f"New features added: {X.shape[1] - 21}")

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE ENSEMBLE MODEL")
    print("=" * 80)
    print("Strategy: Weighted Soft Voting")
    print("Models: XGBoost, Logistic Regression, Random Forest, KNN")
    print("=" * 80)

    # Create individual models with pipelines
    print("\nCreating individual models...")
    
    # XGBoost pipeline - Strong accuracy and ROC-AUC
    xgb_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    # Logistic Regression pipeline - High sensitivity
    lr_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        ))
    ])
    
    # Random Forest pipeline - Balanced sensitivity/specificity
    rf_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # KNN pipeline - Adds diversity, especially on borderline points
    knn_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', KNeighborsClassifier(
            n_neighbors=21,
            weights='uniform',
            metric='minkowski',
            p=2,
            algorithm='kd_tree'
        ))
    ])

    # Create ensemble with weighted soft voting
    # Weights based on individual model strengths:
    # XGBoost: 0.35 (strong accuracy and ROC-AUC)
    # Logistic Regression: 0.30 (high sensitivity, good F1)
    # Random Forest: 0.20 (balanced performance, stabilizes decisions)
    # KNN: 0.15 (adds diversity, especially on borderline cases)
    
    ensemble = VotingClassifier(
        estimators=[
            ('xgboost', xgb_pipeline),
            ('logistic', lr_pipeline),
            ('randomforest', rf_pipeline),
            ('knn', knn_pipeline)
        ],
        voting='soft',  # Use probability voting for calibrated models
        weights=[0.35, 0.30, 0.20, 0.15]  # Weighted based on performance
    )

    print("\nEnsemble Configuration:")
    print("- XGBoost (weight: 0.35) - Strong accuracy and ROC-AUC")
    print("- Logistic Regression (weight: 0.30) - High sensitivity and F1")
    print("- Random Forest (weight: 0.20) - Balanced performance")
    print("- KNN (weight: 0.15) - Algorithmic diversity")

    print("\nTraining ensemble model...")
    ensemble.fit(X_train, y_train)
    
    print("Applying probability calibration...")
    calibrated_ensemble = CalibratedClassifierCV(
        ensemble, 
        method='isotonic',
        cv=3,
        n_jobs=-1
    )
    
    calibrated_ensemble.fit(X_train, y_train)
    print("Calibration completed!")

    # Evaluate with threshold optimization
    best_model = ensemble
    
    y_pred_proba_uncalibrated = best_model.predict_proba(X_val)[:, 1]
    y_pred_proba_calibrated = calibrated_ensemble.predict_proba(X_val)[:, 1]
    y_pred_proba = y_pred_proba_calibrated
    
    # Use explicit threshold grid to avoid indexing issues
    threshold_grid = np.linspace(0.01, 0.99, 99)  # Avoid 0 and 1 to prevent edge cases
    
    f1_scores = []
    youden_j_scores = []
    balanced_acc_scores = []
    
    for threshold in threshold_grid:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred_thresh)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = sensitivity
            
            # Calculate metrics
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            youden_j = sensitivity + specificity - 1
            balanced_acc = (sensitivity + specificity) / 2
        else:
            f1 = 0
            youden_j = 0
            balanced_acc = 0
        
        f1_scores.append(f1)
        youden_j_scores.append(youden_j)
        balanced_acc_scores.append(balanced_acc)
    
    f1_optimal_idx = np.argmax(f1_scores)
    youden_j_optimal_idx = np.argmax(youden_j_scores)
    balanced_acc_optimal_idx = np.argmax(balanced_acc_scores)
    
    f1_optimal_threshold = threshold_grid[f1_optimal_idx]
    youden_j_optimal_threshold = threshold_grid[youden_j_optimal_idx]
    balanced_acc_optimal_threshold = threshold_grid[balanced_acc_optimal_idx]
    
    # Use Youden's J for optimal balance
    optimal_threshold = youden_j_optimal_threshold
    print(f"Using Youden's J optimal threshold: {optimal_threshold:.4f}")
    
    y_val_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    print(f"\nThreshold Comparison:")
    print(f"F1 optimal: {f1_optimal_threshold:.4f}")
    print(f"Youden's J optimal: {youden_j_optimal_threshold:.4f}")
    print(f"Balanced Acc optimal: {balanced_acc_optimal_threshold:.4f}")
    print(f"Selected: {optimal_threshold:.4f}")
    
    print(f"\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION COMPARISON")
    print("=" * 60)
    
    thresholds_to_test = {
        'Default (0.5)': 0.5,
        'F1 Optimal': f1_optimal_threshold,
        'Youden J Optimal': youden_j_optimal_threshold,
        'Balanced Acc Optimal': balanced_acc_optimal_threshold
    }
    
    for name, thresh in thresholds_to_test.items():
        y_pred_test = (y_pred_proba >= thresh).astype(int)
        cm_test = confusion_matrix(y_val, y_pred_test)
        tn, fp, fn, tp = cm_test.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden_j = sensitivity + specificity - 1
        balanced_acc = (sensitivity + specificity) / 2
        f1 = f1_score(y_val, y_pred_test)
        accuracy = accuracy_score(y_val, y_pred_test)
        
        print(f"\n{name} (threshold={thresh:.4f}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Youden's J: {youden_j:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # Calculate Brier Score for calibration comparison
    brier_uncalibrated = brier_score_loss(y_val, y_pred_proba_uncalibrated)
    brier_calibrated = brier_score_loss(y_val, y_pred_proba_calibrated)
    
    print(f"\nBrier Score (Uncalibrated): {brier_uncalibrated:.4f}")
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

    print("\n" + "=" * 60)
    print(f"ENSEMBLE CLASSIFICATION EVALUATION METRICS")
    print(f"Optimized using: YOUDEN_J")
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

    print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Positive Rate:    {fp/(fp+tn):.4f}")
    print(f"False Negative Rate:    {fn/(fn+tp):.4f}")
    print(f"Positive Predictive Value: {tp/(tp+fp):.4f}")
    print(f"Negative Predictive Value: {tn/(tn+fn):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))

    # Ensure output directory exists
    os.makedirs(CLIENT_PUBLIC_DIR, exist_ok=True)

    # Confusion matrix figure
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Ensemble Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["No Diabetes", "Diabetes"])
    plt.yticks([0, 1], ["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    cm_path = os.path.join(CLIENT_PUBLIC_DIR, "ensemble_confusion_matrix.png")
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
    plt.title("Ensemble ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(CLIENT_PUBLIC_DIR, "ensemble_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Ensemble Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve - Ensemble')
    plt.legend()
    plt.grid(True)
    calibration_path = os.path.join(CLIENT_PUBLIC_DIR, "ensemble_calibration_curve.png")
    plt.savefig(calibration_path)
    print("Calibration curve saved")
    plt.close()

    # Model comparison visualization
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Get individual model predictions for comparison
    individual_predictions = {}
    individual_names = ['XGBoost', 'Logistic Regression', 'Random Forest', 'KNN']
    
    for i, (name, estimator) in enumerate(ensemble.named_estimators_.items()):
        y_pred_individual = estimator.predict_proba(X_val)[:, 1]
        y_pred_binary = (y_pred_individual >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred_binary)
        auc = roc_auc_score(y_val, y_pred_individual)
        sens = recall_score(y_val, y_pred_binary)
        spec = (confusion_matrix(y_val, y_pred_binary)[0,0] / 
                (confusion_matrix(y_val, y_pred_binary)[0,0] + confusion_matrix(y_val, y_pred_binary)[0,1]))
        f1 = f1_score(y_val, y_pred_binary)
        
        individual_predictions[name] = {
            'accuracy': acc,
            'auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'f1': f1
        }
        
        print(f"{name}:")
        print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, F1: {f1:.4f}")

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    
    # Use ensemble for proper cross-validation without data leakage
    cv_accuracy = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='precision')
    cv_recall = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='recall')
    cv_f1 = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='f1')
    cv_roc_auc = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='roc_auc')
    cv_balanced_acc = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='balanced_accuracy')
    
    print(f"CV Accuracy:       {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"CV Balanced Acc:   {cv_balanced_acc.mean():.4f} (+/- {cv_balanced_acc.std() * 2:.4f})")
    print(f"CV Precision:      {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    print(f"CV Recall:         {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    print(f"CV F1:             {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    print(f"CV ROC-AUC:        {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
    print("=" * 60)

    # Save classification report
    classification_report_path = os.path.join(CLIENT_PUBLIC_DIR, "ensemble_classification_report.txt")
    with open(classification_report_path, 'w') as f:
        f.write("Comprehensive Ensemble Model Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Model Composition (Weighted Soft Voting):\n")
        f.write("- XGBoost (weight: 0.35) - Strong accuracy and ROC-AUC\n")
        f.write("- Logistic Regression (weight: 0.30) - High sensitivity and F1\n")
        f.write("- Random Forest (weight: 0.20) - Balanced performance\n")
        f.write("- KNN (weight: 0.15) - Algorithmic diversity\n\n")
        f.write("Individual Model Performance:\n")
        for name, metrics in individual_predictions.items():
            f.write(f"{name}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, ")
            f.write(f"Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}, F1={metrics['f1']:.4f}\n")
        f.write(f"\nEnsemble Performance:\n")
        f.write(classification_report(y_val, y_val_pred, digits=4))
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"AUC Score: {roc_auc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Mean CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")

    # Save model and metrics
    dump(calibrated_ensemble, MODEL_PATH)
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
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Saved columns to: {COLUMNS_PATH}")
    print(f"Saved accuracy to: {ACCURACY_PATH}")
    print(f"Saved AUC to: {AUC_PATH}")
    print(f"Saved sensitivity to: {SENSITIVITY_PATH}")
    print(f"Saved specificity to: {SPECIFICITY_PATH}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved ROC curve to: {roc_path}")
    print(f"Saved calibration curve to: {calibration_path}")
    print(f"Saved classification report to: {classification_report_path}")
