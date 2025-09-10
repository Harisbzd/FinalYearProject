import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.svm import SVC, LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Resolve important paths
THIS_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, os.pardir))
CLIENT_PUBLIC_DIR = os.path.join(REPO_ROOT, "client", "public")

def evaluate_svm_model(use_undersampling=True, include_linear_baseline=True):
    """
    Train and evaluate SVM model without saving it.
    This function is for evaluation purposes only.
    
    Args:
        use_undersampling (bool): If True, use RandomUnderSampler for faster training
        include_linear_baseline (bool): If True, also train LinearSVC for interpretability
    """
    
    # -----------------------------
    # Load dataset
    # -----------------------------
    DATA_PATH = os.path.join(SERVER_DIR, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(DATA_PATH)
    
    # -----------------------------
    # Use ALL features (no feature dropping for better performance)
    # -----------------------------
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------
    # Scale ALL features with StandardScaler (essential for SVM)
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -----------------------------
    # Handle class imbalance (choose strategy based on speed preference)
    # -----------------------------
    if use_undersampling:
        print("Using RandomUnderSampler for faster training...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = undersampler.fit_resample(X_train_scaled, y_train)
        print(f"Training set size after undersampling: {X_train_res.shape[0]}")
    else:
        print("Using SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        print(f"Training set size after SMOTE: {X_train_res.shape[0]}")

    # -----------------------------
    # Define SVM model
    # -----------------------------
    svm = SVC(
        probability=True,  # Enable probability estimates for ROC curve
        random_state=42
    )

    # -----------------------------
    # Optimized hyperparameter grid for faster training
    # -----------------------------
    param_grid = {
        'C': [1, 10],
        'gamma': ['scale'],
        'kernel': ['linear', 'rbf']
    }

    # -----------------------------
    # Grid search with minimal parameters for speed
    # -----------------------------
    print("Starting SVM hyperparameter tuning (optimized for speed)...")
    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train_res, y_train_res)

    # -----------------------------
    # Results
    # -----------------------------
    print("SVM Hyperparameter tuning completed!")
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")

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
    
    print("\n" + "="*50)
    print("SVM MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))
    print("\nConfusion Matrix:")
    print(cm)

    # -----------------------------
    # Cross-validation scores
    # -----------------------------
    cv_scores = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # -----------------------------
    # Feature Importance (for linear kernel only)
    # -----------------------------
    if grid.best_params_['kernel'] == 'linear':
        feature_importance = np.abs(best_model.coef_[0])
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        })
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
        plt.title('Feature Importance - SVM (Linear Kernel)')
        plt.xlabel('Feature Importance Score')
        plt.tight_layout()
        feature_importance_path = os.path.join(CLIENT_PUBLIC_DIR, "svm_feature_importance.png")
        plt.savefig(feature_importance_path, dpi=200)
        print(f"Feature importance saved to {feature_importance_path}")
        plt.close()
    else:
        print("Feature importance not available for non-linear kernels")

    # -----------------------------
    # Calibration Curve
    # -----------------------------
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='SVM')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve (Reliability Diagram) - SVM')
    plt.legend()
    plt.grid(True)
    calibration_path = os.path.join(CLIENT_PUBLIC_DIR, "svm_calibration_curve.png")
    plt.savefig(calibration_path, dpi=200)
    print(f"Calibration curve saved to {calibration_path}")
    plt.close()

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - SVM (Validation Set)')
    confusion_path = os.path.join(CLIENT_PUBLIC_DIR, "svm_confusion_matrix.png")
    plt.savefig(confusion_path, dpi=200)
    print(f"Confusion matrix saved to {confusion_path}")
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
    plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
    plt.legend(loc="lower right")
    roc_path = os.path.join(CLIENT_PUBLIC_DIR, "svm_roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    print(f"ROC curve saved to {roc_path}")
    plt.close()

    # -----------------------------
    # Save classification report
    # -----------------------------
    classification_report_path = os.path.join(CLIENT_PUBLIC_DIR, "svm_classification_report.txt")
    with open(classification_report_path, 'w') as f:
        f.write("SVM Model Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best Parameters: {grid.best_params_}\n")
        f.write(f"Best CV Score: {grid.best_score_:.4f}\n\n")
        f.write(classification_report(y_val, y_pred, digits=4))
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"AUC Score: {roc_auc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print(f"Classification report saved to {classification_report_path}")

    # -----------------------------
    # LinearSVC Baseline for Interpretability (if requested)
    # -----------------------------
    linear_results = None
    if include_linear_baseline:
        print("\n" + "="*50)
        print("TRAINING LINEAR SVM BASELINE FOR INTERPRETABILITY")
        print("="*50)
        
        # Train LinearSVC
        linear_svm = LinearSVC(random_state=42, max_iter=2000)
        linear_svm.fit(X_train_res, y_train_res)
        
        # Get predictions (LinearSVC doesn't have predict_proba, so we use decision_function)
        y_pred_linear = linear_svm.predict(X_val_scaled)
        y_scores_linear = linear_svm.decision_function(X_val_scaled)
        
        # Calculate metrics
        accuracy_linear = accuracy_score(y_val, y_pred_linear)
        roc_auc_linear = roc_auc_score(y_val, y_scores_linear)
        
        # Calculate Sensitivity and Specificity
        cm_linear = confusion_matrix(y_val, y_pred_linear)
        tn_linear, fp_linear, fn_linear, tp_linear = cm_linear.ravel()
        sensitivity_linear = tp_linear / (tp_linear + fn_linear)
        specificity_linear = tn_linear / (tn_linear + fp_linear)
        
        print(f"Linear SVM Accuracy: {accuracy_linear:.4f}")
        print(f"Linear SVM AUC: {roc_auc_linear:.4f}")
        print(f"Linear SVM Sensitivity: {sensitivity_linear:.4f}")
        print(f"Linear SVM Specificity: {specificity_linear:.4f}")
        
        # Feature importance for Linear SVM
        feature_importance_linear = np.abs(linear_svm.coef_[0])
        feature_importance_df_linear = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance_linear
        })
        feature_importance_df_linear = feature_importance_df_linear.sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance_df_linear.head(15), x='importance', y='feature')
        plt.title('Feature Importance - Linear SVM')
        plt.xlabel('Feature Importance Score')
        plt.tight_layout()
        linear_feature_path = os.path.join(CLIENT_PUBLIC_DIR, "linear_svm_feature_importance.png")
        plt.savefig(linear_feature_path, dpi=200)
        print(f"Linear SVM feature importance saved to {linear_feature_path}")
        plt.close()
        
        linear_results = {
            'accuracy': accuracy_linear,
            'auc': roc_auc_linear,
            'sensitivity': sensitivity_linear,
            'specificity': specificity_linear,
            'feature_importance': feature_importance_df_linear
        }

    # -----------------------------
    # Summary
    # -----------------------------
    print("\n" + "="*50)
    print("SVM MODEL SUMMARY")
    print("="*50)
    print(f"Best Kernel: {grid.best_params_['kernel']}")
    print(f"Best C: {grid.best_params_['C']}")
    print(f"Best Gamma: {grid.best_params_['gamma']}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final AUC: {roc_auc:.4f}")
    print(f"Final Sensitivity: {sensitivity:.4f}")
    print(f"Final Specificity: {specificity:.4f}")
    
    if linear_results:
        print("\nLinear SVM Baseline:")
        print(f"Linear Accuracy: {linear_results['accuracy']:.4f}")
        print(f"Linear AUC: {linear_results['auc']:.4f}")
        print(f"Linear Sensitivity: {linear_results['sensitivity']:.4f}")
        print(f"Linear Specificity: {linear_results['specificity']:.4f}")
    
    print("="*50)
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'best_params': grid.best_params_,
        'cv_scores': cv_scores,
        'linear_results': linear_results
    }

if __name__ == "__main__":
    # Run the SVM evaluation with optimized settings
    print("Starting optimized SVM evaluation...")
    print("Using undersampling for faster training and including Linear SVM baseline")
    
    results = evaluate_svm_model(
        use_undersampling=True,  # Use undersampling for faster training
        include_linear_baseline=True  # Include Linear SVM for interpretability
    )
    
    print("\nSVM model evaluation completed!")
    print("Note: This model was created for evaluation purposes only and was not saved.")
    print("\nKey optimizations applied:")
    print("- Reduced parameter grid (2 C values, 1 gamma, 2 kernels)")
    print("- Used RandomUnderSampler instead of SMOTE for faster training")
    print("- Added Linear SVM baseline for interpretability")
