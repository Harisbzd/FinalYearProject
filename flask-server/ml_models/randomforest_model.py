import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve, balanced_accuracy_score, make_scorer, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))
from feature_engineering import apply_feature_engineering
from model_utils import load_model_metrics

THIS_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, os.pardir))
CLIENT_PUBLIC_DIR = os.path.join(REPO_ROOT, "client", "public")

MODEL_PATH = os.path.join(SERVER_DIR, "randomforest_model.joblib")
ACCURACY_PATH = os.path.join(SERVER_DIR, "randomforest_accuracy.txt")
AUC_PATH = os.path.join(SERVER_DIR, "randomforest_auc.txt")
SENSITIVITY_PATH = os.path.join(SERVER_DIR, "randomforest_sensitivity.txt")
SPECIFICITY_PATH = os.path.join(SERVER_DIR, "randomforest_specificity.txt")
COLUMNS_PATH = os.path.join(SERVER_DIR, "randomforest_columns.npy")
SCALER_PATH = os.path.join(SERVER_DIR, "randomforest_scaler.joblib")

def load_model():
    model, accuracy, auc, sensitivity, specificity, columns, scaler = load_model_metrics('randomforest')
    return model, accuracy, auc, sensitivity, specificity, columns, scaler

def predict_randomforest(input_dict):
    model, acc, auc_score, sensitivity, specificity, columns, scaler = load_model()
    if model is None:
        raise ValueError("Random Forest model not found. Please train the model first.")
    
    input_df = pd.DataFrame([input_dict])
    
    input_df = apply_feature_engineering(input_df)
    if columns is not None:
        input_df = input_df.reindex(columns=columns, fill_value=0)
    
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    
    return int(pred), acc, auc_score, sensitivity, specificity


if __name__ == "__main__":
    DATA_PATH = os.path.join(SERVER_DIR, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    
    print(f"Original features: {X.shape[1]}")
    X, y = apply_feature_engineering(X, target=y)
    print(f"Total features after engineering: {X.shape[1]}")
    print(f"New features added: {X.shape[1] - 21}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Create pipeline to avoid data leakage - SMOTE and scaling happen inside CV
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Custom scorers for different optimization metrics
    def balanced_accuracy_scorer(y_true, y_pred):
        return balanced_accuracy_score(y_true, y_pred)
    
    def youden_j_scorer(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sensitivity + specificity - 1
    
    def cost_sensitive_scorer(y_true, y_pred, cost_fn=2.0, cost_fp=1.0):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = cost_fn * fn + cost_fp * fp
        return -cost
    
    def balanced_recall_accuracy_scorer(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return 0.6 * recall + 0.4 * accuracy
    
    balanced_acc_scorer = make_scorer(balanced_accuracy_scorer, greater_is_better=True)
    youden_j_scorer_wrapper = make_scorer(youden_j_scorer, greater_is_better=True)
    cost_sensitive_scorer_wrapper = make_scorer(cost_sensitive_scorer, greater_is_better=True)
    custom_scorer = make_scorer(balanced_recall_accuracy_scorer, greater_is_better=True)
    
    SCORING_METRIC = 'youden_j'
    
    if SCORING_METRIC == 'balanced_accuracy':
        selected_scorer = balanced_acc_scorer
        print("Using Balanced Accuracy for optimization (range: 0-1)")
    elif SCORING_METRIC == 'youden_j':
        selected_scorer = youden_j_scorer_wrapper
        print("Using Youden's J statistic for optimization (range: -1 to 1)")
    elif SCORING_METRIC == 'cost_sensitive':
        selected_scorer = cost_sensitive_scorer_wrapper
        print("Using Cost-sensitive metric for optimization (minimizes cost)")
    else:
        selected_scorer = custom_scorer
        print("Using Balanced Recall-Accuracy for optimization (range: 0-1)")

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring=selected_scorer)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\nApplying probability calibration...")
    calibrated_model = CalibratedClassifierCV(
        best_model, 
        method='sigmoid',
        cv=3,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train, y_train)
    print("Calibration completed!")

    best_model = grid_search.best_estimator_
    
    y_pred_proba_uncalibrated = best_model.predict_proba(X_val)[:, 1]
    y_pred_proba_calibrated = calibrated_model.predict_proba(X_val)[:, 1]
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
    
    if SCORING_METRIC == 'youden_j':
        optimal_threshold = youden_j_optimal_threshold
        print(f"Using Youden's J optimal threshold: {optimal_threshold:.4f}")
    elif SCORING_METRIC == 'balanced_accuracy':
        optimal_threshold = balanced_acc_optimal_threshold
        print(f"Using Balanced Accuracy optimal threshold: {optimal_threshold:.4f}")
    else:
        optimal_threshold = f1_optimal_threshold
        print(f"Using F1 optimal threshold: {optimal_threshold:.4f}")
    
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
    
    brier_uncalibrated = brier_score_loss(y_val, y_pred_proba_uncalibrated)
    brier_calibrated = brier_score_loss(y_val, y_pred_proba_calibrated)
    
    print(f"Brier Score (Uncalibrated): {brier_uncalibrated:.4f}")
    print(f"Brier Score (Calibrated):   {brier_calibrated:.4f}")
    print(f"Improvement: {((brier_uncalibrated - brier_calibrated) / brier_uncalibrated * 100):.2f}%")

    accuracy = accuracy_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    mcc = matthews_corrcoef(y_val, y_val_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
    
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n" + "=" * 60)
    print(f"RANDOM FOREST CLASSIFICATION EVALUATION METRICS")
    print(f"Optimized using: {SCORING_METRIC.upper()}")
    print("=" * 60)
    print(f"Validation Accuracy:     {accuracy:.4f}")
    print(f"Balanced Accuracy:      {balanced_acc:.4f}")
    print(f"Precision:              {precision:.4f}")
    print(f"Recall (Sensitivity):   {recall:.4f}")
    print(f"Specificity:            {specificity:.4f}")
    print(f"F1-Score:               {f1:.4f}")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"Matthews Correlation:   {mcc:.4f}")
    print("=" * 50)
    
    print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Positive Rate:    {fp/(fp+tn):.4f}")
    print(f"False Negative Rate:    {fn/(fn+tp):.4f}")
    print(f"Positive Predictive Value: {tp/(tp+fp):.4f}")
    print(f"Negative Predictive Value: {tn/(tn+fn):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, digits=4))


    feature_importance = best_model.named_steps['classifier'].feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Ensure output directory exists
    os.makedirs(CLIENT_PUBLIC_DIR, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    feature_importance_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_feature_importance.png")
    plt.savefig(feature_importance_path, dpi=200)
    print("Feature importance saved to randomforest_feature_importance.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_val, y_pred_proba_uncalibrated, n_bins=10)
    plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated Random Forest', linewidth=2)
    
    prob_true_cal, prob_pred_cal = calibration_curve(y_val, y_pred_proba_calibrated, n_bins=10)
    plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Calibrated Random Forest', linewidth=2)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfectly Calibrated', linewidth=2)
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve Comparison - Random Forest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.text(0.05, 0.95, f'Uncalibrated Brier Score: {brier_uncalibrated:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(0.05, 0.90, f'Calibrated Brier Score: {brier_calibrated:.4f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    calibration_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_calibration_curve.png")
    plt.tight_layout()
    plt.savefig(calibration_path, dpi=300, bbox_inches='tight')
    print("Calibration curve saved to randomforest_calibration_curve.png")
    plt.close()


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

    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
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

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, 
        cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves - Random Forest')
    plt.legend()
    plt.grid(True)
    learning_curves_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_learning_curves.png")
    plt.tight_layout()
    plt.savefig(learning_curves_path, dpi=200)
    print("Learning curves saved to randomforest_learning_curves.png")
    plt.close()

    # Get scaled training data for PCA
    X_train_scaled = best_model.named_steps['scaler'].transform(X_train)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_original = pca.inverse_transform(mesh_points)
    Z = calibrated_model.predict(mesh_original)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Decision Boundary - Random Forest (PCA Projection)')
    decision_boundary_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_decision_boundary.png")
    plt.tight_layout()
    plt.savefig(decision_boundary_path, dpi=200)
    print("Decision boundary saved to randomforest_decision_boundary.png")
    plt.close()

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    
    # Use pipeline for proper cross-validation without data leakage
    cv_accuracy = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='precision')
    cv_recall = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='recall')
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    cv_roc_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    cv_balanced_acc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy')
    
    print(f"CV Accuracy:       {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"CV Balanced Acc:   {cv_balanced_acc.mean():.4f} (+/- {cv_balanced_acc.std() * 2:.4f})")
    print(f"CV Precision:      {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    print(f"CV Recall:         {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    print(f"CV F1:             {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    print(f"CV ROC-AUC:        {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
    print("=" * 60)
    
    print("\nBEST HYPERPARAMETERS:")
    print("=" * 60)
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print("=" * 60)

    classification_report_path = os.path.join(CLIENT_PUBLIC_DIR, "randomforest_classification_report.txt")
    with open(classification_report_path, 'w') as f:
        f.write("Random Forest Model Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(classification_report(y_val, y_val_pred, digits=4))
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"AUC Score: {roc_auc:.4f}\n")
        f.write(f"Matthews Correlation: {mcc:.4f}\n")
        f.write(f"Mean CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")

    joblib.dump(calibrated_model, MODEL_PATH)
    joblib.dump(best_model.named_steps['scaler'], SCALER_PATH)
    np.save(COLUMNS_PATH, X.columns.values)
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
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Saved columns to: {COLUMNS_PATH}")
    print(f"Saved accuracy to: {ACCURACY_PATH}")
    print(f"Saved AUC to: {AUC_PATH}")
    print(f"Saved sensitivity to: {SENSITIVITY_PATH}")
    print(f"Saved specificity to: {SPECIFICITY_PATH}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved ROC curve to: {roc_path}")
    print(f"Saved feature importance to: {feature_importance_path}")
    print(f"Saved calibration curve to: {calibration_path}")
    print(f"Saved learning curves to: {learning_curves_path}")
    print(f"Saved decision boundary to: {decision_boundary_path}")
    print(f"Saved classification report to: {classification_report_path}") 