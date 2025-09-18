import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve, balanced_accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))
from feature_engineering import apply_feature_engineering
from model_utils import load_model_metrics
MODEL_PATH = "xgboost_model.joblib"
ACCURACY_PATH = "xgboost_accuracy.txt"
AUC_PATH = "xgboost_auc.txt"
SENSITIVITY_PATH = "xgboost_sensitivity.txt"
SPECIFICITY_PATH = "xgboost_specificity.txt"
COLUMNS_PATH = "xgboost_columns.npy"



def load_model():
    model, accuracy, auc, sensitivity, specificity, columns, scaler = load_model_metrics('xgboost')
    return model, accuracy, auc, sensitivity, specificity, columns, scaler

def predict_xgboost(input_dict):
    model, acc, auc_score, sensitivity, specificity, columns, scaler = load_model()
    input_df = pd.DataFrame([input_dict])
    
    input_df = apply_feature_engineering(input_df)
    if columns is not None:
        input_df = input_df.reindex(columns=columns, fill_value=0)
    
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values
    
    pred = model.predict(input_scaled)[0]
    return int(pred), acc, auc_score, sensitivity, specificity

if __name__ == "__main__":
    df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    X = df.drop(columns=["Diabetes_binary"])
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
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('classifier', XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=1.0,
            tree_method='hist',
            enable_categorical=False
        ))
    ])
    param_dist = {
        'classifier__n_estimators': [200, 400, 600],  
        'classifier__max_depth': [4, 6, 8, 10, 12],
        'classifier__learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'classifier__min_child_weight': [1, 3, 5, 7, 10],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2, 0.3],
        'classifier__reg_alpha': [0, 0.1, 0.5, 1.0],
        'classifier__reg_lambda': [1, 1.5, 2, 3]
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
    
    grid = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=60, 
        scoring=selected_scorer,  
        cv=3,  
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
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
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
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
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print("=" * 60)
    print(f"XGBOOST CLASSIFICATION EVALUATION METRICS")
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
    print("=" * 60)
    
    print(f"\nConfusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")
    print(f"Positive Predictive Value: {tp/(tp+fp):.4f}")
    print(f"Negative Predictive Value: {tn/(tn+fn):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Ensure output directory exists
    output_dir = '../client/public'
    os.makedirs(output_dir, exist_ok=True)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.named_steps['classifier'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - XGBoost')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xgboost_feature_importance.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve - XGBoost')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'xgboost_calibration_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.named_steps['classifier'].classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - XGBoost')
    plt.savefig(os.path.join(output_dir, 'xgboost_confusion_matrix.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - XGBoost')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'xgboost_roc_curve.png'))
    plt.close()
    joblib.dump(best_model, MODEL_PATH)
    # Extract scaler from pipeline for saving
    scaler = best_model.named_steps['scaler']
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
    for param, value in grid.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best CV Score: {grid.best_score_:.4f}")
    print("=" * 60)
