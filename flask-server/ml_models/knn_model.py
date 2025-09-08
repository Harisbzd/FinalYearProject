import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from joblib import dump, load

# Resolve important paths
THIS_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SERVER_DIR, os.pardir))
CLIENT_PUBLIC_DIR = os.path.join(REPO_ROOT, "client", "public")

# Model saving paths
MODEL_PATH = os.path.join(SERVER_DIR, "knn_model.joblib")
ACCURACY_PATH = os.path.join(SERVER_DIR, "knn_accuracy.txt")
AUC_PATH = os.path.join(SERVER_DIR, "knn_auc.txt")
COLUMNS_PATH = os.path.join(SERVER_DIR, "knn_columns.npy")

_loaded_model = None
_loaded_accuracy = None
_loaded_auc = None
_loaded_columns = None

def load_model():
    global _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_columns
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
    if _loaded_columns is None:
        if os.path.exists(COLUMNS_PATH):
            _loaded_columns = np.load(COLUMNS_PATH, allow_pickle=True)
        else:
            _loaded_columns = None
            
    return _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_columns

def predict_knn(input_dict):
    model_data, acc, auc_score, columns = load_model()
    if model_data is None:
        raise ValueError("KNN model not found. Please train the model first.")
    
    input_df = pd.DataFrame([input_dict])
    
    # Reindex to match expected columns
    if columns is not None:
        input_df = input_df.reindex(columns=columns, fill_value=0)
    
    # Extract model, scaler, and threshold from saved data
    model = model_data['model']
    scaler = model_data['scaler']
    threshold = model_data['threshold']
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Get prediction probability
    pred_proba = model.predict_proba(input_scaled)[:, 1]
    
    # Apply threshold
    pred = int(pred_proba >= threshold)
    
    return pred, acc, auc_score

if __name__ == "__main__":
    # Load dataset
    DATA_PATH = os.path.join(SERVER_DIR, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(DATA_PATH)

    # No log transformations - use original features

    # Features and target
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Hyperparameter grid
    param_grid = {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # 1: Manhattan, 2: Euclidean
    }

    knn = KNeighborsClassifier()
    grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring="recall",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    # Predict probabilities for ROC/threshold tuning
    y_val_prob = best_model.predict_proba(X_val_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    roc_scores = [(thr, (tpr[i] - fpr[i])) for i, thr in enumerate(thresholds)]
    best_threshold = max(roc_scores, key=lambda x: x[1])[0]
    print(f"Best threshold = {best_threshold:.2f}")

    y_val_pred = (y_val_prob >= best_threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_prob)
    print("Validation Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_val, y_val_pred, digits=4))

    # Confusion matrix figure
    cm = confusion_matrix(y_val, y_val_pred)
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
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("KNN ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # -----------------------------
    # Generate learning curves
    # -----------------------------
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, n_jobs=-1, scoring='accuracy'
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='green', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='green')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves for KNN')
    plt.legend(loc='lower right')
    plt.grid(True)
    learning_curves_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_learning_curves.png")
    plt.savefig(learning_curves_path)
    print("Learning curves saved to knn_learning_curves.png")
    plt.close()

    # -----------------------------
    # Feature Importance Plot (using permutation importance approximation)
    # -----------------------------
    from sklearn.inspection import permutation_importance

    # Calculate permutation importance
    perm_importance = permutation_importance(best_model, X_val_scaled, y_val, n_repeats=10, random_state=42)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - KNN (Permutation Importance)')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    feature_importance_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_feature_importance.png")
    plt.savefig(feature_importance_path)
    print("Feature importance plot saved to knn_feature_importance.png")
    plt.close()

    # -----------------------------
    # Calibration Curve
    # -----------------------------
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_val, y_val_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='KNN')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve (Reliability Diagram) - KNN')
    plt.legend()
    plt.grid(True)
    calibration_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_calibration_curve.png")
    plt.savefig(calibration_path)
    print("Calibration curve saved to knn_calibration_curve.png")
    plt.close()

    # -----------------------------
    # Decision Boundary Plot (using top 2 most important features)
    # -----------------------------
    top_features = feature_importance['feature'].iloc[:2].values
    X_top2 = X_val[top_features]

    # Create mesh grid
    x_min, x_max = X_top2.iloc[:, 0].min() - 1, X_top2.iloc[:, 0].max() + 1
    y_min, y_max = X_top2.iloc[:, 1].min() - 1, X_top2.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # Create feature matrix for prediction
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_mesh_full = np.zeros((X_mesh.shape[0], X.shape[1]))
    for i, feature in enumerate(top_features):
        feature_idx = list(X.columns).index(feature)
        X_mesh_full[:, feature_idx] = X_mesh[:, i]

    # Scale the mesh data
    X_mesh_scaled = scaler.transform(X_mesh_full)

    # Get predictions
    Z = best_model.predict(X_mesh_scaled)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X_top2.iloc[:, 0], X_top2.iloc[:, 1], c=y_val, 
                         alpha=0.8, edgecolor='black')
    plt.colorbar(scatter)
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title('Decision Boundary using Top 2 Features - KNN')
    decision_boundary_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_decision_boundary.png")
    plt.savefig(decision_boundary_path)
    print("Decision boundary plot saved to knn_decision_boundary.png")
    plt.close()

    # -----------------------------
    # Cross-validation scores
    # -----------------------------
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")

    # -----------------------------
    # Save classification report
    # -----------------------------
    classification_report_path = os.path.join(CLIENT_PUBLIC_DIR, "knn_classification_report.txt")
    with open(classification_report_path, 'w') as f:
        f.write("KNN Model Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(classification_report(y_val, y_val_pred, digits=4))
        f.write(f"\nAccuracy: {acc:.4f}\n")
        f.write(f"AUC Score: {auc:.4f}\n")
        f.write(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("Classification report saved to knn_classification_report.txt")

    # -----------------------------
    # Persist artifacts (model, columns, metrics)
    # -----------------------------
    dump({"model": best_model, "scaler": scaler, "threshold": float(best_threshold)}, MODEL_PATH)
    np.save(COLUMNS_PATH, X.columns.to_numpy())
    with open(ACCURACY_PATH, "w") as f:
        f.write(f"{acc:.6f}")
    with open(AUC_PATH, "w") as f:
        f.write(f"{auc:.6f}")

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Saved columns to: {COLUMNS_PATH}")
    print(f"Saved accuracy to: {ACCURACY_PATH}")
    print(f"Saved AUC to: {AUC_PATH}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved ROC curve to: {roc_path}")
    print(f"Saved learning curves to: {learning_curves_path}")
    print(f"Saved feature importance to: {feature_importance_path}")
    print(f"Saved calibration curve to: {calibration_path}")
    print(f"Saved decision boundary to: {decision_boundary_path}")
    print(f"Saved classification report to: {classification_report_path}")
