import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Model saving paths
MODEL_PATH = "randomforest_model.joblib"
ACCURACY_PATH = "randomforest_accuracy.txt"
AUC_PATH = "randomforest_auc.txt"
COLUMNS_PATH = "randomforest_columns.npy"

_loaded_model = None
_loaded_accuracy = None
_loaded_auc = None
_loaded_columns = None


def load_model():
    global _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_columns
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
    if _loaded_columns is None:
        if os.path.exists(COLUMNS_PATH):
            _loaded_columns = np.load(COLUMNS_PATH, allow_pickle=True)
        else:
            _loaded_columns = None

    return _loaded_model, _loaded_accuracy, _loaded_auc, _loaded_columns


def predict_randomforest(input_dict):
    model, acc, auc_score, columns = load_model()
    input_df = pd.DataFrame([input_dict])
    if columns is not None:
        input_df = input_df.reindex(columns=columns)

    pred = model.predict(input_df)[0]
    return int(pred), acc, auc_score


if __name__ == "__main__":
    # -----------------------------
    # Load dataset
    # -----------------------------
    DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(DATA_PATH)

    # -----------------------------
    # Drop weak features
    # -----------------------------
    weak_features = ["CholCheck"]
    X = df.drop(columns=["Diabetes_binary"] + weak_features)
    y = df["Diabetes_binary"]

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Hyperparameter tuning with GridSearchCV
    # -----------------------------
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # -----------------------------
    # Evaluate the model
    # -----------------------------
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nValidation Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

    # -----------------------------
    # Generate learning curves
    # -----------------------------
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, n_jobs=-1, scoring='accuracy'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Score')
    plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves - Random Forest')
    plt.legend()
    plt.grid(True)
    plt.savefig('../client/public/randomforest_learning_curves.png')
    print("Learning curves saved to randomforest_learning_curves.png")
    plt.close()

    # -----------------------------
    # Feature Importance
    # -----------------------------
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('../client/public/randomforest_feature_importance.png')
    print("Feature importance saved to randomforest_feature_importance.png")
    plt.close()

    # -----------------------------
    # Calibration Curve
    # -----------------------------
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=10
    )

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Random Forest")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve - Random Forest')
    plt.legend()
    plt.grid(True)
    plt.savefig('../client/public/randomforest_calibration_curve.png')
    print("Calibration curve saved to randomforest_calibration_curve.png")
    plt.close()

    # -----------------------------
    # Decision Boundary (using top 2 features)
    # -----------------------------
    top_2_features = importance_df.head(2)['feature'].values
    X_2d = X_test[top_2_features]

    # Create a mesh
    h = 0.02
    x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
    y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Train a 2D model for visualization
    model_2d = RandomForestClassifier(random_state=42)
    model_2d.fit(X_2d, y_test)

    # Make predictions on the mesh
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_2d.iloc[:, 0], X_2d.iloc[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlabel(top_2_features[0])
    plt.ylabel(top_2_features[1])
    plt.title('Decision Boundary - Random Forest (Top 2 Features)')
    plt.colorbar(scatter)
    plt.savefig('../client/public/randomforest_decision_boundary.png')
    print("Decision boundary saved to randomforest_decision_boundary.png")
    plt.close()

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Random Forest (Validation Set)')
    plt.savefig('../client/public/randomforest_confusion_matrix.png')
    print("Confusion matrix saved to randomforest_confusion_matrix.png")
    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend()
    plt.grid(True)
    plt.savefig('../client/public/randomforest_roc_curve.png')
    print("ROC curve saved to randomforest_roc_curve.png")
    plt.close()

    # -----------------------------
    # Save model, columns, and metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    joblib.dump(best_model, MODEL_PATH)
    np.save(COLUMNS_PATH, X.columns.values)
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(accuracy))
    with open(AUC_PATH, "w") as f:
        f.write(str(auc_score))

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")