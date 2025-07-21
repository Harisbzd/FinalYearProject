import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "logreg_model.joblib"
ACCURACY_PATH = "logreg_accuracy.txt"
COLUMNS_PATH = "logreg_columns.npy"

_loaded_model = None
_loaded_accuracy = None
_loaded_columns = None

def load_model():
    global _loaded_model, _loaded_accuracy, _loaded_columns
    if _loaded_model is None:
        _loaded_model = joblib.load(MODEL_PATH)
    if _loaded_accuracy is None:
        if os.path.exists(ACCURACY_PATH):
            with open(ACCURACY_PATH) as f:
                _loaded_accuracy = float(f.read().strip())
        else:
            _loaded_accuracy = None
    if _loaded_columns is None:
        if os.path.exists(COLUMNS_PATH):
            _loaded_columns = np.load(COLUMNS_PATH, allow_pickle=True)
        else:
            _loaded_columns = None
            
    return _loaded_model, _loaded_accuracy, _loaded_columns

def predict_logreg(input_dict):
    model, acc, columns = load_model()
    input_df = pd.DataFrame([input_dict])
    if columns is not None:
        input_df = input_df.reindex(columns=columns)
    
    pred = model.predict(input_df)[0]
    return int(pred), acc

if __name__ == "__main__":
    DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Build pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear", random_state=42))
    ])

    # Grid search for hyperparameter tuning
    param_grid = {
        "logreg__C": [0.01, 0.1, 1, 10],
        "logreg__class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", verbose=0)
    grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train_res, y_train_res,
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
    plt.title('Learning Curves for Logistic Regression')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('../client/public/learning_curves.png')
    print("Learning curves saved to learning_curves.png")
    plt.close()

    # Feature Importance Plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.named_steps['logreg'].coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('../client/public/feature_importance.png')
    print("Feature importance plot saved to feature_importance.png")
    plt.close()

    # Calibration Curve
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_test, y_prob[:, 1], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../client/public/calibration_curve.png')
    print("Calibration curve saved to calibration_curve.png")
    plt.close()

    # Decision Boundary Plot (using top 2 most important features)
    top_features = feature_importance['feature'].iloc[:2].values
    X_top2 = X_test[top_features]
    
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

    # Get predictions
    Z = best_model.predict(X_mesh_full)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X_top2.iloc[:, 0], X_top2.iloc[:, 1], c=y_test, 
                         alpha=0.8, edgecolor='black')
    plt.colorbar(scatter)
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title('Decision Boundary using Top 2 Features')
    plt.savefig('../client/public/decision_boundary.png')
    print("Decision boundary plot saved to decision_boundary.png")
    plt.close()

    # Save model, columns, and accuracy
    joblib.dump(best_model, MODEL_PATH)
    np.save(COLUMNS_PATH, X.columns.values)
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(accuracy))

    # Print results
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.named_steps['logreg'].classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Logistic Regression (Test Set)')
    plt.savefig('../client/public/confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.close()

    # Generate and save ROC curve
    y_score = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('../client/public/roc_curve.png')
    print("ROC curve saved to roc_curve.png")
    plt.close()

    plt.show()