import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt

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

    # Evaluate on training data
    y_train_pred = best_model.predict(X_train_res)
    train_acc = accuracy_score(y_train_res, y_train_pred)

    # Evaluate on test data
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Save model, columns, and accuracy
    joblib.dump(best_model, MODEL_PATH)
    np.save(COLUMNS_PATH, X.columns.values)
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(test_acc))

    # Print results
    print(f"Model saved to {MODEL_PATH}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Display and save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.named_steps['logreg'].classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Logistic Regression (Test Set)')
    plt.savefig('../client/public/confusion_matrix.png')
    print("Confusion matrix saved to ../client/public/confusion_matrix.png")


    # Generate and save ROC curve
    y_score = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('../client/public/roc_curve.png')
    print("ROC curve saved to ../client/public/roc_curve.png")

    plt.show()