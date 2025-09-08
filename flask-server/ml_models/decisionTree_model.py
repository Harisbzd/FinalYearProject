import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
DATA_PATH = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a decision tree
dt = DecisionTreeClassifier(
    criterion="gini",   # or "entropy"
    max_depth=5,        # limit depth to avoid overfitting
    random_state=42
)

dt.fit(X_train, y_train)

# Predictions
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_acc)
print("Validation Accuracy:", test_acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
