from data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Add this import

# File path to your dataset
file_path = "C:\\Users\\14439\\OneDrive\\Desktop\\Research\\Threat-Detection-in-Cyber-Security-Using-AI-master\\Data\\HEROdata2.csv"

# Step 1: Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# Step 2: Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Step 3: Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining and evaluating model: {model_name}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Collect metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    # Print metrics
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}")
    plt.figure(figsize=(6, 5))
    plt.title(f"Confusion Matrix - {model_name}")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {model_name}")
    plt.show()

# Step 4: Visualize Performance Metrics
def plot_metrics(models, accuracy, precision, recall, f1_score):
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, accuracy, width, label='Accuracy', color='skyblue', alpha=0.9)
    plt.bar(x - 0.5 * width, precision, width, label='Precision', color='orange', alpha=0.9)
    plt.bar(x + 0.5 * width, recall, width, label='Recall', color='limegreen', alpha=0.9)
    plt.bar(x + 1.5 * width, f1_score, width, label='F1-Score', color='red', alpha=0.9)

    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Scores', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=45, fontsize=10)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Call the plot_metrics function after all models are evaluated
plot_metrics(
    models=list(models.keys()),
    accuracy=accuracy_list,
    precision=precision_list,
    recall=recall_list,
    f1_score=f1_list
)
