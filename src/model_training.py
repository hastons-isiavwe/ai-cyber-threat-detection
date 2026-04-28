from data_preprocessing import load_and_preprocess_data
from model_training import train_model
from evaluation import evaluate_model
from visualization import plot_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# File path to your dataset
file_path = "C:\\Users\\14439\\OneDrive\\Desktop\\Research\\Threat-Detection-in-Cyber-Security-Using-AI-master\\Data\\HEROdata2.csv"

# Step 1: Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# Step 2: Define models (one-time training for each distinct model)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    # Add other models here if needed
}

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Step 3: Train and evaluate each model
for model_name, model in models.items():
    print(f"Training and evaluating model: {model_name}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Collect metrics
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, average="weighted"))
    recall_list.append(recall_score(y_test, y_pred, average="weighted"))
    f1_list.append(f1_score(y_test, y_pred, average="weighted"))
    
    # Optionally, plot confusion matrix and ROC curve
    evaluate_model(model, X_test, y_test)

# Step 4: Visualize Performance Metrics
plot_metrics(
    models=list(models.keys()),
    accuracy=accuracy_list,
    precision=precision_list,
    recall=recall_list,
    f1_score=f1_list
)
