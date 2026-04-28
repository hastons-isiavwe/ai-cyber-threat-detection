import numpy as np
import matplotlib.pyplot as plt

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
