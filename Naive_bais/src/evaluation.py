import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ['negative', 'neutral', 'positive']

def evaluate_model(model, X_test, y_test, results_dir, model_name):
    preds = model.predict(X_test)

    # Deep learning case
    if hasattr(preds, 'shape') and preds.ndim > 1:
        preds = preds.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=LABELS)

    os.makedirs(results_dir, exist_ok=True)

    # Save report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"{model_name} Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABELS,
                yticklabels=LABELS,
                cmap="Blues")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
