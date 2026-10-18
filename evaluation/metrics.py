# Accuracy, precision, recall and F1-score.
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Works for binary and multiclass classification.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report
    }


def print_metrics(y_true, y_pred):
    """
    Print evaluation metrics.
    """

    results = calculate_metrics(y_true, y_pred)

    print("\n" + "=" * 55)
    print("              MODEL EVALUATION")
    print("=" * 55)

    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1_score']:.4f}")

    print("\n" + "=" * 55)
    print("              CLASSIFICATION REPORT")
    print("=" * 55)

    print(results["classification_report"])

    return results

if __name__ == "__main__":

    # Example test
    y_true = [
        0, 0, 0, 0,
        1, 1, 1, 1
    ]

    y_pred = [
        0, 0, 1, 0,
        1, 1, 0, 1
    ]

    print_metrics(y_true, y_pred)