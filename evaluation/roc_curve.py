# ROC-AUC and ROC curve generation.
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


def generate_roc_curve(
    y_true,
    y_probability,
    save_path="results/roc_curve.png"
):
    """
    Generate and save ROC curve for binary classification.

    Parameters:
        y_true        : Actual labels (0 or 1)
        y_probability : Probability of malignant class
        save_path     : Location to save ROC curve

    Returns:
        ROC-AUC score
    """

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    # Flatten probability array
    y_probability = y_probability.reshape(-1)

    # Create results folder
    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True
    )

    # Calculate ROC values
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_probability
    )

    # Calculate AUC
    auc_score = roc_auc_score(
        y_true,
        y_probability
    )

    print("\n==============================")
    print("ROC CURVE")
    print("==============================")
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # Create plot
    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC Curve (AUC = {auc_score:.4f})"
    )

    # Random classifier line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "Breast Cancer Classification - ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nROC curve saved to:")
    print(save_path)

    return auc_score


if __name__ == "__main__":

    # Test data
    y_true = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    # Probability of malignant class
    y_probability = np.array([
        0.05,
        0.10,
        0.20,
        0.35,
        0.40,
        0.60,
        0.75,
        0.80,
        0.90,
        0.95
    ])

    generate_roc_curve(
        y_true,
        y_probability,
        "results/roc_curve.png"
    )