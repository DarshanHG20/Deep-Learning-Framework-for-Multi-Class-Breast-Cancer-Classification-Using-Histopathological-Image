# Confusion matrix generation.
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def generate_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    save_path="results/confusion_matrix.png"
):
    """
    Generate and save confusion matrix.

    Parameters:
        y_true       : Actual class labels
        y_pred       : Predicted class labels
        class_names  : List of class names
        save_path    : Location to save the image

    Returns:
        Confusion matrix
    """

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Default class names
    if class_names is None:
        class_names = ["Benign", "Malignant"]

    # Create results folder
    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True
    )

    # Calculate confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names))
    )

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")
    print(cm)

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True
    )

    ax.set_title("Breast Cancer Classification - Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    plt.tight_layout()

    # Save figure
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nConfusion matrix saved to:")
    print(save_path)

    return cm


if __name__ == "__main__":

    # Test data
    y_true = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ])

    y_pred = np.array([
        0, 0, 0, 1, 0,
        1, 1, 0, 1, 1
    ])

    class_names = [
        "Benign",
        "Malignant"
    ]

    generate_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        "results/confusion_matrix.png"
    )