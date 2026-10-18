# ============================================================
# RESNET50 - STRATIFIED EVALUATION
# 8-Class Breast Cancer Histopathological Image Classification
# ============================================================

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "resnet50.keras"
)

SPLIT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "stratified_split.json"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "resnet50_stratified"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\n" + "=" * 60)
print("RESNET50 STRATIFIED EVALUATION")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError(
        f"Stratified split not found:\n{SPLIT_PATH}"
    )

print("\nModel : FOUND")
print("Split : FOUND")


# ============================================================
# 4. LOAD STRATIFIED SPLIT
# ============================================================

print("\nLoading stratified split...")

with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)

class_names = split_data["class_names"]
val_paths = split_data["val_paths"]
val_labels = split_data["val_labels"]

NUM_CLASSES = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(
    f"\nValidation images: {len(val_paths)}"
)


# ============================================================
# 5. CHECK VALIDATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 60)

for i, class_name in enumerate(class_names):

    count = val_labels.count(i)

    print(
        f"{class_name:25s}: {count}"
    )


# ============================================================
# 6. LOAD MODEL
# ============================================================

print("\nLoading ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print(
    "Model output shape:",
    model.output_shape
)

if model.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but dataset has {NUM_CLASSES} classes."
    )

print(
    f"Output verification: PASS "
    f"({NUM_CLASSES}-class model)"
)


# ============================================================
# 7. IMAGE LOADING FUNCTION
# ============================================================

def load_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image = tf.keras.utils.img_to_array(
        image
    )

    return image


# ============================================================
# 8. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_true = []
y_prob = []

total_images = len(val_paths)

for i, image_path in enumerate(val_paths):

    try:

        image = load_image(
            image_path
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        # IMPORTANT:
        # ResNet50 model already contains
        # preprocess_input internally.
        #
        # DO NOT divide by 255 here.

        prediction = model.predict(
            image,
            verbose=0
        )[0]

        y_prob.append(
            prediction
        )

        y_true.append(
            val_labels[i]
        )

    except Exception as e:

        print(
            f"\nError reading image: {image_path}"
        )

        print(
            "Error:",
            e
        )

    if (i + 1) % 50 == 0 or (i + 1) == total_images:

        print(
            f"\rProcessed "
            f"{i + 1}/{total_images}",
            end=""
        )


print("\n\nPrediction generation completed.")


# ============================================================
# 9. CONVERT TO NUMPY
# ============================================================

y_true = np.array(
    y_true,
    dtype=np.int32
)

y_prob = np.array(
    y_prob
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

print(
    f"\nTotal evaluated images: {len(y_true)}"
)


# ============================================================
# 10. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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


# ============================================================
# 11. PRINT MAIN METRICS
# ============================================================

print("\n" + "=" * 60)
print("RESNET50 EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall    : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score  : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0
)

print("\n")
print(report)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print("\n")
print(cm)


# ============================================================
# 14. CONFUSION MATRIX GRAPH
# ============================================================

print("\nCreating confusion matrix graph...")

plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ResNet50 Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    NUM_CLASSES
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    class_names
)

threshold = cm.max() / 2.0

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

plt.ylabel(
    "True Label"
)

plt.xlabel(
    "Predicted Label"
)

plt.tight_layout()

confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "resnet50_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Confusion matrix saved:"
)

print(
    confusion_matrix_path
)


# ============================================================
# 15. MULTI-CLASS ROC + AUC
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ROC CURVE AND AUC")
print("=" * 60)

y_true_binary = label_binarize(
    y_true,
    classes=list(range(NUM_CLASSES))
)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(NUM_CLASSES):

    # Every class should have positive samples
    # because stratified split was used.

    fpr[i], tpr[i], _ = roc_curve(
        y_true_binary[:, i],
        y_prob[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )

    print(
        f"{class_names[i]:25s} "
        f"AUC = {roc_auc[i]:.4f}"
    )


# ============================================================
# 16. MACRO AUC
# ============================================================

macro_auc = np.mean(
    list(roc_auc.values())
)

print(
    f"\nMacro Average AUC: "
    f"{macro_auc:.4f}"
)


# ============================================================
# 17. ROC GRAPH
# ============================================================

plt.figure(
    figsize=(12, 10)
)

for i in range(NUM_CLASSES):

    plt.plot(
        fpr[i],
        tpr[i],
        label=(
            f"{class_names[i]} "
            f"(AUC = {roc_auc[i]:.3f})"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ResNet50 Multi-Class ROC Curve"
)

plt.legend(
    loc="lower right",
    fontsize=8
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

roc_path = os.path.join(
    RESULTS_DIR,
    "resnet50_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nROC curve saved:"
)

print(
    roc_path
)


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "resnet50_metrics.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "RESNET50 STRATIFIED EVALUATION\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Validation Samples: "
        f"{len(y_true)}\n\n"
    )

    f.write(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)\n"
    )

    f.write(
        f"Precision : {precision:.4f} "
        f"({precision * 100:.2f}%)\n"
    )

    f.write(
        f"Recall    : {recall:.4f} "
        f"({recall * 100:.2f}%)\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f} "
        f"({f1 * 100:.2f}%)\n"
    )

    f.write(
        f"Macro AUC : {macro_auc:.4f}\n"
    )


# ============================================================
# 19. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "resnet50_classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(report)


# ============================================================
# 20. SAVE NUMPY DATA
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "y_true.npy"
    ),
    y_true
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "y_pred.npy"
    ),
    y_pred
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "y_prob.npy"
    ),
    y_prob
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.npy"
    ),
    cm
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RESNET50 STRATIFIED EVALUATION COMPLETED")
print("=" * 60)

print("\nFINAL METRICS")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"Macro AUC : {macro_auc:.4f}"
)

print("\nResults folder:")
print(RESULTS_DIR)

print("\nDONE")