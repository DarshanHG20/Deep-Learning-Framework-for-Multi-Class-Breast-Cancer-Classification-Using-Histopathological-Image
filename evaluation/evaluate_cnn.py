import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(
    BASE_DIR, "saved_models", "cnn_model.keras"
)
CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "saved_models", "class_names.json"
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# =========================================================
# LOAD CLASS NAMES
# =========================================================

print("\nLoading class names...")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nClasses:")
print(class_names)

NUM_CLASSES = len(class_names)

print("\nNumber of classes:", NUM_CLASSES)


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    shuffle=False
)

print("\nDataset classes:")
print(val_ds.class_names)


# =========================================================
# CHECK CLASS ORDER
# =========================================================

if val_ds.class_names != class_names:
    raise RuntimeError(
        f"""
CLASS ORDER MISMATCH!

Dataset classes : {val_ds.class_names}
Saved classes   : {class_names}
"""
    )

print("\nClass order matches successfully.")


# =========================================================
# IMPORTANT:
# DO NOT NORMALIZE HERE
#
# Training was performed using raw image values.
# Therefore evaluation must use the SAME input scale.
# =========================================================

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully.")

print("\nModel output shape:")
print(model.output_shape)


# =========================================================
# PREDICTIONS
# =========================================================

print("\nRunning predictions...")

y_true = []
y_probability = []

batch_count = 0

for images, labels in val_ds:

    predictions = model(
        images,
        training=False
    ).numpy()

    y_true.extend(
        labels.numpy().tolist()
    )

    y_probability.extend(
        predictions.tolist()
    )

    batch_count += 1

    print(
        f"Processed batch {batch_count}",
        end="\r"
    )


print(
    f"\n\nTotal batches processed: {batch_count}"
)


# =========================================================
# CONVERT TO NUMPY
# =========================================================

y_true = np.array(
    y_true,
    dtype=np.int32
)

y_probability = np.array(
    y_probability,
    dtype=np.float32
)

print(
    "Total validation samples:",
    len(y_true)
)

print(
    "Prediction shape:",
    y_probability.shape
)


# =========================================================
# CHECK DATA
# =========================================================

if len(y_true) == 0:
    raise RuntimeError(
        "No validation labels were collected."
    )

if len(y_probability) == 0:
    raise RuntimeError(
        "No predictions were generated."
    )

if len(y_true) != len(y_probability):
    raise RuntimeError(
        "Number of labels and predictions do not match."
    )


# =========================================================
# PREDICTED CLASS
# =========================================================

y_pred = np.argmax(
    y_probability,
    axis=1
)


# =========================================================
# METRICS
# =========================================================

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


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n================================")
print("CNN EVALUATION RESULTS")
print("================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=class_names,
    zero_division=0
)

print(report)


# =========================================================
# SAVE CLASSIFICATION REPORT
# =========================================================

report_path = os.path.join(
    RESULTS_DIR,
    "cnn_classification_report.txt"
)

with open(report_path, "w") as f:

    f.write("CNN EVALUATION RESULTS\n")
    f.write("======================\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("=====================\n\n")

    f.write(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("CNN Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "cnn_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# ROC CURVE
# =========================================================

print("\nGenerating ROC curve...")

y_true_binary = label_binarize(
    y_true,
    classes=np.arange(NUM_CLASSES)
)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(NUM_CLASSES):

    fpr[i], tpr[i], _ = roc_curve(
        y_true_binary[:, i],
        y_probability[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )


# =========================================================
# MICRO AVERAGE ROC
# =========================================================

fpr["micro"], tpr["micro"], _ = roc_curve(
    y_true_binary.ravel(),
    y_probability.ravel()
)

roc_auc["micro"] = auc(
    fpr["micro"],
    tpr["micro"]
)


# =========================================================
# ROC PLOT
# =========================================================

plt.figure(
    figsize=(10, 8)
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
    fpr["micro"],
    tpr["micro"],
    linestyle="--",
    label=(
        f"Micro-average "
        f"(AUC = {roc_auc['micro']:.3f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "CNN Multi-Class ROC Curve"
)

plt.legend(
    loc="lower right",
    fontsize=8
)

plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(
    RESULTS_DIR,
    "cnn_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# SAVE AUC
# =========================================================

auc_path = os.path.join(
    RESULTS_DIR,
    "cnn_auc.txt"
)

with open(auc_path, "w") as f:

    f.write("CNN ROC-AUC RESULTS\n")
    f.write("===================\n\n")

    for i in range(NUM_CLASSES):

        f.write(
            f"{class_names[i]} : "
            f"{roc_auc[i]:.4f}\n"
        )

    f.write(
        f"\nMicro-average AUC : "
        f"{roc_auc['micro']:.4f}\n"
    )


# =========================================================
# DONE
# =========================================================

print("\n================================")
print("EVALUATION COMPLETED")
print("================================")

print("\nClassification report:")
print(report_path)

print("\nConfusion matrix:")
print(cm_path)

print("\nROC curve:")
print(roc_path)

print("\nAUC results:")
print(auc_path)

print("\nDONE")