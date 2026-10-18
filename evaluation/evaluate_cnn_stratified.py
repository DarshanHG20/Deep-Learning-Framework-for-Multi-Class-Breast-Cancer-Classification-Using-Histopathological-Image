import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from tensorflow.keras.preprocessing import image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cnn_model.keras"
)

SPLIT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "stratified_split.json"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "cnn_stratified"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 60)
print("CNN STRATIFIED EVALUATION")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nCNN model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError(
        f"\nStratified split not found:\n{SPLIT_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN model loaded successfully.")

print("\nModel output shape:")
print(model.output_shape)


# ============================================================
# LOAD STRATIFIED SPLIT
# ============================================================

print("\nLoading stratified validation split...")

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


# ============================================================
# GET VALIDATION PATHS AND LABELS
# ============================================================

val_paths = split_data["val_paths"]
val_labels = split_data["val_labels"]

class_names = split_data["class_names"]

num_classes = len(class_names)

print("\nNumber of validation images:")
print(len(val_paths))

print("\nNumber of classes:")
print(num_classes)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, name)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_true = []
y_pred = []
y_prob = []


for i, img_path in enumerate(val_paths):

    # Load image
    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    # Convert to array
    img_array = image.img_to_array(
        img
    )

    # IMPORTANT:
    # CNN was trained using raw pixel values.
    # Therefore DO NOT divide by 255.

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_class = np.argmax(
        prediction
    )

    y_true.append(
        val_labels[i]
    )

    y_pred.append(
        predicted_class
    )

    y_prob.append(
        prediction
    )

    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{len(val_paths)} images"
        )


# Convert to NumPy arrays

y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)

y_prob = np.array(
    y_prob
)


# ============================================================
# METRICS
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
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


# ============================================================
# MACRO AUC
# ============================================================

print("\n" + "=" * 60)
print("ROC AUC")
print("=" * 60)

try:

    macro_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

    print(
        f"Macro Average AUC: {macro_auc:.4f}"
    )

except Exception as e:

    print(
        "Could not calculate AUC:"
    )

    print(e)

    macro_auc = 0.0


# ============================================================
# SAVE PROBABILITIES
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


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "CNN",

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "macro_auc": float(
        macro_auc
    ),

    "num_validation_images": int(
        len(val_paths)
    ),

    "num_classes": int(
        num_classes
    )

}


metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CNN STRATIFIED EVALUATION COMPLETED")
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
    f"Macro AUC : {macro_auc * 100:.2f}%"
)

print("\nResults folder:")
print(RESULTS_DIR)

print("\nDONE")