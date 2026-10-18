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

# ==============================
# PATHS
# ==============================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "saved_models", "efficientnetb0.keras"
)

SPLIT_PATH = os.path.join(
    PROJECT_ROOT, "results", "stratified_split.json"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "results", "efficientnet_stratified"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================
# LOAD MODEL
# ==============================
print("=" * 60)
print("EFFICIENTNETB0 STRATIFIED EVALUATION")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)

val_paths = split_data["val_paths"]
val_labels = np.array(split_data["val_labels"])

class_names = split_data["class_names"]
num_classes = len(class_names)

print("\nValidation images:", len(val_paths))
print("Classes:", class_names)

print("\nModel output shape:", model.output_shape)

if model.output_shape[-1] != num_classes:
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but dataset has {num_classes} classes."
    )

print("Output verification: PASS (8-class model)")


# ==============================
# LOAD IMAGES
# ==============================
def load_image(path):
    image = tf.keras.utils.load_img(
        path,
        target_size=(224, 224)
    )

    image = tf.keras.utils.img_to_array(image)

    # EfficientNet preprocessing is handled
    # internally by the Keras EfficientNet model.
    return image


# ==============================
# PREDICTIONS
# ==============================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

predictions = []

for i, path in enumerate(val_paths):

    image = load_image(path)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predictions.append(prediction)

    if (i + 1) % 200 == 0:
        print(f"Processed {i + 1}/{len(val_paths)}")

predictions = np.array(predictions)

y_pred = np.argmax(predictions, axis=1)

print("Prediction generation completed.")


# ==============================
# METRICS
# ==============================
accuracy = accuracy_score(
    val_labels,
    y_pred
)

precision = precision_score(
    val_labels,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    val_labels,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    val_labels,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    val_labels,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(report)


# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(
    val_labels,
    y_pred
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

np.save(
    os.path.join(
        RESULTS_DIR,
        "efficientnet_confusion_matrix.npy"
    ),
    cm
)


# ==============================
# MACRO AUC
# ==============================
print("\n" + "=" * 60)
print("GENERATING ROC CURVE AND AUC")
print("=" * 60)

y_true_onehot = tf.keras.utils.to_categorical(
    val_labels,
    num_classes=num_classes
)

auc_scores = []

for i, class_name in enumerate(class_names):

    try:
        auc = roc_auc_score(
            y_true_onehot[:, i],
            predictions[:, i]
        )

        auc_scores.append(auc)

        print(
            f"{class_name:<25} AUC = {auc:.4f}"
        )

    except ValueError:
        print(
            f"{class_name:<25} AUC unavailable"
        )

macro_auc = np.mean(auc_scores)

print(
    f"\nMacro Average AUC: {macro_auc:.4f}"
)


# ==============================
# SAVE RESULTS
# ==============================
with open(
    os.path.join(
        RESULTS_DIR,
        "efficientnet_classification_report.txt"
    ),
    "w"
) as f:
    f.write(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "efficientnet_metrics.txt"
    ),
    "w"
) as f:

    f.write("EFFICIENTNETB0 STRATIFIED EVALUATION\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Accuracy  : {accuracy * 100:.2f}%\n")
    f.write(f"Precision : {precision * 100:.2f}%\n")
    f.write(f"Recall    : {recall * 100:.2f}%\n")
    f.write(f"F1 Score  : {f1 * 100:.2f}%\n")
    f.write(f"Macro AUC : {macro_auc * 100:.2f}%\n")


np.save(
    os.path.join(
        RESULTS_DIR,
        "efficientnet_predictions.npy"
    ),
    predictions
)


# ==============================
# FINAL OUTPUT
# ==============================
print("\n" + "=" * 60)
print("EFFICIENTNETB0 STRATIFIED EVALUATION COMPLETED")
print("=" * 60)

print("\nFINAL METRICS")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"Macro AUC : {macro_auc * 100:.2f}%")

print("\nResults folder:")
print(RESULTS_DIR)

print("\nDONE")