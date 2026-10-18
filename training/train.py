import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.cnn import build_cnn


# =========================================================
# PATHS
# =========================================================

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.20
SEED = 42


# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

print("\n======================================")
print("BREAST CANCER CNN TRAINING")
print("======================================")


# =========================================================
# COLLECT IMAGE PATHS
# =========================================================

print("\nCollecting image paths...")

class_names = sorted([
    folder
    for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

print("\nNumber of classes:", num_classes)

if num_classes != 8:
    raise ValueError(
        f"Expected 8 classes, found {num_classes}"
    )


image_paths = []
labels = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    for root, _, files in os.walk(class_dir):

        for file in files:

            if file.lower().endswith(valid_extensions):

                image_paths.append(
                    os.path.join(root, file)
                )

                labels.append(class_index)


image_paths = np.array(image_paths)
labels = np.array(labels)

print("\nTotal images:", len(image_paths))


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

print("\nCreating stratified train/validation split...")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=VALIDATION_SPLIT,
    random_state=SEED,
    stratify=labels
)

print("\nTraining images:", len(train_paths))
print("Validation images:", len(val_paths))


# =========================================================
# SAVE VALIDATION FILES
# =========================================================

val_info = []

for path, label in zip(val_paths, val_labels):

    val_info.append({
        "path": os.path.relpath(path, BASE_DIR),
        "label": int(label),
        "class_name": class_names[label]
    })

val_info_path = os.path.join(
    SAVED_MODEL_DIR,
    "validation_files.json"
)

with open(val_info_path, "w") as f:
    json.dump(
        val_info,
        f,
        indent=4
    )

print("\nValidation file list saved:")
print(val_info_path)


# =========================================================
# SAVE CLASS NAMES
# =========================================================

class_path = os.path.join(
    SAVED_MODEL_DIR,
    "class_names.json"
)

with open(class_path, "w") as f:
    json.dump(
        class_names,
        f,
        indent=4
    )


# =========================================================
# CLASS WEIGHTS
# =========================================================

print("\n======================================")
print("CLASS WEIGHTS")
print("======================================")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)

class_weights = {
    int(i): float(class_weights_array[i])
    for i in range(num_classes)
}

for i, name in enumerate(class_names):

    print(
        f"{name}: {class_weights[i]:.4f}"
    )


class_weights_path = os.path.join(
    SAVED_MODEL_DIR,
    "class_weights.json"
)

with open(class_weights_path, "w") as f:

    json.dump(
        {
            class_names[i]: class_weights[i]
            for i in range(num_classes)
        },
        f,
        indent=4
    )


# =========================================================
# IMAGE LOADING FUNCTION
# =========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


# =========================================================
# CREATE DATASETS
# =========================================================

print("\nCreating TensorFlow datasets...")

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# BUILD MODEL
# =========================================================

print("\n======================================")
print("BUILDING CNN MODEL")
print("======================================")

model = build_cnn(
    input_shape=(224, 224, 3),
    num_classes=num_classes
)

model.summary()


# =========================================================
# MODEL PATH
# =========================================================

model_path = os.path.join(
    SAVED_MODEL_DIR,
    "cnn_model.keras"
)


# =========================================================
# CALLBACKS
# =========================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


# =========================================================
# TRAIN
# =========================================================

print("\n======================================")
print("STARTING CNN TRAINING")
print("======================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# =========================================================
# LOAD BEST MODEL
# =========================================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    model_path
)


# =========================================================
# VERIFY BEST MODEL
# =========================================================

print("\n======================================")
print("VERIFYING BEST MODEL")
print("======================================")

loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print(
    f"\nBest Model Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Best Model Validation Loss: "
    f"{loss:.4f}"
)


# =========================================================
# SAVE HISTORY
# =========================================================

history_path = os.path.join(
    RESULTS_DIR,
    "cnn_training_history.json"
)

with open(history_path, "w") as f:

    json.dump(
        history.history,
        f,
        indent=4
    )


# =========================================================
# ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "cnn_accuracy.png"
    ),
    dpi=300
)

plt.close()


# =========================================================
# LOSS GRAPH
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "CNN Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "cnn_loss.png"
    ),
    dpi=300
)

plt.close()


# =========================================================
# FINAL
# =========================================================

best_train_accuracy = max(
    history.history["accuracy"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)

print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")

print(
    f"Best Training Accuracy: "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Epoch: {best_epoch}"
)

print("\nSaved:")
print("saved_models/cnn_model.keras")
print("saved_models/class_names.json")
print("saved_models/class_weights.json")
print("saved_models/validation_files.json")
print("results/cnn_training_history.json")
print("results/cnn_accuracy.png")
print("results/cnn_loss.png")

print("\n======================================")
print("DONE")
print("======================================")