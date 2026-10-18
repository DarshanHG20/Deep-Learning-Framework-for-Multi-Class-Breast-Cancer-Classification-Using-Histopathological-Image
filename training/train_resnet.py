import os
import sys
import json
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from models.resnet50 import build_resnet50


# =========================================================
# PATHS
# =========================================================

BASE_DIR = PROJECT_ROOT

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

SAVED_MODELS_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    SAVED_MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 64

EPOCHS = 10

VALIDATION_SPLIT = 0.20

SEED = 42

NUM_EXPECTED_CLASSES = 8


# =========================================================
# LOAD DATASET
# =========================================================

print("\n========================================")
print("LOADING DATASET")
print("========================================")

print("\nDataset path:")
print(DATASET_DIR)


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# =========================================================
# CLASS NAMES
# =========================================================

class_names = train_ds.class_names

print("\n========================================")
print("CLASS NAMES")
print("========================================")

for i, name in enumerate(class_names):
    print(i, ":", name)


NUM_CLASSES = len(class_names)

print("\nNumber of classes:", NUM_CLASSES)


# =========================================================
# CHECK NUMBER OF CLASSES
# =========================================================

if NUM_CLASSES != NUM_EXPECTED_CLASSES:

    raise RuntimeError(
        f"Expected {NUM_EXPECTED_CLASSES} classes, "
        f"but found {NUM_CLASSES}"
    )


# =========================================================
# SAVE CLASS NAMES
# =========================================================

class_names_path = os.path.join(
    SAVED_MODELS_DIR,
    "resnet_class_names.json"
)


with open(
    class_names_path,
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


print("\nClass names saved to:")
print(class_names_path)


# =========================================================
# PREFETCH DATA
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# =========================================================
# BUILD RESNET50 MODEL
# =========================================================

print("\n========================================")
print("BUILDING RESNET50 MODEL")
print("========================================")


model = build_resnet50(
    input_shape=(224, 224, 3),
    num_classes=NUM_CLASSES
)


model.summary()


# =========================================================
# MODEL PATH
# =========================================================

model_path = os.path.join(
    SAVED_MODELS_DIR,
    "resnet50.keras"
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
    patience=5,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# =========================================================
# START TRAINING
# =========================================================

print("\n========================================")
print("STARTING RESNET50 TRAINING")
print("========================================")

print("\nEpochs       :", EPOCHS)
print("Batch size   :", BATCH_SIZE)
print("Image size   :", IMG_SIZE)
print("Classes      :", NUM_CLASSES)
print("Validation   :", VALIDATION_SPLIT)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_path = os.path.join(
    RESULTS_DIR,
    "resnet50_history.json"
)


with open(
    history_path,
    "w"
) as f:

    json.dump(
        history.history,
        f,
        indent=4
    )


print("\nTraining history saved to:")
print(history_path)


# =========================================================
# TRAINING ACCURACY GRAPH
# =========================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "ResNet50 Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


accuracy_path = os.path.join(
    RESULTS_DIR,
    "resnet50_accuracy.png"
)


plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# TRAINING LOSS GRAPH
# =========================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)


plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "ResNet50 Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


loss_path = os.path.join(
    RESULTS_DIR,
    "resnet50_loss.png"
)


plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# BEST RESULTS
# =========================================================

best_train_accuracy = max(
    history.history["accuracy"]
)


best_val_accuracy = max(
    history.history["val_accuracy"]
)


best_train_epoch = (
    history.history["accuracy"].index(
        best_train_accuracy
    ) + 1
)


best_val_epoch = (
    history.history["val_accuracy"].index(
        best_val_accuracy
    ) + 1
)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n========================================")
print("RESNET50 TRAINING COMPLETED")
print("========================================")


print(
    f"\nBest Training Accuracy   : "
    f"{best_train_accuracy:.4f}"
)


print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.4f}"
)


print(
    f"Best Training Epoch      : "
    f"{best_train_epoch}"
)


print(
    f"Best Validation Epoch    : "
    f"{best_val_epoch}"
)


print("\n========================================")
print("FILES GENERATED")
print("========================================")


print("\nBest model:")
print(model_path)


print("\nClass names:")
print(class_names_path)


print("\nTraining history:")
print(history_path)


print("\nAccuracy graph:")
print(accuracy_path)


print("\nLoss graph:")
print(loss_path)


print("\n========================================")
print("DONE")
print("========================================")