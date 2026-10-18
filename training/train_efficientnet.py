import os
import sys
import json
import tensorflow as tf
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from models.efficientnet import build_efficientnet


# ==============================
# SETTINGS
# ==============================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10
VAL_SPLIT = 0.20
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================
# LOAD DATASET
# ==============================
print("=" * 60)
print("EFFICIENTNETB0 TRAINING")
print("=" * 60)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)

print("\nNumber of classes:", num_classes)


# ==============================
# SAVE CLASS NAMES
# ==============================
class_names_path = os.path.join(
    MODEL_DIR,
    "efficientnet_class_names.json"
)

with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=4)


# ==============================
# PERFORMANCE
# ==============================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ==============================
# BUILD MODEL
# ==============================
model = build_efficientnet(
    num_classes=num_classes
)

print("\nModel output shape:")
print(model.output_shape)

model.summary()


# ==============================
# CALLBACKS
# ==============================
model_path = os.path.join(
    MODEL_DIR,
    "efficientnetb0.keras"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
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


# ==============================
# TRAIN
# ==============================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

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


# ==============================
# SAVE FINAL MODEL
# ==============================
model.save(model_path)


# ==============================
# SAVE HISTORY
# ==============================
history_path = os.path.join(
    RESULTS_DIR,
    "efficientnet_history.json"
)

with open(history_path, "w") as f:
    json.dump(
        {
            key: [float(x) for x in values]
            for key, values in history.history.items()
        },
        f,
        indent=4
    )


# ==============================
# ACCURACY GRAPH
# ==============================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNetB0 Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "efficientnet_accuracy.png"
    )
)

plt.close()


# ==============================
# LOSS GRAPH
# ==============================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNetB0 Loss")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "efficientnet_loss.png"
    )
)

plt.close()


# ==============================
# FINAL OUTPUT
# ==============================
best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_train_accuracy = max(
    history.history["accuracy"]
)

best_epoch = (
    history.history["val_accuracy"].index(
        best_val_accuracy
    ) + 1
)

print("\n" + "=" * 60)
print("EFFICIENTNETB0 TRAINING COMPLETED")
print("=" * 60)

print(
    f"\nBest Training Accuracy   : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(f"Best Epoch               : {best_epoch}")

print("\nModel:")
print(model_path)

print("\nClass names:")
print(class_names_path)

print("\nDONE")