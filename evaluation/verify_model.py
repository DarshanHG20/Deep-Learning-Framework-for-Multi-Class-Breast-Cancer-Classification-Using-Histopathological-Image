import os
import tensorflow as tf

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "cnn_model.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

val_ds = val_ds.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")

print("\nEvaluating saved CNN...")

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\n==============================")
print("SAVED MODEL VERIFICATION")
print("==============================")

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")

print("==============================")