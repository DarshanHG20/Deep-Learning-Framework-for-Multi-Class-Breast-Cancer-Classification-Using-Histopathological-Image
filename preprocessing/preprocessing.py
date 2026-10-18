# ============================================================
# HISTOPATHOLOGY PREPROCESSING & AUGMENTATION PIPELINE
# Breast Cancer XAI Framework (BreakHis 400X)
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads histopathology image from disk, converts BGR to RGB,
    resizes to target dimensions, and casts to float32.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"OpenCV could not decode image at: {image_path}")

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize with bilinear interpolation
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert to float32
    image = image.astype(np.float32)

    return image


def apply_clahe(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    in the LAB color space to enhance cell nuclei and tissue texture
    without distorting H&E (hematoxylin and eosin) stain balance.
    """
    # Convert RGB to LAB
    lab = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    merged = cv2.merge([l_clahe, a_channel, b_channel])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return enhanced.astype(np.float32)


def get_data_augmentation_generator():
    """
    Builds a robust ImageDataGenerator designed for histopathological images
    (rotation invariant, zoom, horizontal and vertical flips).
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect"
    )


def create_dataset_generators(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    validation_split=0.20,
    seed=42
):
    """
    Generates training and validation data generators with stratified splitting.
    """
    train_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        fill_mode="reflect"
    )

    val_datagen = ImageDataGenerator(
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=seed,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=seed,
        shuffle=False
    )

    return train_generator, val_generator


if __name__ == "__main__":
    print("Preprocessing module verified.")
