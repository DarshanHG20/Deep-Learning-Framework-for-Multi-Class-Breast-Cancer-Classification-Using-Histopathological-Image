# ============================================================
# RESNET50 GRAD-CAM EXPLAINABILITY
# Breast Cancer Histopathological Image Classification
# ============================================================

import os
import json
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root = parent of explainability folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "resnet50.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "resnet_class_names.json"
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "gradcam_resnet50_result.jpg"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# PRINT HEADER
# ============================================================

def print_header():

    print()
    print("=" * 60)
    print("RESNET50 GRAD-CAM EXPLAINABILITY")
    print("=" * 60)
    print()


# ============================================================
# FIND FIRST IMAGE IN DATASET
# ============================================================

def find_dataset_image():

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_DIR}"
        )

    for root, dirs, files in os.walk(DATASET_DIR):

        for file in files:

            if file.lower().endswith(valid_extensions):

                return os.path.join(
                    root,
                    file
                )

    raise FileNotFoundError(
        "No image found inside dataset folder."
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    print("Loading class names...")

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    # --------------------------------------------------------
    # Handle dictionary format
    # --------------------------------------------------------

    if isinstance(class_names, dict):

        # Example:
        # {"0": "adenosis", "1": "ductal_carcinoma"}

        if all(
            str(i) in class_names
            for i in range(len(class_names))
        ):

            class_names = [
                class_names[str(i)]
                for i in range(len(class_names))
            ]

        else:

            class_names = list(
                class_names.values()
            )

    # --------------------------------------------------------
    # Handle list format
    # --------------------------------------------------------

    if not isinstance(class_names, list):
        raise ValueError(
            "Invalid class names JSON format."
        )

    print()

    for i, name in enumerate(class_names):

        print(f"{i}: {name}")

    print()

    return class_names


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    print("Loading image...")

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            f"Could not load image:\n{image_path}"
        )

    # OpenCV BGR -> RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Keep original image for overlay
    original_image = image_rgb.copy()

    # Resize for ResNet50
    resized = cv2.resize(
        image_rgb,
        IMAGE_SIZE
    )

    # Float32
    resized = resized.astype(
        np.float32
    )

    # IMPORTANT:
    # Do NOT divide by 255 here.
    #
    # Your ResNet50 model/training pipeline uses
    # ResNet preprocessing.

    # Add batch dimension
    input_image = np.expand_dims(
        resized,
        axis=0
    )

    return input_image, original_image


# ============================================================
# FIND RESNET50 BASE MODEL
# ============================================================

def find_resnet_base(model):

    print("Searching for ResNet50 base model...")

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            if "resnet" in layer.name.lower():

                print(
                    "Base model:",
                    layer.name
                )

                return layer

    raise ValueError(
        "ResNet50 base model could not be found."
    )


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(base_model):

    print(
        "Searching for last convolutional layer..."
    )

    for layer in reversed(
        base_model.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            print(
                "Grad-CAM layer:",
                layer.name
            )

            return layer

    raise ValueError(
        "No Conv2D layer found in ResNet50."
    )


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image,
    class_names
):

    print()
    print("Generating Grad-CAM...")

    # --------------------------------------------------------
    # Find ResNet50 base
    # --------------------------------------------------------

    base_model = find_resnet_base(
        model
    )

    # --------------------------------------------------------
    # Find last convolution layer
    # --------------------------------------------------------

    conv_layer = find_last_conv_layer(
        base_model
    )

    # --------------------------------------------------------
    # Find position of ResNet base
    # --------------------------------------------------------

    base_index = None

    for i, layer in enumerate(
        model.layers
    ):

        if layer is base_model:

            base_index = i
            break

    if base_index is None:

        raise ValueError(
            "Could not determine ResNet50 position."
        )

    # --------------------------------------------------------
    # Create model directly from ResNet50 base
    #
    # This is important for nested Keras models.
    #
    # Output:
    #   1. Last convolution feature maps
    #   2. ResNet50 feature output
    # --------------------------------------------------------

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            conv_layer.output,
            base_model.output
        ]
    )

    # --------------------------------------------------------
    # Apply layers BEFORE ResNet50
    #
    # Example:
    # Input layer
    # ResNet preprocessing Lambda
    # --------------------------------------------------------

    x = image

    for layer in model.layers[
        :base_index
    ]:

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):
            continue

        x = layer(
            x,
            training=False
        )

    # --------------------------------------------------------
    # Gradient Tape
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # ----------------------------------------------
        # Run ResNet50
        # ----------------------------------------------

        conv_outputs, base_output = grad_model(
            x,
            training=False
        )

        # ----------------------------------------------
        # Pass ResNet output through classifier head
        # ----------------------------------------------

        classifier_output = base_output

        for layer in model.layers[
            base_index + 1:
        ]:

            if isinstance(
                layer,
                tf.keras.layers.InputLayer
            ):
                continue

            classifier_output = layer(
                classifier_output,
                training=False
            )

        predictions = classifier_output

        # ----------------------------------------------
        # Find predicted class
        # ----------------------------------------------

        predicted_class = tf.argmax(
            predictions[0]
        )

        # ----------------------------------------------
        # Score for predicted class
        # ----------------------------------------------

        class_score = predictions[
            0,
            predicted_class
        ]

    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    if gradients is None:

        raise ValueError(
            "\nGradients are None.\n"
            "Grad-CAM could not calculate gradients "
            "through ResNet50."
        )

    print(
        "Gradients calculated successfully."
    )

    # --------------------------------------------------------
    # Global Average Pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # Calculate weighted feature maps
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_heatmap = tf.reduce_max(
        heatmap
    )

    if float(max_heatmap) > 0:

        heatmap = (
            heatmap /
            max_heatmap
        )

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_array = predictions[
        0
    ].numpy()

    predicted_index = int(
        np.argmax(
            prediction_array
        )
    )

    confidence = float(
        prediction_array[
            predicted_index
        ]
    )

    predicted_class_name = class_names[
        predicted_index
    ]

    # --------------------------------------------------------
    # Print prediction
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("PREDICTION RESULT")
    print("-" * 60)

    print(
        "Predicted Class:",
        predicted_class_name
    )

    print(
        "Class Index:",
        predicted_index
    )

    print(
        "Confidence: {:.2f}%".format(
            confidence * 100
        )
    )

    print("-" * 60)

    return (
        heatmap,
        predicted_class_name,
        confidence
    )


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    heatmap,
    original_image
):

    # --------------------------------------------------------
    # Resize heatmap to original image size
    # --------------------------------------------------------

    height, width = original_image.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    # --------------------------------------------------------
    # Convert 0-1 -> 0-255
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # --------------------------------------------------------
    # Apply JET color map
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # RGB -> BGR
    # --------------------------------------------------------

    original_bgr = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_bgr,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    return overlay


# ============================================================
# SAVE GRAD-CAM
# ============================================================

def save_gradcam(
    model,
    image_path,
    output_path,
    class_names
):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image, original_image = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    heatmap, predicted_class, confidence = (
        generate_gradcam(
            model,
            image,
            class_names
        )
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = create_gradcam_overlay(
        heatmap,
        original_image
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_path
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    success = cv2.imwrite(
        output_path,
        overlay
    )

    if not success:

        raise IOError(
            f"Could not save Grad-CAM image:\n{output_path}"
        )

    print()
    print("=" * 60)
    print("GRAD-CAM COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()
    print(
        "Input image:"
    )

    print(
        image_path
    )

    print()
    print(
        "Predicted class:"
    )

    print(
        predicted_class
    )

    print()
    print(
        "Confidence:"
    )

    print(
        "{:.2f}%".format(
            confidence * 100
        )
    )

    print()
    print(
        "Grad-CAM output:"
    )

    print(
        output_path
    )

    print()
    print("=" * 60)

    return overlay


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_header()

    # --------------------------------------------------------
    # Find input image
    # --------------------------------------------------------

    image_path = find_dataset_image()

    print(
        "Input image:"
    )

    print(
        image_path
    )

    print()

    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    class_names = load_class_names()

    # --------------------------------------------------------
    # Check number of classes
    # --------------------------------------------------------

    if len(class_names) != 8:

        raise ValueError(
            f"Expected 8 classes, "
            f"but found {len(class_names)} classes."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading ResNet50 model..."
    )

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"ResNet50 model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "ResNet50 model loaded successfully."
    )

    print()

    print(
        "Model output shape:",
        model.output_shape
    )

    # --------------------------------------------------------
    # Check output classes
    # --------------------------------------------------------

    if model.output_shape[-1] != 8:

        raise ValueError(
            "Model does not have 8 output classes."
        )

    # --------------------------------------------------------
    # Generate and save Grad-CAM
    # --------------------------------------------------------

    save_gradcam(
        model=model,
        image_path=image_path,
        output_path=OUTPUT_PATH,
        class_names=class_names
    )