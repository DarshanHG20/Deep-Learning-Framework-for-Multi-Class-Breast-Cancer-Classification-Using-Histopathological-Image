# ============================================================
# DenseNet121 Model Architecture for Breast Cancer Classification
# BreakHis Histopathological Image Classification
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_densenet121(
    input_shape=(224, 224, 3),
    num_classes=8,
    freeze_base=True
):
    """
    Constructs a DenseNet121 transfer learning architecture with dense connectivity
    for feature reuse and gradient flow across deep histopathological representations.
    """
    # Load pretrained DenseNet121 feature extractor
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    if freeze_base:
        base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    # Preprocessing for DenseNet (scales pixel values between 0 and 1, standardizes)
    x = tf.keras.applications.densenet.preprocess_input(inputs)

    # DenseNet feature extraction
    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="densenet121_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_densenet121()
    model.summary()
