# ResNet50 model for 8-class BreakHis classification

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_resnet50(
    input_shape=(224, 224, 3),
    num_classes=8
):

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze ImageNet feature extractor
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    # ResNet50 preprocessing
    x = tf.keras.applications.resnet50.preprocess_input(inputs)

    # Feature extraction
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.5)(x)

    # 8 BreakHis classes
    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_resnet50()

    model.summary()