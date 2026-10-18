import tensorflow as tf
from tensorflow.keras import layers, models


def build_efficientnet(num_classes=8):

    # Load pretrained EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build complete model
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            256,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model