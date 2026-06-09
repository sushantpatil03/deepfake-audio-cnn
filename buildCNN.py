import tensorflow as tf


def conv_branch(inputs, name_prefix):
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu", name=f"{name_prefix}_conv_1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn_1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool_1")(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu", name=f"{name_prefix}_conv_2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn_2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool_2")(x)

    x = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu", name=f"{name_prefix}_conv_3"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn_3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool_3")(x)

    x = tf.keras.layers.Conv2D(
        256, (3, 3), padding="same", activation="relu", name=f"{name_prefix}_conv_4"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn_4")(x)
    return tf.keras.layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)


def build_audio_deepfake_cnn(
    mel_input_shape=(64, 192, 1),
    linear_input_shape=(257, 192, 1),
):
    mel_input = tf.keras.layers.Input(shape=mel_input_shape, name="mel_input")
    linear_input = tf.keras.layers.Input(shape=linear_input_shape, name="linear_input")

    mel_features = conv_branch(mel_input, "mel")
    linear_features = conv_branch(linear_input, "linear")

    x = tf.keras.layers.Concatenate(name="feature_fusion")(
        [mel_features, linear_features]
    )
    x = tf.keras.layers.Dense(256, activation="relu", name="fusion_dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="fusion_dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="classifier_dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="classifier_dropout")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    model = tf.keras.Model(
        inputs={"mel_input": mel_input, "linear_input": linear_input},
        outputs=output,
        name="dual_frontend_audio_deepfake_cnn",
    )

    return model


def compile_audio_deepfake_cnn(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model
