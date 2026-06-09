import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="BiCrossMambaST")
class SpectroTemporalAttention2D(tf.keras.layers.Layer):
    """Convolutional 2D attention over spectro-temporal feature maps."""

    def __init__(self, hidden_filters=32, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_filters = hidden_filters
        self.dropout_rate = dropout
        self.norm = tf.keras.layers.BatchNormalization()
        self.conv_freq = tf.keras.layers.Conv2D(
            hidden_filters, (5, 1), padding="same", activation="swish"
        )
        self.conv_time = tf.keras.layers.Conv2D(
            hidden_filters, (1, 7), padding="same", activation="swish"
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attn = tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid")

    def call(self, inputs, training=None):
        x = self.norm(inputs, training=training)
        x = self.conv_freq(x)
        x = self.conv_time(x)
        x = self.dropout(x, training=training)
        attention = self.attn(x)
        return inputs * (1.0 + attention)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_filters": self.hidden_filters,
                "dropout": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="BiCrossMambaST")
class SpectralSubBandTokenizer(tf.keras.layers.Layer):
    """Creates a sequence by averaging local spectral sub-bands per time step."""

    def __init__(self, freq_bins=64, time_steps=80, bands=8, **kwargs):
        super().__init__(**kwargs)
        if freq_bins % bands != 0:
            raise ValueError("freq_bins must be divisible by bands.")
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.bands = bands

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[-1]
        band_height = self.freq_bins // self.bands
        x = tf.reshape(
            inputs,
            [batch, self.bands, band_height, self.time_steps, channels],
        )
        x = tf.reduce_mean(x, axis=2)
        return tf.reshape(x, [batch, self.bands * self.time_steps, channels])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.bands * self.time_steps, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "freq_bins": self.freq_bins,
                "time_steps": self.time_steps,
                "bands": self.bands,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="BiCrossMambaST")
class TemporalIntervalTokenizer(tf.keras.layers.Layer):
    """Creates a sequence by averaging local temporal intervals per frequency bin."""

    def __init__(self, freq_bins=64, time_steps=80, intervals=8, **kwargs):
        super().__init__(**kwargs)
        if time_steps % intervals != 0:
            raise ValueError("time_steps must be divisible by intervals.")
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.intervals = intervals

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[-1]
        interval_width = self.time_steps // self.intervals
        x = tf.reshape(
            inputs,
            [batch, self.freq_bins, self.intervals, interval_width, channels],
        )
        x = tf.reduce_mean(x, axis=3)
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch, self.intervals * self.freq_bins, channels])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.intervals * self.freq_bins, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "freq_bins": self.freq_bins,
                "time_steps": self.time_steps,
                "intervals": self.intervals,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="BiCrossMambaST")
class SelectiveSSMMixer(tf.keras.layers.Layer):
    """Small selective state-space mixer inspired by Mamba's recurrent scan."""

    def __init__(self, d_model, expansion=2, kernel_size=5, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.inner_dim = d_model * expansion
        self.in_proj = tf.keras.layers.Dense(self.inner_dim)
        self.conv = tf.keras.layers.Conv1D(
            self.inner_dim,
            kernel_size,
            padding="same",
            groups=self.inner_dim,
            activation="swish",
        )
        self.alpha_proj = tf.keras.layers.Dense(self.inner_dim, bias_initializer="ones")
        self.candidate_proj = tf.keras.layers.Dense(self.inner_dim, activation="tanh")
        self.gate_proj = tf.keras.layers.Dense(self.inner_dim, activation="sigmoid")
        self.out_proj = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.in_proj(inputs)
        x = self.conv(x)
        alpha = tf.sigmoid(self.alpha_proj(x))
        candidate = self.candidate_proj(x)
        gate = self.gate_proj(inputs)

        alpha_t = tf.transpose(alpha, [1, 0, 2])
        candidate_t = tf.transpose(candidate, [1, 0, 2])
        init = tf.zeros_like(candidate[:, 0, :])

        def step(previous_state, current_values):
            current_alpha, current_candidate = current_values
            return current_alpha * previous_state + (1.0 - current_alpha) * current_candidate

        states = tf.scan(step, (alpha_t, candidate_t), initializer=init)
        states = tf.transpose(states, [1, 0, 2])
        x = self.out_proj(states * gate)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "expansion": self.expansion,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="BiCrossMambaST")
class BiMambaBlock(tf.keras.layers.Layer):
    """Bidirectional sequence block using forward/backward selective SSM scans."""

    def __init__(self, d_model, expansion=2, kernel_size=5, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.forward_mixer = SelectiveSSMMixer(d_model, expansion, kernel_size, dropout)
        self.backward_mixer = SelectiveSSMMixer(d_model, expansion, kernel_size, dropout)
        self.fuse = tf.keras.layers.Dense(d_model)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model * 4, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, inputs, training=None):
        x = self.norm(inputs)
        forward = self.forward_mixer(x, training=training)
        backward = tf.reverse(x, axis=[1])
        backward = self.backward_mixer(backward, training=training)
        backward = tf.reverse(backward, axis=[1])
        x = inputs + self.fuse(tf.concat([forward, backward], axis=-1))
        return x + self.ffn(self.ffn_norm(x), training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "expansion": self.expansion,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="BiCrossMambaST")
class MutualCrossAttentionBlock(tf.keras.layers.Layer):
    """Mutual spectral-to-temporal and temporal-to-spectral cross-attention."""

    def __init__(self, d_model, num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        key_dim = max(1, d_model // num_heads)
        self.spectral_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.temporal_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.spectral_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.temporal_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.spectral_ffn = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(d_model * 2, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.temporal_ffn = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(d_model * 2, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
            ]
        )

    def call(self, inputs, training=None):
        spectral, temporal = inputs
        spectral_query = self.spectral_norm(spectral)
        temporal_query = self.temporal_norm(temporal)
        spectral = spectral + self.spectral_attn(
            query=spectral_query,
            value=temporal_query,
            key=temporal_query,
            training=training,
        )
        temporal = temporal + self.temporal_attn(
            query=temporal_query,
            value=spectral_query,
            key=spectral_query,
            training=training,
        )
        spectral = spectral + self.spectral_ffn(spectral, training=training)
        temporal = temporal + self.temporal_ffn(temporal, training=training)
        return spectral, temporal

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
            }
        )
        return config


def _resize_feature_map(inputs, size, name):
    return tf.keras.layers.Resizing(size[0], size[1], interpolation="bilinear", name=name)(
        inputs
    )


def _conv_stem(inputs, d_model, dropout):
    x = tf.keras.layers.Conv2D(d_model // 2, (5, 5), padding="same", activation="swish")(
        inputs
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(d_model, (3, 3), padding="same", activation="swish")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = SpectroTemporalAttention2D(hidden_filters=max(16, d_model // 2), dropout=dropout)(
        x
    )
    return tf.keras.layers.AveragePooling2D(pool_size=(2, 4), name="token_map_pool")(x)


def _sequence_pool(sequence, prefix):
    avg = tf.keras.layers.GlobalAveragePooling1D(name=f"{prefix}_avg_pool")(sequence)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name=f"{prefix}_max_pool")(sequence)
    return tf.keras.layers.Concatenate(name=f"{prefix}_pool_fusion")([avg, max_pool])


def build_bicrossmamba_st(
    mel_input_shape=(64, 192, 1),
    linear_input_shape=(257, 192, 1),
    d_model=96,
    depth=3,
    num_heads=4,
    resized_freq_bins=128,
    resized_time_steps=192,
    pooled_freq_bins=64,
    pooled_time_steps=48,     # 192 / 4 (AveragePooling2D pool_size=(2, 4))
    spectral_bands=8,
    temporal_intervals=8,     # 48 % 8 == 0 ✓
    dropout=0.15,
):
    """Builds a dual-branch BiCrossMamba-ST detector for fair comparison inputs."""

    mel_input = tf.keras.layers.Input(shape=mel_input_shape, name="mel_input")
    linear_input = tf.keras.layers.Input(shape=linear_input_shape, name="linear_input")

    mel = _resize_feature_map(
        mel_input, (resized_freq_bins, resized_time_steps), "mel_resample_grid"
    )
    linear = _resize_feature_map(
        linear_input, (resized_freq_bins, resized_time_steps), "linear_resample_grid"
    )
    x = tf.keras.layers.Concatenate(axis=-1, name="raw_feature_pair")([mel, linear])
    feature_map = _conv_stem(x, d_model=d_model, dropout=dropout)

    spectral = SpectralSubBandTokenizer(
        freq_bins=pooled_freq_bins,
        time_steps=pooled_time_steps,
        bands=spectral_bands,
        name="spectral_subband_tokens",
    )(feature_map)
    temporal = TemporalIntervalTokenizer(
        freq_bins=pooled_freq_bins,
        time_steps=pooled_time_steps,
        intervals=temporal_intervals,
        name="temporal_interval_tokens",
    )(feature_map)

    spectral = tf.keras.layers.Dense(d_model, name="spectral_token_projection")(spectral)
    temporal = tf.keras.layers.Dense(d_model, name="temporal_token_projection")(temporal)

    for index in range(depth):
        spectral = BiMambaBlock(
            d_model=d_model,
            expansion=2,
            kernel_size=5,
            dropout=dropout,
            name=f"spectral_bimamba_{index + 1}",
        )(spectral)
        temporal = BiMambaBlock(
            d_model=d_model,
            expansion=2,
            kernel_size=5,
            dropout=dropout,
            name=f"temporal_bimamba_{index + 1}",
        )(temporal)
        spectral, temporal = MutualCrossAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f"mutual_cross_attention_{index + 1}",
        )([spectral, temporal])

    spectral_features = _sequence_pool(spectral, "spectral")
    temporal_features = _sequence_pool(temporal, "temporal")
    map_features = tf.keras.layers.GlobalAveragePooling2D(name="attention_map_pool")(
        feature_map
    )

    x = tf.keras.layers.Concatenate(name="spectro_temporal_fusion")(
        [spectral_features, temporal_features, map_features]
    )
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="fusion_norm")(x)
    x = tf.keras.layers.Dense(384, activation="gelu", name="fusion_dense_1")(x)
    x = tf.keras.layers.Dropout(dropout, name="fusion_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="gelu", name="fusion_dense_2")(x)
    x = tf.keras.layers.Dropout(dropout, name="fusion_dropout_2")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    return tf.keras.Model(
        inputs={"mel_input": mel_input, "linear_input": linear_input},
        outputs=output,
        name="BiCrossMamba_ST_flagship",
    )


def compile_bicrossmamba_st(model, learning_rate=3e-4):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
