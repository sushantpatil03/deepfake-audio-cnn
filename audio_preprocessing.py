import numpy as np
import librosa
import tensorflow as tf

SAMPLE_RATE = 16000
DURATION = 3.0          # reduced from 10.0 s → ~3x fewer samples per clip
N_MELS = 64
N_LINEAR_FREQS = 257    # N_FFT // 2 + 1 = 512 // 2 + 1
MAX_TIME_STEPS = 192    # ceil(SAMPLE_RATE * DURATION / HOP_LENGTH) rounded up
BATCH_SIZE = 64         # doubled from 32 → better CPU vectorization
N_FFT = 512             # halved from 1024
HOP_LENGTH = 256        # halved from 512


def ensure_mono_float(audio):
    audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    return audio


def pad_or_trim_audio(audio):
    audio = ensure_mono_float(audio)
    target_len = int(SAMPLE_RATE * DURATION)

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]

    return audio


def prepare_audio(audio, sample_rate=SAMPLE_RATE):
    audio = ensure_mono_float(audio)

    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    return pad_or_trim_audio(audio)


def load_fixed_audio(file_path):
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        mono=True,
    )

    return pad_or_trim_audio(audio)


def fix_time_steps(spec):
    if spec.shape[1] < MAX_TIME_STEPS:
        pad_width = MAX_TIME_STEPS - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :MAX_TIME_STEPS]

    return spec


def normalize_spec(spec):
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return np.expand_dims(spec, axis=-1).astype(np.float32)


def audio_to_spectrograms(audio, sample_rate=SAMPLE_RATE):
    audio = prepare_audio(audio, sample_rate=sample_rate)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = fix_time_steps(librosa.power_to_db(mel_spec, ref=np.max))

    linear_spec = np.abs(
        librosa.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window="hann",
        )
    ) ** 2
    linear_db = fix_time_steps(librosa.power_to_db(linear_spec, ref=np.max))

    return mel_db, linear_db


def audio_to_model_inputs(audio, sample_rate=SAMPLE_RATE):
    mel_db, linear_db = audio_to_spectrograms(audio, sample_rate=sample_rate)

    return {
        "mel_input": np.expand_dims(normalize_spec(mel_db), axis=0),
        "linear_input": np.expand_dims(normalize_spec(linear_db), axis=0),
    }


def file_to_model_inputs(file_path):
    return audio_to_model_inputs(load_fixed_audio(file_path), sample_rate=SAMPLE_RATE)


def process_audio_to_features(file_path_tensor, label_tensor):
    file_path = file_path_tensor.numpy().decode("utf-8")
    label = label_tensor.numpy()
    mel_db, linear_db = audio_to_spectrograms(load_fixed_audio(file_path))

    return normalize_spec(mel_db), normalize_spec(linear_db), np.float32(label)


def tf_parse_audio(file_path, label):
    mel, linear, label = tf.py_function(
        func=process_audio_to_features,
        inp=[file_path, label],
        Tout=[tf.float32, tf.float32, tf.float32],
    )

    mel.set_shape((N_MELS, MAX_TIME_STEPS, 1))
    linear.set_shape((N_LINEAR_FREQS, MAX_TIME_STEPS, 1))
    label.set_shape(())

    return {"mel_input": mel, "linear_input": linear}, label


def create_dataset(paths, labels, batch_size=BATCH_SIZE, training=True, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed)

    ds = ds.map(tf_parse_audio, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
