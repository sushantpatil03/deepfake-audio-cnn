# Mel + Linear Spectrogram CNN Explained

This document explains the older baseline model in this project: the dual-input
CNN that uses a mel spectrogram and a linear-frequency spectrogram from the same
audio clip.

The important files are:

- `audio_preprocessing.py`: converts audio into model-ready spectrogram tensors.
- `buildCNN.py`: defines and compiles the CNN model.
- `main.py`: collects the dataset, trains the model, saves the best checkpoint,
  and evaluates validation performance.
- `realtime_mic_demo.py`: uses a trained checkpoint for microphone inference.
- `predict_audio.py`: uses a trained checkpoint for file-based inference.

## High-Level Idea

The model tries to classify an audio clip as real or fake by looking at two
different spectrogram views:

- Mel spectrogram: a perceptual frequency representation that roughly follows
  how humans hear sound.
- Linear spectrogram: a regular STFT frequency representation that preserves
  more raw frequency-bin detail.

Both are extracted from the same fixed-length audio window. The model processes
each view with its own CNN branch, merges the two learned feature vectors, and
then uses dense layers to produce one fake probability.

The output is:

```text
fake_probability in range [0, 1]
```

Interpretation:

- Close to `0.0`: model thinks the audio is real.
- Close to `1.0`: model thinks the audio is fake.
- Classification threshold defaults to `0.5`.

## Audio Preprocessing

The preprocessing code lives in `audio_preprocessing.py`.

### Core Constants

```python
SAMPLE_RATE = 16000
DURATION = 10.0
N_MELS = 64
N_LINEAR_FREQS = 513
MAX_TIME_STEPS = 320
BATCH_SIZE = 32
N_FFT = 1024
HOP_LENGTH = 512
```

What each parameter does:

- `SAMPLE_RATE = 16000`: all audio is converted to 16 kHz. This standardizes
  every file so the model sees the same number of samples per second.
- `DURATION = 10.0`: every training or inference example is exactly 10 seconds.
  Shorter audio is padded with silence. Longer audio is trimmed.
- `N_MELS = 64`: the mel spectrogram has 64 mel frequency bands.
- `N_LINEAR_FREQS = 513`: the linear spectrogram has 513 frequency bins. This
  comes from `N_FFT // 2 + 1`, because an STFT on real audio keeps the positive
  frequency bins.
- `MAX_TIME_STEPS = 320`: every spectrogram is forced to 320 time frames.
- `BATCH_SIZE = 32`: default number of examples processed together during
  training.
- `N_FFT = 1024`: FFT window size used for spectrogram extraction. A larger
  value gives better frequency resolution but worse time precision.
- `HOP_LENGTH = 512`: number of audio samples between adjacent STFT frames. At
  16 kHz, this is 32 milliseconds per hop.

### Audio Standardization

The function `ensure_mono_float(audio)` makes audio safe for feature extraction:

- Converts stereo or multi-channel audio to mono by averaging channels.
- Converts integer PCM samples to floating-point values.
- Ensures output type is `np.float32`.

The function `pad_or_trim_audio(audio)` ensures each clip has:

```text
SAMPLE_RATE * DURATION = 16000 * 10 = 160000 samples
```

If the clip is shorter than 160000 samples, zeros are appended. If it is longer,
only the first 160000 samples are used.

The function `prepare_audio(audio, sample_rate)` also resamples audio to 16 kHz
if needed.

### Spectrogram Generation

The function `audio_to_spectrograms(audio, sample_rate)` creates two
representations.

Mel spectrogram:

```python
librosa.feature.melspectrogram(
    y=audio,
    sr=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
)
```

This produces a power spectrogram shaped approximately:

```text
(64, time_frames)
```

Linear spectrogram:

```python
np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
```

This produces a power spectrogram shaped approximately:

```text
(513, time_frames)
```

Both spectrograms are converted to decibels using:

```python
librosa.power_to_db(spec, ref=np.max)
```

Why decibels are used:

- Audio energy has a large numeric range.
- Decibel scaling compresses this range.
- CNNs usually learn better from compressed, normalized values.

### Time-Step Fixing

The function `fix_time_steps(spec)` makes every spectrogram exactly 320 frames:

- If fewer than 320 frames, it pads zeros on the time axis.
- If more than 320 frames, it cuts off after 320 frames.

Final raw spectrogram shapes before normalization:

```text
mel_db:    (64, 320)
linear_db: (513, 320)
```

### Normalization

The function `normalize_spec(spec)` applies per-example z-score normalization:

```python
spec = (spec - spec.mean()) / (spec.std() + 1e-6)
```

Meaning:

- Subtracting the mean centers the spectrogram around zero.
- Dividing by standard deviation normalizes scale.
- `1e-6` prevents division by zero.

The function then adds a channel dimension:

```text
(frequency_bins, time_steps) -> (frequency_bins, time_steps, 1)
```

Final model input shapes:

```text
mel_input:    (64, 320, 1)
linear_input: (513, 320, 1)
```

For prediction, the batch dimension is added:

```text
mel_input:    (1, 64, 320, 1)
linear_input: (1, 513, 320, 1)
```

## Dataset Pipeline

The function `create_dataset(paths, labels, batch_size, training, seed)` builds a
TensorFlow dataset.

What each argument means:

- `paths`: list or array of audio file paths.
- `labels`: numeric labels. In this project, `1.0` means fake and `0.0` means
  real.
- `batch_size`: number of examples in each training batch.
- `training`: if `True`, the dataset is shuffled.
- `seed`: controls reproducible shuffling.

The pipeline does this:

1. Builds a dataset from file paths and labels.
2. Shuffles if training.
3. Loads and converts audio using `tf.py_function`.
4. Sets static tensor shapes so Keras knows the input dimensions.
5. Batches examples.
6. Prefetches examples for faster training.

The model receives each batch as a dictionary:

```python
{
    "mel_input": mel_tensor,
    "linear_input": linear_tensor,
}
```

This must match the input names in `buildCNN.py`.

## Model Architecture

The model is defined in `buildCNN.py`.

Main builder:

```python
build_audio_deepfake_cnn(
    mel_input_shape=(64, 320, 1),
    linear_input_shape=(513, 320, 1),
)
```

### Inputs

Two Keras input layers are created:

```python
mel_input = Input(shape=(64, 320, 1), name="mel_input")
linear_input = Input(shape=(513, 320, 1), name="linear_input")
```

The input names matter because the dataset returns a dictionary with these exact
keys.

### CNN Branch Function

Both spectrograms are passed through the same branch structure:

```python
conv_branch(inputs, name_prefix)
```

The weights are not shared. The branch structure is reused, but each branch gets
its own layers because each call creates new Keras layers.

Each branch contains:

1. Conv2D with 32 filters.
2. BatchNormalization.
3. MaxPooling2D.
4. Conv2D with 64 filters.
5. BatchNormalization.
6. MaxPooling2D.
7. Conv2D with 128 filters.
8. BatchNormalization.
9. MaxPooling2D.
10. Conv2D with 256 filters.
11. BatchNormalization.
12. GlobalAveragePooling2D.

### Conv2D Layers

Example:

```python
Conv2D(32, (3, 3), padding="same", activation="relu")
```

Parameter meanings:

- `32`: number of filters. Each filter learns a different local pattern.
- `(3, 3)`: kernel size. The filter looks at a 3 by 3 local patch.
- `padding="same"`: output keeps the same height and width before pooling.
- `activation="relu"`: negative values are set to zero, helping nonlinear
  learning.

As the branch gets deeper, filter count increases:

```text
32 -> 64 -> 128 -> 256
```

Why this is common:

- Early layers learn simple local patterns.
- Later layers learn more abstract patterns.
- Increasing filters gives later layers more representational capacity.

### BatchNormalization

Batch normalization normalizes intermediate activations during training.

Benefits:

- Can stabilize training.
- Can allow faster convergence.
- Reduces sensitivity to initialization.

### MaxPooling2D

Example:

```python
MaxPooling2D((2, 2))
```

This halves both frequency and time dimensions.

Effect:

- Reduces computation.
- Makes features more spatially compact.
- Gives the model some tolerance to small shifts in time or frequency.

Approximate mel branch shape flow:

```text
(64, 320, 1)
-> conv 32
-> pool: (32, 160, 32)
-> conv 64
-> pool: (16, 80, 64)
-> conv 128
-> pool: (8, 40, 128)
-> conv 256
-> global average pool: (256)
```

Approximate linear branch shape flow:

```text
(513, 320, 1)
-> conv 32
-> pool: (256, 160, 32)
-> conv 64
-> pool: (128, 80, 64)
-> conv 128
-> pool: (64, 40, 128)
-> conv 256
-> global average pool: (256)
```

Because pooling uses floor behavior for odd dimensions, exact intermediate
linear shapes can differ slightly depending on TensorFlow shape calculation. The
important point is that `GlobalAveragePooling2D` reduces each branch to one
fixed-length vector.

### GlobalAveragePooling2D

This layer averages every feature map across frequency and time.

Input example:

```text
(height, width, 256)
```

Output:

```text
(256)
```

Why it is used:

- It creates a fixed-size vector regardless of remaining spatial dimensions.
- It reduces parameter count compared with flattening.
- It forces each filter to represent whether a learned pattern exists anywhere
  in the spectrogram.

### Feature Fusion

The two branch outputs are concatenated:

```python
Concatenate(name="feature_fusion")([mel_features, linear_features])
```

Each branch produces 256 features:

```text
mel_features:    (256)
linear_features: (256)
fusion:          (512)
```

The fused vector contains both perceptual mel information and detailed
linear-frequency information.

### Classifier Head

The classifier is:

```python
Dense(256, activation="relu")
Dropout(0.4)
Dense(128, activation="relu")
Dropout(0.3)
Dense(1, activation="sigmoid")
```

Parameter meanings:

- `Dense(256)`: learns combinations of the fused mel and linear features.
- `Dropout(0.4)`: randomly removes 40 percent of activations during training to
  reduce overfitting.
- `Dense(128)`: compresses the decision representation.
- `Dropout(0.3)`: another regularization layer.
- `Dense(1)`: outputs one logit-like value before activation.
- `sigmoid`: converts the final value into a probability between 0 and 1.

The final output layer is named:

```python
fake_probability
```

## Compilation

Compilation is handled by:

```python
compile_audio_deepfake_cnn(model, learning_rate=1e-3)
```

Optimizer:

```python
Adam(learning_rate=1e-3)
```

Parameter meanings:

- `Adam`: adaptive optimizer that adjusts learning rates per parameter.
- `learning_rate=1e-3`: step size for weight updates. Larger values train
  faster but can be unstable. Smaller values train slower but may be more stable.

Loss:

```python
binary_crossentropy
```

This is appropriate because the task has two classes:

- real: `0.0`
- fake: `1.0`

Metrics:

- `accuracy`: fraction of correct predictions using the default threshold.
- `precision`: of predicted fake clips, how many were actually fake.
- `recall`: of actual fake clips, how many were detected.
- `auc`: area under the ROC curve. This measures ranking quality across
  thresholds and is usually more useful than accuracy for imbalanced data.

## Training Script

Training happens in `main.py`.

### Dataset Layout

The script expects:

```text
data/deepfake mp3 archive/
    real_samples/
    fake_folder_1/
    fake_folder_2/
    ...
```

The folder named `real_samples` is treated as real audio. Every other folder
inside the dataset root is treated as fake audio.

Labels:

```text
fake = 1.0
real = 0.0
```

### Reproducibility

The script uses:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

This makes train-validation splitting and many random operations more
repeatable. Exact results can still vary because TensorFlow operations and CPU
optimizations may be nondeterministic.

### Train-Validation Split

The split uses:

```python
train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels,
)
```

Parameter meanings:

- `test_size=0.2`: 20 percent of samples go to validation.
- `random_state=SEED`: makes the split repeatable.
- `stratify=labels`: keeps real/fake proportions similar in train and
  validation sets.

### Class Weights

The script computes balanced class weights:

```python
compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
```

Why:

- If one class has many more samples than the other, the model may learn to
  prefer the majority class.
- Class weights make mistakes on the minority class count more.

The final dictionary is:

```python
class_weights = {
    0: weight_for_real,
    1: weight_for_fake,
}
```

### Checkpoint Path

The existing script saves to:

```text
audio_deepfake_dual_10s_detector.keras
```

This file stores the best model according to validation AUC.

### Callbacks

The script uses three callbacks.

`ModelCheckpoint`:

```python
monitor="val_auc"
mode="max"
save_best_only=True
```

Meaning:

- Watch validation AUC.
- Higher is better.
- Save only the best model.

`EarlyStopping`:

```python
monitor="val_auc"
mode="max"
patience=5
restore_best_weights=True
```

Meaning:

- Stop training if validation AUC does not improve for 5 epochs.
- Restore weights from the best epoch.

`ReduceLROnPlateau`:

```python
monitor="val_loss"
factor=0.5
patience=2
min_lr=1e-6
```

Meaning:

- If validation loss stops improving for 2 epochs, cut learning rate in half.
- Never reduce below `1e-6`.

### Epochs

The script trains with:

```python
epochs=30
```

An epoch means one full pass through the training set.

Because early stopping is enabled, training may stop before 30 epochs.

## Inference

For file inference, use:

```powershell
.\.venv\Scripts\python.exe predict_audio.py path\to\audio.wav
```

For realtime microphone inference, use:

```powershell
.\.venv\Scripts\python.exe realtime_mic_demo.py
```

The realtime script:

1. Opens the microphone through PyAudio.
2. Continuously records chunks.
3. Keeps a rolling 10-second buffer.
4. Converts the buffer to mel and linear spectrograms.
5. Runs model prediction.
6. Prints fake probability and label.

Important realtime parameters:

- `--model`: path to the `.keras` model file.
- `--threshold`: fake probability cutoff. Default is `0.5`.
- `--interval`: seconds between predictions. Default is `10.0`.
- `--chunk-seconds`: microphone read size. Default is `0.25`.
- `--device`: PyAudio device index.
- `--list-devices`: lists microphone input devices.
- `--visualize`: opens a live spectrogram plot.
- `--save-visual`: saves the latest spectrogram image.

## Strengths

- Simple and easy to train.
- Uses two complementary views of the same audio.
- CNNs are good at local spectro-temporal pattern detection.
- Few custom layers, so saving and loading are straightforward.
- Good baseline for comparing more advanced architectures.

## Limitations

- CNN branches mainly learn local patterns and may miss long-range temporal
  dependencies.
- Global average pooling can discard exact location information.
- Mel and linear branches are fused only after each branch has already compressed
  its features.
- There is no explicit cross-attention between mel and linear representations.
- The model does not explicitly separate spectral sub-band reasoning from
  temporal interval reasoning.

## How To Compare Against BiCrossMamba-ST

For a fair comparison:

- Use the same dataset root.
- Use the same train-validation split seed.
- Use the same preprocessing constants.
- Compare validation AUC, precision, recall, and loss.
- Do not compare models trained on different durations or different sample
  rates.

The baseline model is best understood as:

```text
Audio
-> mel spectrogram + linear spectrogram
-> separate CNN branches
-> concatenate features
-> dense classifier
-> fake probability
```

