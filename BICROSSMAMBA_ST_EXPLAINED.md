# BiCrossMamba-ST Explained

This document explains the new BiCrossMamba-ST-style model added in this project.
It is designed as a stronger experimental model to compare against the existing
mel plus linear spectrogram CNN.

The important files are:

- `bicrossmamba_st/model.py`: defines the BiCrossMamba-ST-style architecture.
- `bicrossmamba_st/train.py`: trains the model using the same project dataset
  convention as the older baseline.
- `realtime_mamba_mic_demo.py`: runs realtime microphone inference using the
  trained BiCrossMamba-ST checkpoint.
- `audio_preprocessing.py`: still provides the shared mel and linear
  spectrogram inputs.

## Important Practical Note

The paper-style name says Mamba, but this project does not install an official
Mamba CUDA kernel or a third-party Mamba package. Instead, the model implements a
compact selective state-space sequence mixer in pure TensorFlow/Keras. This keeps
the architecture runnable with the current `requirements.txt`.

So this model is best described as:

```text
BiCrossMamba-ST-style TensorFlow implementation
```

It follows the idea of:

- Bidirectional sequence mixing.
- Spectral and temporal branches.
- Mutual cross-attention.
- 2D spectro-temporal attention.

It is not an official reproduction of a specific paper implementation.

## High-Level Idea

The older CNN compresses each spectrogram branch with convolution and only then
fuses the features. BiCrossMamba-ST does something more structured:

1. It starts from the same mel and linear spectrogram inputs.
2. It resizes them to a shared spectro-temporal grid.
3. It combines them as channels in one feature map.
4. It applies a convolutional 2D attention map to emphasize important
   time-frequency regions.
5. It creates a spectral sequence by grouping frequency sub-bands.
6. It creates a temporal sequence by grouping time intervals.
7. It processes both sequences with bidirectional Mamba-like state-space blocks.
8. It lets the spectral and temporal branches exchange information through
   mutual cross-attention.
9. It pools both branches and classifies the audio as real or fake.

The model output is:

```text
fake_probability in range [0, 1]
```

Interpretation:

- Close to `0.0`: model thinks the audio is real.
- Close to `1.0`: model thinks the audio is fake.
- The default classification threshold is `0.5`.

## Why This Architecture Exists

Speech deepfakes can contain subtle artifacts that may appear:

- In narrow frequency bands.
- In short time regions.
- Across longer temporal patterns.
- As unnatural interactions between frequency and time.

A regular CNN can detect local image-like patterns, but it may not be strong at
modeling longer sequential structure. BiCrossMamba-ST adds explicit sequence
reasoning over both spectral and temporal views.

## Shared Input Format

The model uses the same input dictionary as the older CNN:

```python
{
    "mel_input": mel_tensor,
    "linear_input": linear_tensor,
}
```

Default input shapes:

```text
mel_input:    (64, 320, 1)
linear_input: (513, 320, 1)
```

This is deliberate. It makes the comparison with the older model more fair
because both models see the same preprocessed features.

## Main Builder Function

The architecture is built by:

```python
build_bicrossmamba_st(
    mel_input_shape=(64, 320, 1),
    linear_input_shape=(513, 320, 1),
    d_model=96,
    depth=3,
    num_heads=4,
    resized_freq_bins=128,
    resized_time_steps=320,
    pooled_freq_bins=64,
    pooled_time_steps=80,
    spectral_bands=8,
    temporal_intervals=8,
    dropout=0.15,
)
```

## Builder Parameters

### `mel_input_shape`

Default:

```python
(64, 320, 1)
```

Meaning:

- 64 mel bands.
- 320 time frames.
- 1 channel.

This must match `audio_preprocessing.py`.

### `linear_input_shape`

Default:

```python
(513, 320, 1)
```

Meaning:

- 513 STFT frequency bins.
- 320 time frames.
- 1 channel.

This also must match `audio_preprocessing.py`.

### `d_model`

Default:

```python
96
```

This is the hidden feature dimension used in the sequence branches.

Effects:

- Higher `d_model` increases model capacity.
- Higher `d_model` increases memory usage and training time.
- Lower `d_model` trains faster but may underfit.

Example:

- `d_model=48`: lighter smoke-test model.
- `d_model=96`: default balanced model.
- `d_model=128` or higher: larger experiment if hardware allows.

### `depth`

Default:

```python
3
```

This controls how many repeated blocks are used.

Each depth level contains:

- One spectral bidirectional Mamba-like block.
- One temporal bidirectional Mamba-like block.
- One mutual cross-attention block.

Effects:

- Higher `depth` means deeper reasoning.
- Higher `depth` increases training cost.
- Too high can overfit on small datasets.

### `num_heads`

Default:

```python
4
```

This controls the number of attention heads in mutual cross-attention.

Meaning:

- Multi-head attention splits representation space into several subspaces.
- Each head can learn a different relationship between spectral and temporal
  tokens.

Constraint:

- `d_model // num_heads` should be at least 1.
- Ideally, `d_model` should divide cleanly by `num_heads`.

With defaults:

```text
key_dim = 96 // 4 = 24
```

### `resized_freq_bins`

Default:

```python
128
```

The mel and linear spectrograms have different frequency dimensions:

```text
mel:    64
linear: 513
```

Before combining them, both are resized to a shared frequency grid:

```text
128 frequency bins
```

Why:

- Concatenation requires matching height and width.
- A shared grid lets the model combine mel and linear views channel-wise.

### `resized_time_steps`

Default:

```python
320
```

This is the shared time dimension after resizing.

It stays equal to the original `MAX_TIME_STEPS` from preprocessing, so time
resolution is preserved before pooling.

### `pooled_freq_bins`

Default:

```python
64
```

After the convolutional stem, the feature map is pooled by `(2, 4)`.

Default resize before pooling:

```text
(128, 320)
```

After pooling:

```text
(64, 80)
```

So `pooled_freq_bins` must match the feature map frequency size after pooling.

### `pooled_time_steps`

Default:

```python
80
```

This is the time size after pooling:

```text
320 / 4 = 80
```

This parameter is used by the tokenizers to reshape the feature map correctly.

### `spectral_bands`

Default:

```python
8
```

The spectral tokenizer divides the frequency axis into 8 sub-bands.

With default pooled frequency bins:

```text
64 pooled frequency bins / 8 bands = 8 bins per band
```

Each token summarizes one frequency sub-band at one time step.

Constraint:

```text
pooled_freq_bins must be divisible by spectral_bands
```

### `temporal_intervals`

Default:

```python
8
```

The temporal tokenizer divides the time axis into 8 intervals.

With default pooled time steps:

```text
80 pooled time steps / 8 intervals = 10 frames per interval
```

Each token summarizes one time interval at one frequency bin.

Constraint:

```text
pooled_time_steps must be divisible by temporal_intervals
```

### `dropout`

Default:

```python
0.15
```

Dropout randomly disables activations during training.

Purpose:

- Reduces overfitting.
- Forces the model not to depend too heavily on one feature path.

Effects:

- Higher dropout gives stronger regularization but may underfit.
- Lower dropout allows more capacity but may overfit.

## Architecture Flow

Default input:

```text
mel_input:    (batch, 64, 320, 1)
linear_input: (batch, 513, 320, 1)
```

Step 1: resize both inputs:

```text
mel resized:    (batch, 128, 320, 1)
linear resized: (batch, 128, 320, 1)
```

Step 2: concatenate as channels:

```text
raw_feature_pair: (batch, 128, 320, 2)
```

Step 3: convolutional stem:

```text
(batch, 128, 320, 2)
-> Conv2D(d_model // 2)
-> BatchNormalization
-> Conv2D(d_model)
-> BatchNormalization
-> SpectroTemporalAttention2D
-> AveragePooling2D(pool_size=(2, 4))
```

Default output:

```text
feature_map: (batch, 64, 80, 96)
```

Step 4: spectral tokenizer:

```text
(batch, 64, 80, 96)
-> 8 frequency sub-bands
-> sequence length 8 * 80 = 640
-> spectral tokens: (batch, 640, 96)
```

Step 5: temporal tokenizer:

```text
(batch, 64, 80, 96)
-> 8 time intervals
-> sequence length 8 * 64 = 512
-> temporal tokens: (batch, 512, 96)
```

Step 6: repeated BiMamba and cross-attention blocks:

```text
for each block:
    spectral tokens -> BiMambaBlock
    temporal tokens -> BiMambaBlock
    spectral and temporal tokens -> MutualCrossAttentionBlock
```

Step 7: pooling:

```text
spectral avg pool + spectral max pool
temporal avg pool + temporal max pool
feature map global average pool
```

Step 8: classifier:

```text
Concatenate
LayerNormalization
Dense(384, gelu)
Dropout
Dense(128, gelu)
Dropout
Dense(1, sigmoid)
```

## SpectroTemporalAttention2D

Class:

```python
SpectroTemporalAttention2D(hidden_filters=32, dropout=0.1)
```

Purpose:

This layer learns a 2D attention map over the time-frequency feature map.

Input:

```text
(batch, frequency, time, channels)
```

Internal flow:

```text
BatchNormalization
Conv2D(hidden_filters, kernel=(5, 1), activation="swish")
Conv2D(hidden_filters, kernel=(1, 7), activation="swish")
Dropout
Conv2D(1, kernel=(3, 3), activation="sigmoid")
```

Parameter meanings:

- `hidden_filters`: number of hidden filters used to compute the attention map.
- `dropout`: regularization used while computing attention.
- `(5, 1)` convolution: focuses mostly along the frequency axis.
- `(1, 7)` convolution: focuses mostly along the time axis.
- final `sigmoid`: produces attention values between 0 and 1.

Output formula:

```python
inputs * (1.0 + attention)
```

Why `1.0 + attention`:

- If attention is near 0, the original feature remains.
- If attention is near 1, the feature is amplified up to about 2x.
- This avoids completely deleting regions early in training.

## SpectralSubBandTokenizer

Class:

```python
SpectralSubBandTokenizer(
    freq_bins=64,
    time_steps=80,
    bands=8,
)
```

Purpose:

Create a sequence where each token represents one frequency sub-band at one time
step.

Input:

```text
(batch, 64, 80, channels)
```

Process:

1. Split 64 frequency bins into 8 bands.
2. Each band contains 8 frequency bins.
3. Average within each band.
4. Keep all 80 time positions.
5. Flatten band and time into a sequence.

Output:

```text
(batch, 8 * 80, channels)
(batch, 640, channels)
```

Parameter meanings:

- `freq_bins`: frequency size of the input feature map.
- `time_steps`: time size of the input feature map.
- `bands`: number of spectral sub-bands.

Constraint:

```text
freq_bins % bands == 0
```

What this branch learns:

- Frequency-specific fake artifacts.
- Sub-band distortions.
- Patterns that appear differently in low, middle, and high frequencies.

## TemporalIntervalTokenizer

Class:

```python
TemporalIntervalTokenizer(
    freq_bins=64,
    time_steps=80,
    intervals=8,
)
```

Purpose:

Create a sequence where each token represents one time interval at one frequency
bin.

Input:

```text
(batch, 64, 80, channels)
```

Process:

1. Split 80 time frames into 8 intervals.
2. Each interval contains 10 time frames.
3. Average within each interval.
4. Keep all 64 frequency positions.
5. Flatten interval and frequency into a sequence.

Output:

```text
(batch, 8 * 64, channels)
(batch, 512, channels)
```

Parameter meanings:

- `freq_bins`: frequency size of the input feature map.
- `time_steps`: time size of the input feature map.
- `intervals`: number of temporal intervals.

Constraint:

```text
time_steps % intervals == 0
```

What this branch learns:

- Time-local synthesis artifacts.
- Long-range temporal inconsistency.
- Repeated or unnatural speech-generation patterns.

## SelectiveSSMMixer

Class:

```python
SelectiveSSMMixer(
    d_model,
    expansion=2,
    kernel_size=5,
    dropout=0.1,
)
```

Purpose:

This is the Mamba-like part of the model. It scans across a token sequence and
updates a hidden state at every step. The update is selective because the model
learns how much of the previous state to keep for each token and channel.

Input:

```text
(batch, sequence_length, d_model)
```

### Parameters

`d_model`:

- Size of each token vector.
- Must match the branch hidden dimension.

`expansion`:

- Multiplies the internal hidden dimension.
- Default `2` means internal dimension is `2 * d_model`.
- Higher expansion increases capacity and cost.

`kernel_size`:

- Size of the depthwise 1D convolution over the token sequence.
- Default `5` means each token sees nearby context before the recurrent scan.

`dropout`:

- Regularization applied to the output.

### Internal Layers

`in_proj`:

```python
Dense(inner_dim)
```

Projects each token from `d_model` to `inner_dim`.

`conv`:

```python
Conv1D(inner_dim, kernel_size, groups=inner_dim, activation="swish")
```

This is a depthwise sequence convolution.

Meaning:

- Each channel is convolved separately.
- It adds local sequence context.
- `swish` gives smooth nonlinear activation.

`alpha_proj`:

```python
Dense(inner_dim, bias_initializer="ones")
```

Learns a keep gate called `alpha`.

After sigmoid:

```text
alpha in range [0, 1]
```

Interpretation:

- Alpha near 1: keep more previous state.
- Alpha near 0: replace more with current candidate.

`candidate_proj`:

```python
Dense(inner_dim, activation="tanh")
```

Creates the candidate state for the current token.

`gate_proj`:

```python
Dense(inner_dim, activation="sigmoid")
```

Creates an output gate that controls how much of the scanned state is exposed.

`out_proj`:

```python
Dense(d_model)
```

Projects internal features back to the original token dimension.

### State Update

The recurrent scan uses:

```python
state_t = alpha_t * state_previous + (1.0 - alpha_t) * candidate_t
```

This is similar in spirit to a gated state-space update.

Why it matters:

- It can carry information across long sequences.
- It is more sequence-aware than plain convolution.
- The gate decides what to remember and what to update.

## BiMambaBlock

Class:

```python
BiMambaBlock(
    d_model,
    expansion=2,
    kernel_size=5,
    dropout=0.1,
)
```

Purpose:

Process a sequence in both directions.

Why bidirectional:

- A forward scan sees past tokens.
- A backward scan sees future tokens.
- Combining both gives each token context from both sides.

Internal flow:

```text
LayerNormalization
forward SelectiveSSMMixer
reverse sequence
backward SelectiveSSMMixer
reverse back
concatenate forward and backward
Dense(d_model)
residual connection
LayerNormalization
feed-forward network
residual connection
```

Parameter meanings:

- `d_model`: token feature size.
- `expansion`: internal SSM expansion factor.
- `kernel_size`: local sequence convolution size.
- `dropout`: regularization.

Residual connections:

The block adds learned updates back to the original input. This helps gradients
flow through deep networks and reduces training instability.

Feed-forward network:

```text
Dense(d_model * 4, gelu)
Dropout
Dense(d_model)
Dropout
```

This lets each token transform its own features after sequence mixing.

## MutualCrossAttentionBlock

Class:

```python
MutualCrossAttentionBlock(
    d_model,
    num_heads=4,
    dropout=0.1,
)
```

Purpose:

Let spectral tokens and temporal tokens exchange information.

It does two cross-attention operations:

```text
spectral queries attend to temporal keys/values
temporal queries attend to spectral keys/values
```

Why this matters:

- Spectral artifacts may only make sense when linked to time behavior.
- Temporal artifacts may only make sense when linked to frequency regions.
- Mutual attention allows both branches to refine each other.

Parameter meanings:

- `d_model`: feature size of both token branches.
- `num_heads`: number of attention heads.
- `dropout`: attention and feed-forward regularization.

Key dimension:

```python
key_dim = max(1, d_model // num_heads)
```

With defaults:

```text
d_model = 96
num_heads = 4
key_dim = 24
```

Each branch also has its own feed-forward network after attention.

## Pooling and Classifier

After all sequence blocks, each branch is pooled.

Spectral pooling:

```text
GlobalAveragePooling1D
GlobalMaxPooling1D
Concatenate
```

Temporal pooling:

```text
GlobalAveragePooling1D
GlobalMaxPooling1D
Concatenate
```

Why both average and max:

- Average pooling captures overall evidence.
- Max pooling captures strongest local evidence.
- Combining them gives a richer summary.

The original 2D feature map is also pooled:

```python
GlobalAveragePooling2D(name="attention_map_pool")
```

The final classifier receives:

```text
spectral pooled features
temporal pooled features
2D attention map pooled features
```

Classifier:

```text
LayerNormalization
Dense(384, gelu)
Dropout
Dense(128, gelu)
Dropout
Dense(1, sigmoid)
```

Parameter meanings:

- `LayerNormalization`: normalizes the fused vector.
- `Dense(384)`: large fusion layer.
- `gelu`: smooth nonlinear activation often used in transformer-style models.
- `Dropout`: regularization.
- `Dense(128)`: smaller decision layer.
- `Dense(1)`: one output value.
- `sigmoid`: converts output to fake probability.

## Compilation

The model is compiled by:

```python
compile_bicrossmamba_st(model, learning_rate=3e-4)
```

Optimizer:

```python
AdamW(
    learning_rate=learning_rate,
    weight_decay=1e-4,
    clipnorm=1.0,
)
```

Parameter meanings:

- `AdamW`: Adam optimizer with decoupled weight decay.
- `learning_rate`: update step size.
- `weight_decay=1e-4`: penalizes large weights and helps reduce overfitting.
- `clipnorm=1.0`: clips gradients if their norm gets too large, improving
  stability.

Default learning rate:

```text
3e-4
```

This is lower than the older CNN's `1e-3` because this model is deeper and uses
attention plus recurrent scans, so a smaller learning rate is safer.

Loss:

```text
binary_crossentropy
```

Metrics:

- `accuracy`: threshold-based correctness.
- `precision`: fake predictions that are actually fake.
- `recall`: actual fake examples detected.
- `auc`: threshold-independent ranking metric.

## Training Script

Training is run with:

```powershell
.\.venv\Scripts\python.exe -m bicrossmamba_st.train
```

The script is in:

```text
bicrossmamba_st/train.py
```

### Training Arguments

`--data-root`:

- Dataset folder.
- Default: `data/deepfake mp3 archive`.
- Must contain `real_samples` and one or more fake folders.

`--real-dir-name`:

- Name of the real audio folder.
- Default: `real_samples`.

`--epochs`:

- Maximum number of training epochs.
- Default: `40`.
- Early stopping may stop earlier.

`--batch-size`:

- Number of clips per batch.
- Default comes from `audio_preprocessing.BATCH_SIZE`, currently `32`.
- Lower this if memory is limited.

`--learning-rate`:

- Optimizer step size.
- Default: `3e-4`.
- Lower it if training is unstable.
- Raise carefully if training is too slow and stable.

`--d-model`:

- Hidden feature dimension.
- Default: `96`.
- Lower for faster runs.
- Higher for larger experiments.

`--depth`:

- Number of repeated BiMamba plus cross-attention stages.
- Default: `3`.

`--num-heads`:

- Attention heads in mutual cross-attention.
- Default: `4`.

`--dropout`:

- Dropout rate throughout the model.
- Default: `0.15`.

`--output-dir`:

- Folder where checkpoints and history are written.
- Default: `bicrossmamba_st/checkpoints`.

### Smoke Test Training

Use this to verify the pipeline quickly:

```powershell
.\.venv\Scripts\python.exe -m bicrossmamba_st.train --epochs 1 --d-model 48 --depth 1 --batch-size 4
```

This is not meant for final performance. It is only to check that training runs.

### Output Files

Default checkpoint:

```text
bicrossmamba_st/checkpoints/bicrossmamba_st_detector.keras
```

Default history:

```text
bicrossmamba_st/checkpoints/bicrossmamba_st_history.json
```

The history file stores:

- Training metric history.
- Validation results.
- Saved model path.

## Training Flow

The training script does this:

1. Sets random seeds.
2. Collects real and fake audio file paths.
3. Labels fake as `1.0` and real as `0.0`.
4. Creates an 80/20 stratified train-validation split.
5. Builds TensorFlow datasets using shared preprocessing.
6. Builds and compiles the BiCrossMamba-ST model.
7. Computes balanced class weights.
8. Trains with callbacks.
9. Evaluates on validation data.
10. Saves history and validation results.

## Callbacks

`ModelCheckpoint`:

```text
monitor="val_auc"
mode="max"
save_best_only=True
```

Saves the model with the best validation AUC.

`EarlyStopping`:

```text
monitor="val_auc"
mode="max"
patience=7
restore_best_weights=True
```

Stops if validation AUC does not improve for 7 epochs.

`ReduceLROnPlateau`:

```text
monitor="val_loss"
factor=0.5
patience=2
min_lr=1e-6
```

Cuts learning rate in half when validation loss plateaus.

## Realtime Microphone Inference

Run:

```powershell
.\.venv\Scripts\python.exe realtime_mamba_mic_demo.py
```

Default model path:

```text
bicrossmamba_st/checkpoints/bicrossmamba_st_detector.keras
```

The script imports:

```python
import bicrossmamba_st.model
```

This is important because the `.keras` file contains custom Keras layers. The
import registers those layers before loading the saved model.

Realtime arguments:

- `--model`: path to a trained BiCrossMamba-ST `.keras` file.
- `--threshold`: fake probability threshold. Default is `0.5`.
- `--interval`: seconds between predictions. Default is `10.0`.
- `--chunk-seconds`: microphone read chunk size. Default is `0.25`.
- `--device`: PyAudio input device index.
- `--list-devices`: list microphone devices and exit.
- `--visualize`: show live mel and linear spectrogram plots.
- `--save-visual`: save the latest spectrogram image.

The realtime flow is:

```text
microphone audio
-> rolling 10-second buffer
-> shared mel and linear preprocessing
-> BiCrossMamba-ST prediction
-> fake probability and label
```

## Comparison With Older CNN

Older model:

```text
mel spectrogram -> CNN branch
linear spectrogram -> CNN branch
concatenate
dense classifier
```

BiCrossMamba-ST:

```text
mel + linear shared grid
2D attention
spectral sub-band sequence branch
temporal interval sequence branch
bidirectional Mamba-like sequence mixing
mutual cross-attention
fusion classifier
```

Main differences:

- The older CNN fuses after branch compression.
- BiCrossMamba-ST fuses early into a shared feature map.
- The older CNN uses only convolutional local pattern extraction.
- BiCrossMamba-ST adds sequence modeling.
- The older CNN does not explicitly model spectral and temporal branches.
- BiCrossMamba-ST explicitly separates and then reconnects spectral and temporal
  reasoning.
- The older CNN is simpler and faster.
- BiCrossMamba-ST is heavier and may need more data to show its advantage.

## Tuning Guidance

If training is too slow:

- Reduce `--d-model` to `48` or `64`.
- Reduce `--depth` to `1` or `2`.
- Reduce `--batch-size` if memory is the issue.

If training overfits:

- Increase `--dropout`.
- Use stronger data balance.
- Add more diverse training data.
- Keep `depth` moderate.

If validation AUC is unstable:

- Lower `--learning-rate`.
- Increase batch size if memory allows.
- Keep `clipnorm=1.0`.

If model underfits:

- Increase `--d-model`.
- Increase `--depth`.
- Train for more epochs.
- Lower dropout slightly.

## Fair Evaluation Checklist

Use this checklist when comparing with the older model:

- Same dataset.
- Same real and fake folders.
- Same preprocessing constants.
- Same train-validation seed.
- Same validation split strategy.
- Same metrics.
- Same threshold only for threshold-based metrics.
- Prefer validation AUC for model comparison.

The most important comparison metric in this project is usually:

```text
val_auc
```

Accuracy can be misleading if the dataset is imbalanced.

## Mental Model

Think of BiCrossMamba-ST like this:

```text
CNN front-end:
    finds local spectro-temporal patterns

2D attention:
    emphasizes suspicious regions

spectral branch:
    asks what is happening across frequency sub-bands

temporal branch:
    asks what is happening across time intervals

bidirectional Mamba-like blocks:
    pass information forward and backward through each sequence

mutual cross-attention:
    lets frequency evidence and time evidence explain each other

classifier:
    combines everything into one fake probability
```

