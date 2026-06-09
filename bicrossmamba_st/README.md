# BiCrossMamba-ST Flagship Detector

This folder is isolated from the existing CNN baseline. It adds a TensorFlow/Keras
implementation of a BiCrossMamba-ST-style detector that consumes the same
`mel_input` and `linear_input` tensors as the current mel + linear spectrogram
model, so validation metrics are directly comparable.

## Architecture

- Shared raw feature grid built from mel and linear spectrogram inputs.
- Convolutional 2D spectro-temporal attention map to emphasize localized cues.
- Spectral branch tokenized by frequency sub-bands.
- Temporal branch tokenized by time intervals.
- Bidirectional selective state-space mixers inspired by Mamba blocks.
- Mutual cross-attention between spectral and temporal branches.
- Fused classifier head for binary real/fake detection.

The implementation is dependency-free beyond the current TensorFlow stack. It is
not a drop-in official Mamba kernel; it uses a compact selective SSM scan so the
model can train in this project without adding custom CUDA or third-party Mamba
packages.

## Train

From the project root:

```powershell
python -m bicrossmamba_st.train
```

Useful smaller smoke-test run:

```powershell
python -m bicrossmamba_st.train --epochs 1 --d-model 48 --depth 1 --batch-size 4
```

Outputs are written to:

```text
bicrossmamba_st/checkpoints/bicrossmamba_st_detector.keras
bicrossmamba_st/checkpoints/bicrossmamba_st_history.json
```

## Comparison Notes

For a fair head-to-head comparison with the existing mel + linear spectrogram
CNN, keep the same dataset split policy, preprocessing constants, sample rate,
duration, and validation metric. This script mirrors the existing `main.py`
split, class weighting, callbacks, and AUC-based checkpointing.
