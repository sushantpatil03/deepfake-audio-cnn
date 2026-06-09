import os
import glob
import random
import sys
from pathlib import Path

import numpy as np
import librosa
try:
    import tensorflow as tf
except ImportError:
    raise RuntimeError(
        "TensorFlow could not be imported. This project is running on "
        f"Python {sys.version.split()[0]}, but the installed TensorFlow build is "
        "not compatible with it. Create a Python 3.11 or 3.12 virtualenv and "
        "install TensorFlow there; do not use the macOS TensorFlow 1.12 wheel "
        "on Windows."
    ) from None
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from audio_preprocessing import BATCH_SIZE, create_dataset
from buildCNN import build_audio_deepfake_cnn, compile_audio_deepfake_cnn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "deepfake mp3 archive"
REAL_DIR_NAME = "real_samples"
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")

if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Could not find dataset folder: {DATA_ROOT}")

REAL_DIR = DATA_ROOT / REAL_DIR_NAME
if not REAL_DIR.exists():
    raise FileNotFoundError(f"Could not find real samples folder: {REAL_DIR}")

fake_dirs = sorted(
    path for path in DATA_ROOT.iterdir()
    if path.is_dir() and path.name != REAL_DIR_NAME
)

if not fake_dirs:
    raise FileNotFoundError(f"Could not find fake sample folders in: {DATA_ROOT}")


def collect_audio_paths(directory):
    paths = []
    for extension in AUDIO_EXTENSIONS:
        paths.extend(glob.glob(str(directory / extension)))
    return sorted(paths)


fake_paths = []
for fake_dir in fake_dirs:
    dir_paths = collect_audio_paths(fake_dir)
    fake_paths.extend(dir_paths)
    print(f"Fake samples in {fake_dir.name}: {len(dir_paths)}")

real_paths = collect_audio_paths(REAL_DIR)

if not fake_paths:
    raise FileNotFoundError(f"No fake audio files found in: {DATA_ROOT}")

if not real_paths:
    raise FileNotFoundError(f"No real audio files found in: {REAL_DIR}")

print("Using data root:", DATA_ROOT)
print("Using real dir:", REAL_DIR)

print("Fake samples:", len(fake_paths))
print("Real samples:", len(real_paths))

print("Example fake:", fake_paths[:3])
print("Example real:", real_paths[:3])

file_paths = np.array(fake_paths + real_paths)
labels = np.array(
    [1.0] * len(fake_paths) + [0.0] * len(real_paths),
    dtype=np.float32
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

print("Train samples:", len(train_paths))
print("Validation samples:", len(val_paths))

train_ds = create_dataset(train_paths, train_labels, BATCH_SIZE, training=True, seed=SEED)
val_ds = create_dataset(val_paths, val_labels, BATCH_SIZE, training=False)

print("Datasets ready")

model = build_audio_deepfake_cnn()
model = compile_audio_deepfake_cnn(model)
model.summary()

# ============================================================
# 4. Class weights for imbalance
# ============================================================

classes = np.array([0.0, 1.0])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("Class weights:", class_weights)

# ============================================================
# 5. Train model
# ============================================================

MODEL_PATH = PROJECT_ROOT / "audio_deepfake_dual_10s_detector.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks,
)


# ============================================================
# 6. Evaluate model
# ============================================================

results = model.evaluate(val_ds, return_dict=True)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
