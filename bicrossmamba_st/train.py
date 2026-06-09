import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_preprocessing import BATCH_SIZE, create_dataset
from bicrossmamba_st import build_bicrossmamba_st, compile_bicrossmamba_st


SEED = 42
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")


def collect_audio_paths(directory):
    paths = []
    for extension in AUDIO_EXTENSIONS:
        paths.extend(glob.glob(str(directory / extension)))
    return sorted(paths)


def build_file_list(data_root, real_dir_name):
    real_dir = data_root / real_dir_name
    if not data_root.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {data_root}")
    if not real_dir.exists():
        raise FileNotFoundError(f"Could not find real samples folder: {real_dir}")

    fake_dirs = sorted(
        path for path in data_root.iterdir()
        if path.is_dir() and path.name != real_dir_name
    )
    if not fake_dirs:
        raise FileNotFoundError(f"Could not find fake sample folders in: {data_root}")

    fake_paths = []
    for fake_dir in fake_dirs:
        dir_paths = collect_audio_paths(fake_dir)
        fake_paths.extend(dir_paths)
        print(f"Fake samples in {fake_dir.name}: {len(dir_paths)}")

    real_paths = collect_audio_paths(real_dir)
    if not fake_paths:
        raise FileNotFoundError(f"No fake audio files found in: {data_root}")
    if not real_paths:
        raise FileNotFoundError(f"No real audio files found in: {real_dir}")

    file_paths = np.array(fake_paths + real_paths)
    labels = np.array([1.0] * len(fake_paths) + [0.0] * len(real_paths), dtype=np.float32)
    return file_paths, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the BiCrossMamba-ST flagship detector."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "deepfake mp3 archive",
        help="Dataset folder containing real_samples and fake subfolders.",
    )
    parser.add_argument("--real-dir-name", default="real_samples")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    file_paths, labels = build_file_list(args.data_root, args.real_dir_name)
    print("Using data root:", args.data_root)
    print("Total samples:", len(file_paths))
    print("Fake samples:", int(np.sum(labels == 1.0)))
    print("Real samples:", int(np.sum(labels == 0.0)))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    train_ds = create_dataset(
        train_paths, train_labels, args.batch_size, training=True, seed=SEED
    )
    val_ds = create_dataset(val_paths, val_labels, args.batch_size, training=False)

    model = build_bicrossmamba_st(
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = compile_bicrossmamba_st(model, learning_rate=args.learning_rate)
    model.summary()

    classes = np.array([0.0, 1.0])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )
    class_weights = {0: weights[0], 1: weights[1]}
    print("Class weights:", class_weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "bicrossmamba_st_detector.keras"
    history_path = args.output_dir / "bicrossmamba_st_history.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
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
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    results = model.evaluate(val_ds, return_dict=True)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "history": history.history,
                "validation_results": results,
                "model_path": str(model_path),
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
