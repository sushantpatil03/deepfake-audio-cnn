import argparse
from pathlib import Path

import tensorflow as tf

from audio_preprocessing import audio_to_spectrograms, file_to_model_inputs, load_fixed_audio
from prediction_explanation import explain_prediction
from spectrogram_visualization import save_spectrogram_visualization

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "audio_deepfake_dual_10s_detector.keras"


def predict_audio(model, file_path, threshold=0.5, visualize_dir=None):
    audio_path = Path(file_path).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    prob = float(model.predict(file_to_model_inputs(audio_path), verbose=0)[0][0])
    label = "FAKE" if prob >= threshold else "REAL"

    print("File:", audio_path)
    print(f"Fake probability: {prob:.4f}")
    print("Prediction:", label)
    print("Why:", explain_prediction(prob, threshold))

    if visualize_dir:
        mel_db, linear_db = audio_to_spectrograms(load_fixed_audio(audio_path))
        output_path = Path(visualize_dir) / f"{audio_path.stem}_spectrograms.png"
        save_spectrogram_visualization(mel_db, linear_db, output_path)
        print("Spectrogram image:", output_path.resolve())

    return prob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict whether audio files are real or fake using a trained model."
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="One or more audio file paths to classify.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained .keras model file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fake probability threshold. Default: 0.5",
    )
    parser.add_argument(
        "--visualize-dir",
        help="Optional directory where Mel and linear spectrogram images are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run main.py once to train and save it."
        )

    model = tf.keras.models.load_model(model_path, compile=False)

    for index, audio_file in enumerate(args.audio_files):
        if index:
            print()
        predict_audio(
            model,
            audio_file,
            threshold=args.threshold,
            visualize_dir=args.visualize_dir,
        )


if __name__ == "__main__":
    main()
