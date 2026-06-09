import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import bicrossmamba_st.model  # Registers custom BiCrossMamba-ST Keras layers.
from audio_preprocessing import (
    DURATION,
    SAMPLE_RATE,
    audio_to_model_inputs,
    audio_to_spectrograms,
)
from prediction_explanation import explain_prediction
from spectrogram_visualization import save_spectrogram_visualization

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "bicrossmamba_st" / "checkpoints" / "bicrossmamba_st_detector.keras"
)
DEFAULT_SPECTROGRAM_PATH = (
    PROJECT_ROOT / "visualizations" / "latest_mamba_realtime_spectrogram.png"
)


def load_pyaudio():
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError(
            "PyAudio is not installed. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install pyaudio"
        ) from None

    return pyaudio


def list_input_devices():
    pyaudio = load_pyaudio()
    audio = pyaudio.PyAudio()

    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(
                    f"{index}: {info['name']} "
                    f"({int(info['maxInputChannels'])} input channels)"
                )
    finally:
        audio.terminate()


def create_live_plot():
    import matplotlib.pyplot as plt

    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    return plt, fig, axes


def update_live_plot(plt, axes, mel_db, linear_db, prob):
    axes[0].clear()
    axes[1].clear()
    axes[0].imshow(mel_db, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].imshow(linear_db, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title("Mel Spectrogram")
    axes[1].set_title("Linear-Frequency Spectrogram")
    axes[1].set_xlabel(f"BiCrossMamba-ST fake probability over {DURATION:.1f}s: {prob:.4f}")
    plt.pause(0.001)


def predict_window(model, audio_window, threshold):
    inputs = audio_to_model_inputs(audio_window, sample_rate=SAMPLE_RATE)
    prob = float(model.predict(inputs, verbose=0)[0][0])
    label = "FAKE" if prob >= threshold else "REAL"
    return prob, label


def run_realtime_demo(args):
    pyaudio = load_pyaudio()
    model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained BiCrossMamba-ST model not found: {model_path}. "
            "Train first with: .\\.venv\\Scripts\\python.exe -m bicrossmamba_st.train"
        )

    model = tf.keras.models.load_model(model_path, compile=False)

    chunk_size = int(SAMPLE_RATE * args.chunk_seconds)
    window_size = int(SAMPLE_RATE * DURATION)
    rolling_audio = np.zeros(0, dtype=np.float32)

    live_plot = None
    if args.visualize:
        live_plot = create_live_plot()

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=args.device,
        frames_per_buffer=chunk_size,
    )

    print("BiCrossMamba-ST realtime detector started. Press Ctrl+C to stop.")
    print(f"Using model: {model_path}")
    print(f"Decision window: {DURATION:.1f}s, update interval: {args.interval:.1f}s")

    last_prediction_time = 0.0

    try:
        while True:
            chunk = stream.read(chunk_size, exception_on_overflow=False)
            samples = np.frombuffer(chunk, dtype=np.float32)
            rolling_audio = np.concatenate([rolling_audio, samples])[-window_size:]

            now = time.time()
            if len(rolling_audio) < window_size or now - last_prediction_time < args.interval:
                continue

            last_prediction_time = now
            prob, label = predict_window(model, rolling_audio, args.threshold)
            print(
                f"{time.strftime('%H:%M:%S')} | "
                f"mamba_fake={prob:.4f} | window={DURATION:.1f}s | {label}"
            )
            print("Why:", explain_prediction(prob, args.threshold))

            if args.visualize or args.save_visual:
                mel_db, linear_db = audio_to_spectrograms(rolling_audio)

                if args.visualize:
                    plt, _, axes = live_plot
                    update_live_plot(plt, axes, mel_db, linear_db, prob)

                if args.save_visual:
                    save_spectrogram_visualization(mel_db, linear_db, args.save_visual)
    except KeyboardInterrupt:
        print("\nBiCrossMamba-ST realtime detector stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run realtime audio deepfake detection with BiCrossMamba-ST."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained BiCrossMamba-ST .keras model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fake probability threshold. Default: 0.5",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between predictions. Default: 10.0",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.25,
        help="Microphone read chunk size in seconds. Default: 0.25",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="PyAudio input device index. Use --list-devices to inspect devices.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show a live Mel/linear spectrogram window for the latest audio segment.",
    )
    parser.add_argument(
        "--save-visual",
        default=str(DEFAULT_SPECTROGRAM_PATH),
        help="Path where the latest spectrogram image is overwritten.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        list_input_devices()
        return

    run_realtime_demo(args)


if __name__ == "__main__":
    main()
