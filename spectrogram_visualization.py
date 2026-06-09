from pathlib import Path

from audio_preprocessing import SAMPLE_RATE, HOP_LENGTH


def save_spectrogram_visualization(mel_db, linear_db, output_path):
    import librosa.display
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[0],
    )
    axes[0].set_title("Mel Spectrogram")

    librosa.display.specshow(
        linear_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="linear",
        ax=axes[1],
    )
    axes[1].set_title("Linear-Frequency Spectrogram")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path
