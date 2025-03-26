import argparse
from dataclasses import dataclass
import os
import random
from typing import List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import mido
import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    y: np.ndarray
    sr: int
    onset_samples: np.ndarray = None


def load_files(input_path: str, recurse: bool) -> list[AudioData]:
    # Check if input is a single file or a directory
    if os.path.isfile(input_path):
        # Load the audio file
        print(f"Loading {input_path}")
        y, sr = librosa.load(input_path)
        return [AudioData(y, sr)]
    elif os.path.isdir(input_path):
        # Load all audio files in the directory
        if recurse:
            audio_files = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.endswith(".wav"):
                        print(f"Loading {os.path.join(root, file)}")
                        y, sr = librosa.load(os.path.join(root, file))
                        audio_files.append(AudioData(y, sr))
            return audio_files
        else:
            audio_files = []
            for file in os.listdir(input_path):
                if file.endswith(".wav"):
                    print(f"Loading {os.path.join(input_path, file)}")
                    y, sr = librosa.load(os.path.join(input_path, file))
                    audio_files.appennd(AudioData(y, sr))
            return audio_files
    else:
        raise ValueError("Invalid input path.")


def write_output(
    output_path: str,
    data: List[AudioData],
    length: int = 512,
    gap: bool = False,
    sort: bool = False,
    randomize: bool = False,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    transients = {"all": [], "low": [], "mid": [], "high": []}

    for audio_data in data:
        for t, onset in enumerate(audio_data.onset_samples):
            end = onset + length
            if end > len(audio_data.y):
                end = len(audio_data.y)

            transient = audio_data.y[onset:end]

            # if transient is empty, skip
            if len(transient) == 0:
                continue

            transient = apply_fade_in(transient)
            transient = apply_fade_out(transient)
            transient = normalize_audio(transient)

            centroid = compute_spectral_centroid(transient, audio_data.sr)

            if gap:
                transient = np.concatenate((transient, np.zeros_like(transient)))

            if sort:
                if centroid < 1000:
                    transients["low"].append(transient)
                elif centroid < 4000:
                    transients["mid"].append(transient)
                else:
                    transients["high"].append(transient)
            else:
                transients["all"].append(transient)

    if randomize:
        random.shuffle(transients["all"])
        random.shuffle(transients["low"])
        random.shuffle(transients["mid"])
        random.shuffle(transients["high"])

    # Sort transients by spectral centroid (low → high frequency)
    # all_transients.sort(key=lambda x: x[0])
    # Filter out tuple from all_transients
    # all_transients = [x[1] for x in all_transients]

    # Batch into files of 128 transients

    for key, value in transients.items():
        if value:
            num_files = len(value) // 128
            num_files += 1 if len(value) % 128 != 0 else 0
            for i in range(num_files):
                transients_to_save = np.concatenate(value[i * 128 : (i + 1) * 128])
                output_file = os.path.join(output_path, f"{key}_transients_{i}.wav")
                sf.write(output_file, transients_to_save, audio_data.sr)
                print(f"Wrote Wav {output_file}")

def extract_transients(data: AudioData, plot: bool = False) -> Optional[np.ndarray]:
    onset_env = librosa.onset.onset_strength(y=data.y, sr=data.sr)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=data.sr,
        # backtrack=True,
        normalize=True,
    )

    if onset_frames.size == 0:
        return None

    onset_frames = librosa.onset.onset_backtrack(onset_frames, onset_env)

    onset_times = librosa.frames_to_time(onset_frames, sr=data.sr)
    data.onset_samples = librosa.frames_to_samples(onset_frames)

    if plot:
        # Plot waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(data.y, sr=data.sr, alpha=0.6)

        # Plot onset markers
        plt.vlines(
            onset_times,
            ymin=-1,
            ymax=1,
            color="r",
            linestyle="dashed",
            label="Detected Transients",
        )

        # Labels and display settings
        plt.title("Waveform with Transient Markers")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    print(f"Detected {len(onset_times)} transients: {onset_times}")

    if len(onset_times) == 0:
        return None

    return data

def apply_fade_out(y, fade_length=8):
    """Apply a short linear fade-out."""
    fade_length = min(fade_length, len(y))  # Ensure fade is within bounds
    fade_curve = np.linspace(1, 0, fade_length)
    y[-fade_length:] *= fade_curve  # Apply fade to end portion
    return y


def apply_fade_out(y, fade_length=4):
    """Apply a short linear fade-out to prevent clicks at the end."""
    fade_length = min(fade_length, len(y))  # Ensure fade is within bounds
    fade_curve = np.linspace(1, 0, fade_length)
    y[-fade_length:] *= fade_curve  # Apply fade-out to end portion
    return y


def normalize_audio(y):
    """Normalize the audio to peak at 1.0."""
    peak = np.max(np.abs(y))  # Find the max absolute value
    if peak > 0:  # Avoid division by zero
        y = y / peak  # Scale the waveform
    return y


def apply_fade_in(y, fade_length=4):
    """Apply a very fast fade-in to ensure the transient starts at zero."""
    fade_length = min(fade_length, len(y))  # Ensure fade is within bounds
    fade_curve = np.linspace(0, 1, fade_length)
    y[:fade_length] *= fade_curve  # Apply fade-in to start portion
    return y


def compute_spectral_centroid(y, sr):
    """Compute spectral centroid as a rough estimate of frequency content."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)  # Return the average centroid over time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract transient features from given audio file(s)."
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Path to the input audio file(s)."
    )
    parser.add_argument(
        "-r",
        "--recurse",
        action="store_true",
        help="Recursively load audio files in the input directory.",
    )
    parser.add_argument("-o", "--output", type=str, help="Path to the output folder.")

    parser.add_argument(
        "-l", "--length", type=int, help="Length of the transient in samples. Default 512."
    )
    parser.add_argument(
        "--gap",
        action="store_true",
        help="Appends a gap between transients the same size as the transient.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort the transients by spectral centroid into kick/snare/hat.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the order of the transients.",
    )
    parser.add_argument(
        "--syx",
        action="store_true",
        help="Output transients as SDS-compatible SysEx files.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the detected transients on top of the waveform.",
    )

    args = parser.parse_args()

    input_files = load_files(args.input, args.recurse)

    output_files = []

    for audio_data in input_files:
        transients = extract_transients(audio_data, plot=args.plot)
        if transients is not None:
            output_files.append(transients)

    write_output(
        args.output, 
        output_files, 
        length=args.length, 
        gap=args.gap, 
        sort=args.sort, 
        randomize=args.randomize,
    )

    if args.syx:
        # Convert all wav files in the output folder to SDS files with SoX
        # sox input.wav -t sds -> output.syx

        # Get all .wav files in output folder
        wav_files = [f for f in os.listdir(args.output) if f.endswith(".wav")]

        for wav_file in wav_files:
            output_file = os.path.join(args.output, wav_file)
            output_file_syx = output_file.replace(".wav", ".syx")
            os.system(f"sox {output_file} -c 1 -t sds {output_file_syx}")
            print(f"Converted {output_file} to {output_file_syx}")
