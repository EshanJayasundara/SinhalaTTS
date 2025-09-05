"""Text and audio preprocessing utilities.
Functions:
- scan_dataset: discover wav + transcript pairs
- load_audio: load, resample, normalize
- compute_mel: compute mel-spectrogram and save
- create_manifest: write metadata CSV/JSON for training
"""
import os
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import numpy as np
import librosa
from typing import Iterator

from utils import load_config, makedirs, get_logger
logger = get_logger(__name__)

from utils import read_file

def scan_dataset(dataset_dir: str) -> List[Tuple[str, str]]:
    """Return list of (wav_path, transcript)
    Expects a simple folder structure or a manifest file.
    """
    pairs = []
    for line in read_file(f"{dataset_dir}/metadata.txt"):
        pairs.append(tuple(line.split("|")))
    return pairs

def load_audio(wav_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    wav, orig_sr = sf.read(wav_path)
    if orig_sr != sr:
        wav = librosa.resample(y=wav.astype(float), orig_sr=orig_sr, target_sr=sr)
    # normalize
    wav = wav / np.abs(wav).max()
    return wav, sr

def compute_mel(wav: np.ndarray, sr: int = 22050, n_mels: int = 80, hop_length: int = 256, n_fft: int = 1024):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel, mel_db

def reconstruct_wav(mel: np.ndarray, sr: int = 22050) -> None:
    """Griffin-Lim algorithm not precise
    """
    # First convert mel -> linear-frequency spectrogram
    inv_S = librosa.feature.inverse.mel_to_stft(mel, sr=sr)
    # Then reconstruct waveform using Griffin-Lim algorithm
    y_inv = librosa.griffinlim(inv_S)
    sf.write("reconstructed.wav", y_inv, sr)

# TODO #