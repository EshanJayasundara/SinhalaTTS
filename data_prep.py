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

from utils import load_config, makedirs, get_logger
logger = get_logger(__name__)

def scan_dataset(dataset_dir: str) -> List[Tuple[str, str]]:
    """Return list of (wav_path, transcript)
    Expects a simple folder structure or a manifest file.
    """
    pairs = []
    # TODO: implement scanning logic (e.g., read a CSV/TSV manifest)
    return pairs

def load_audio(wav_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    wav, orig_sr = sf.read(wav_path)
    if orig_sr != sr:
        wav = librosa.resample(wav.astype(float), orig_sr, sr)
    # normalize
    wav = wav / np.abs(wav).max()
    return wav, sr

def compute_mel(wav: np.ndarray, sr: int = 22050, n_mels: int = 80, hop_length: int = 256, n_fft: int = 1024):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# TODO #