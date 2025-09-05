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
import argparse
import json

import soundfile as sf
import numpy as np
import librosa
from typing import Iterator

from utils import load_config, makedirs, get_logger
from g2p import sentence_to_phonemes
from tokenizer import PhonemeTokenizer

logger = get_logger(__name__)

from utils import read_file

def scan_dataset(dataset_dir: str) -> List[Tuple[str, str]]:
    """Return list of (wav_path, transcript)
    Expects a simple folder structure or a manifest file.
    """
    pairs = []
    for line in read_file(f"{dataset_dir}"):
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

def compute_mel(wav: np.ndarray, sr: int = 22050, n_mels: int = 80, 
                hop_length: int = 256, n_fft: int = 1024) -> np.ndarray:
    """Compute mel-spectrogram from audio waveform."""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, 
        hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=1.0)
    return mel_db.T  # Transpose to [time, n_mels]

def extract_pitch(wav: np.ndarray, sr: int = 22050, hop_length: int = 256) -> np.ndarray:
    """Extract pitch using librosa's piptrack."""
    pitches, magnitudes = librosa.piptrack(y=wav, sr=sr, hop_length=hop_length)
    
    # Get the pitch with maximum magnitude at each frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(pitch if pitch > 0 else 0.0)
    
    return np.array(pitch_values)

def extract_energy(wav: np.ndarray, hop_length: int = 256) -> np.ndarray:
    """Extract energy (RMS) from audio."""
    energy = librosa.feature.rms(y=wav, hop_length=hop_length)[0]
    return energy

def compute_durations(phonemes: List[str], mel_length: int) -> List[int]:
    """Simple duration estimation - distribute frames evenly across phonemes."""
    if not phonemes:
        return []
    
    base_duration = mel_length // len(phonemes)
    remainder = mel_length % len(phonemes)
    
    durations = [base_duration] * len(phonemes)
    for i in range(remainder):
        durations[i] += 1
    
    return durations

def prepare_dataset(config_path: str):
    """Main function to prepare dataset for training."""
    cfg = load_config(config_path)
    
    # Create output directories
    output_dir = cfg['paths']['output_dir']
    mel_dir = os.path.join(output_dir, 'mels')
    makedirs(mel_dir)
    
    # Scan dataset
    metadata_path = os.path.join(cfg['paths']['metadata_dir'], 'metadata.txt')
    audio_pairs = scan_dataset(metadata_path)
    
    # Collect all phonemes to build vocabulary
    all_phonemes = set()
    for wav_path, transcript in audio_pairs:
        phonemes = sentence_to_phonemes(transcript)
        all_phonemes.update(phonemes)
    
    # Create tokenizer
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    tokenizer = PhonemeTokenizer(special_tokens + sorted(list(all_phonemes)))
    
    # Save tokenizer
    tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))
    
    # Process each audio file
    train_manifest = []
    val_manifest = []
    
    for i, (wav_path, transcript) in enumerate(audio_pairs):
        try:
            # Load audio
            full_wav_path = wav_path
            audio, sr = load_audio(full_wav_path, cfg['audio']['sample_rate'])
            
            # Compute mel spectrogram
            mel = compute_mel(
                audio, sr, 
                n_mels=cfg['audio']['n_mels'],
                hop_length=cfg['audio']['hop_size'],
                n_fft=cfg['audio']['win_size']
            )
            
            # Extract pitch and energy
            pitch = extract_pitch(audio, sr, cfg['audio']['hop_size'])
            energy = extract_energy(audio, cfg['audio']['hop_size'])
            
            # Convert text to phonemes
            phonemes = sentence_to_phonemes(transcript)
            token_ids = tokenizer.encode(phonemes)
            
            # Estimate durations (simplified)
            durations = compute_durations(phonemes, mel.shape[0])
            
            # Save mel spectrogram
            base_name = Path(wav_path).stem
            mel_filename = f"{base_name}.npy"
            mel_path = os.path.join(mel_dir, mel_filename)
            np.save(mel_path, mel)
            
            # Create manifest entry
            manifest_entry = {
                'id': base_name,
                'mel_file': mel_filename,
                'token_ids': token_ids,
                'phonemes': phonemes,
                'text': transcript,
                'durations': durations,
                'pitch': pitch.tolist(),
                'energy': energy.tolist(),
                'mel_length': mel.shape[0],
                'text_length': len(phonemes)
            }
            
            # Split into train/val (80/20)
            if i % 5 == 0:  # 20% for validation
                val_manifest.append(manifest_entry)
            else:
                train_manifest.append(manifest_entry)
                
            logger.info(f"Processed {i+1}/{len(audio_pairs)}: {base_name}")
            
        except Exception as e:
            logger.error(f"Error processing {wav_path}: {e}")
            continue
    
    # Save manifests
    with open(os.path.join(output_dir, 'train_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(train_manifest, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'val_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(val_manifest, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dataset preparation complete!")
    logger.info(f"Training samples: {len(train_manifest)}")
    logger.info(f"Validation samples: {len(val_manifest)}")
    logger.info(f"Vocabulary size: {len(tokenizer.tokens)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset for TTS training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    prepare_dataset(args.config)
