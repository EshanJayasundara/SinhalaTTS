"""Vocoder skeleton (e.g., HiFi-GAN style). Replace with real model code.
"""
import torch
import torch.nn as nn

class SimpleVocoder(nn.Module):
    def __init__(self, mel_dim: int = 80):
        super().__init__()
        # A tiny example conv net
        self.net = nn.Sequential(
        nn.Conv1d(mel_dim, 512, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.Conv1d(512, 1, kernel_size=7, padding=3),
        )

    def forward(self, mel):
    # mel: (B, T, mel_dim) -> (B, 1, T)
        x = mel.transpose(1, 2)
        wav = self.net(x)
        return wav.squeeze(1)
    
# TODO #