"""Skeleton PyTorch model for an acoustic model.
You can replace the body with a real FastSpeech2/Tacotron/VITS implementation.
"""
import torch
import torch.nn as nn

class SimpleAcousticModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, mel_dim: int = 80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
        nn.Linear(embed_dim, mel_dim),
        )

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.encoder(x)
        mel = self.decoder(x)
        return mel