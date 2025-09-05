"""FastSpeech2 PyTorch implementation for an acoustic model."""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
import math

# Initialize global embedding table for pitch and energy
_hidden_dim = 256   # must match model hidden dimension
_n_bins = 256       # number of discrete pitch/energy bins
_pitch_embedding_table = nn.Embedding(_n_bins, _hidden_dim)
_energy_embedding_table = nn.Embedding(_n_bins, _hidden_dim)

def pitch_embedding(pitch: torch.Tensor, min_val: float = 0.0, max_val: float = 500.0) -> torch.Tensor:
    """
    Convert scalar pitch values into hidden embeddings.
    
    Args:
        pitch: [B, T] tensor of pitch values
        min_val: minimum expected pitch
        max_val: maximum expected pitch

    Returns:
        [B, T, hidden_dim] embedding tensor
    """
    # Clamp pitch to expected range
    pitch_clamped = torch.clamp(pitch, min_val, max_val)
    
    # Normalize to [0, 1]
    pitch_norm = (pitch_clamped - min_val) / (max_val - min_val)
    
    # Quantize to bins
    bins = (pitch_norm * (_n_bins - 1)).long()
    
    # Lookup embedding
    return _pitch_embedding_table(bins)

def energy_embedding(energy: torch.Tensor, min_val: float = 0.0, max_val: float = 10.0) -> torch.Tensor:
    """
    Convert scalar energy values into hidden embeddings.
    
    Args:
        energy: [B, T] tensor of energy values
        min_val: minimum expected energy
        max_val: maximum expected energy

    Returns:
        [B, T, hidden_dim] embedding tensor
    """
    # Clamp energy to expected range
    energy_clamped = torch.clamp(energy, min_val, max_val)
    
    # Normalize to [0, 1]
    energy_norm = (energy_clamped - min_val) / (max_val - min_val)
    
    # Quantize to bins
    bins = (energy_norm * (_n_bins - 1)).long()
    
    # Lookup embedding
    return _energy_embedding_table(bins)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, src):
        src = self.pos_encoder(src)
        return self.transformer_encoder(src)

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, mel_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
        decoder_layers = TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.linear = nn.Linear(hidden_dim, mel_dim)
        
    def forward(self, tgt):
        tgt = self.pos_encoder(tgt)
        # For simplicity, we use the same sequence as both target and memory
        # In a more complete implementation, we might use a different approach
        output = self.transformer_decoder(tgt, tgt)
        return self.linear(output)

class PostNet(nn.Module):
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 512, num_layers: int = 5, kernel_size: int = 5):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = mel_dim if i == 0 else hidden_dim
            out_ch = mel_dim if i == num_layers - 1 else hidden_dim
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1)//2)
            layers.append(conv)
            if i != num_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(0.5))
            else:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.Dropout(0.5))
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, mel):
        """
        mel: [B, T, mel_dim]
        returns: residual [B, T, mel_dim]
        """
        x = mel.transpose(1, 2)  # [B, mel_dim, T]
        x = self.conv_stack(x)
        return x.transpose(1, 2)  # [B, T, mel_dim]

class VariancePredictor(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)  # predicts scalar per time step

    def forward(self, x):
        # x: [B, T, H] -> [B, H, T] for Conv1d
        x = x.transpose(1, 2)
        x = F.relu(self.layer_norm1(self.conv1(x).transpose(1,2))).transpose(1,2)
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.conv2(x).transpose(1,2))).transpose(1,2)
        x = self.dropout(x)
        x = self.linear(x.transpose(1,2)).squeeze(-1)  # [B, T]
        return x


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, durations):
        """
        x: [B, T, H] encoder outputs
        durations: [B, T] int tensor (number of frames per token)
        """
        output = []
        for batch_idx, (seq, dur) in enumerate(zip(x, durations)):
            expanded = []
            for i, d in enumerate(dur):
                d_int = int(d.item())
                expanded.append(seq[i].unsqueeze(0).expand(d_int, -1))
            if expanded:
                output.append(torch.cat(expanded, dim=0))
            else:
                # Handle empty case
                output.append(torch.zeros(0, seq.size(-1), device=x.device))
        
        # Pad sequences to same length
        max_len = max([o.size(0) for o in output]) if output else 0
        output_padded = []
        for o in output:
            if o.size(0) < max_len:
                pad = torch.zeros(max_len - o.size(0), o.size(1), device=o.device)
                o = torch.cat([o, pad], dim=0)
            output_padded.append(o.unsqueeze(0))
        
        if output_padded:
            return torch.cat(output_padded, dim=0)  # [B, T_mel, H]
        else:
            return torch.zeros(x.size(0), 0, x.size(2), device=x.device)


class FastSpeech2AcousticModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        mel_dim: int = 80,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder (stack of Transformer blocks)
        self.encoder = TransformerEncoder(
            embed_dim, hidden_dim, num_heads, num_layers
        )
        
        # Variance predictors
        self.duration_predictor = VariancePredictor(hidden_dim)
        self.pitch_predictor = VariancePredictor(hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim)
        
        # Length regulator (expands sequence by predicted durations)
        self.length_regulator = LengthRegulator()
        
        # Decoder (Transformer blocks again)
        self.decoder = TransformerDecoder(
            hidden_dim, mel_dim, num_heads, num_layers
        )
        
        # Optional post-net (Conv1D layers for mel refinement)
        self.postnet = PostNet(mel_dim)
        
        # Projection layer to match dimensions
        self.proj = nn.Linear(embed_dim, hidden_dim)

        self.hidden_dim = hidden_dim

    def forward(self, token_ids, durations=None, pitches=None, energies=None, max_duration=None):
        # Embed tokens
        x = self.embedding(token_ids)  # [B, T, E]
        
        # Project to hidden dimension if needed
        if x.size(-1) != self.hidden_dim:
            x = self.proj(x)
            
        # Encode sequence
        x = self.encoder(x)            # contextualized embeddings [B, T, H]

        # Variance modeling
        pred_durations = self.duration_predictor(x)
        pred_pitch = self.pitch_predictor(x)
        pred_energy = self.energy_predictor(x)

        # Use ground truth or predictions
        used_durations = pred_durations if durations is None else durations
        used_pitches = pred_pitch if pitches is None else pitches
        used_energies = pred_energy if energies is None else energies
        
        # Length regulation (expands sequence to match mel frames)
        x = self.length_regulator(x, used_durations)

        # Add pitch/energy embeddings
        if x.size(1) > 0:  # Only if we have sequences
            # We need to expand pitch/energy to match the regulated length
            pitch_emb = pitch_embedding(used_pitches)
            energy_emb = energy_embedding(used_energies)
            
            # Expand pitch and energy embeddings using the same durations
            pitch_emb_expanded = self.length_regulator(pitch_emb, used_durations)
            energy_emb_expanded = self.length_regulator(energy_emb, used_durations)
            
            x = x + pitch_emb_expanded + energy_emb_expanded

        # Decode into mel spectrograms
        mel = self.decoder(x)
        mel_residual = self.postnet(mel)
        mel_refined = mel + mel_residual  # refined mel

        return mel_refined, pred_durations, pred_pitch, pred_energy
    