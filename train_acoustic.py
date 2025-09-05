"""Training loop for FastSpeech2 acoustic model."""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import os
import numpy as np
from pathlib import Path
import time
import logging

from acoustic_model import FastSpeech2AcousticModel
from utils import load_config, makedirs, get_logger

logger = get_logger(__name__)

class TTSDataset(Dataset):
    def __init__(self, manifest_path: str, mel_dir: str):
        self.mel_dir = mel_dir
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.items = json.load(f)
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load token IDs
        token_ids = torch.tensor(item['token_ids'], dtype=torch.long)
        
        # Load mel spectrogram
        mel_path = os.path.join(self.mel_dir, item['mel_file'])
        mel = torch.from_numpy(np.load(mel_path)).float()
        
        # Load durations, pitch, and energy if available
        durations = torch.tensor(item['durations'], dtype=torch.long) if 'durations' in item else None
        pitch = torch.tensor(item['pitch'], dtype=torch.float) if 'pitch' in item else None
        energy = torch.tensor(item['energy'], dtype=torch.float) if 'energy' in item else None
        
        return {
            'token_ids': token_ids,
            'mel': mel,
            'durations': durations,
            'pitch': pitch,
            'energy': energy
        }

def collate_fn(batch):
    """Custom collate function to pad sequences."""
    token_ids = [item['token_ids'] for item in batch]
    mels = [item['mel'] for item in batch]
    
    # Get durations, pitch, energy if available
    has_durations = batch[0]['durations'] is not None
    has_pitch = batch[0]['pitch'] is not None
    has_energy = batch[0]['energy'] is not None
    
    durations = [item['durations'] for item in batch] if has_durations else None
    pitches = [item['pitch'] for item in batch] if has_pitch else None
    energies = [item['energy'] for item in batch] if has_energy else None
    
    # Pad token sequences
    token_lengths = [x.size(0) for x in token_ids]
    max_token_len = max(token_lengths)
    token_ids_padded = torch.zeros(len(batch), max_token_len, dtype=torch.long)
    for i, (seq, length) in enumerate(zip(token_ids, token_lengths)):
        token_ids_padded[i, :length] = seq
    
    # Pad mel sequences
    mel_lengths = [x.size(0) for x in mels]
    max_mel_len = max(mel_lengths)
    n_mels = mels[0].size(1)
    mels_padded = torch.zeros(len(batch), max_mel_len, n_mels)
    for i, (seq, length) in enumerate(zip(mels, mel_lengths)):
        mels_padded[i, :length, :] = seq
    
    # Pad durations, pitch, energy if available
    durations_padded = None
    if has_durations:
        durations_padded = torch.zeros(len(batch), max_token_len, dtype=torch.long)
        for i, (seq, length) in enumerate(zip(durations, token_lengths)):
            durations_padded[i, :length] = seq
    
    pitch_padded = None
    if has_pitch:
        pitch_padded = torch.zeros(len(batch), max_token_len)
        for i, (seq, length) in enumerate(zip(pitches, token_lengths)):
            pitch_padded[i, :length] = seq
    
    energy_padded = None
    if has_energy:
        energy_padded = torch.zeros(len(batch), max_token_len)
        for i, (seq, length) in enumerate(zip(energies, token_lengths)):
            energy_padded[i, :length] = seq
    
    return {
        'token_ids': token_ids_padded,
        'token_lengths': torch.tensor(token_lengths),
        'mel': mels_padded,
        'mel_lengths': torch.tensor(mel_lengths),
        'durations': durations_padded,
        'pitch': pitch_padded,
        'energy': energy_padded
    }

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """Save training checkpoint."""
    makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    logger.info('Saved checkpoint to %s', checkpoint_path)

def train(config_path: str):
    cfg = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)
    
    # Create datasets and dataloaders
    train_dataset = TTSDataset(
        manifest_path=os.path.join(cfg['paths']['metadata_dir'], 'metadata.txt'),
        mel_dir=cfg['paths']['output_dir']
    )
    
    val_dataset = TTSDataset(
        manifest_path=os.path.join(cfg['paths']['metadata_dir'], 'val_manifest.json'),
        mel_dir=cfg['paths']['output_dir']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Instantiate model
    model = FastSpeech2AcousticModel(
        vocab_size=100,  # You'll need to set this based on your tokenizer
        mel_dim=cfg['audio']['n_mels']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    
    # TensorBoard
    writer = SummaryWriter(cfg['paths']['output_dir'])
    
    # Training loop
    for epoch in range(cfg['training']['epochs']):
        logger.info('Epoch %d', epoch)
        model.train()
        
        # Training
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            token_ids = batch['token_ids'].to(device)
            mel_target = batch['mel'].to(device)
            
            # Get optional features
            durations = batch['durations'].to(device) if batch['durations'] is not None else None
            pitch = batch['pitch'].to(device) if batch['pitch'] is not None else None
            energy = batch['energy'].to(device) if batch['energy'] is not None else None
            
            # Forward pass
            optimizer.zero_grad()
            mel_pred, pred_durations, pred_pitch, pred_energy = model(
                token_ids, durations, pitch, energy
            )
            
            # Loss calculation (simple MSE for now)
            loss = torch.nn.functional.mse_loss(mel_pred, mel_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            if batch_idx % 100 == 0:
                logger.info('Epoch: %d, Batch: %d, Loss: %.4f', epoch, batch_idx, loss.item())
                writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch['token_ids'].to(device)
                mel_target = batch['mel'].to(device)
                
                durations = batch['durations'].to(device) if batch['durations'] is not None else None
                pitch = batch['pitch'].to(device) if batch['pitch'] is not None else None
                energy = batch['energy'].to(device) if batch['energy'] is not None else None
                
                mel_pred, _, _, _ = model(token_ids, durations, pitch, energy)
                val_loss += torch.nn.functional.mse_loss(mel_pred, mel_target).item()
        
        val_loss /= len(val_loader)
        logger.info('Validation loss: %.4f', val_loss)
        writer.add_scalar('val/loss', val_loss, epoch)
        
        # Save checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, cfg['paths']['output_dir'])
    
    # Save final model
    final_model_path = os.path.join(cfg['paths']['output_dir'], 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info('Saved final model to %s', final_model_path)
    
    writer.close()

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    train(config_path)