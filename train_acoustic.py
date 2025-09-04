"""Training loop skeleton for acoustic model.
- load dataset manifest
- DataLoader that yields token_ids, mel_targets
- optimizer, scheduler, checkpointing
- tensorboard logging
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from acoustic_model import SimpleAcousticModel
from utils import load_config, makedirs, get_logger

logger = get_logger(__name__)

class TTSDataset(Dataset):
    def __init__(self, manifest_path: str):
        # TODO: read manifest, load precomputed mels
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # return token_ids (tensor), mel (tensor)
        return None

def train(config_path: str):
    cfg = load_config(config_path)
    # TODO: instantiate model, dataset, dataloader
    model = SimpleAcousticModel(vocab_size=100)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # placeholder loop
    for epoch in range(1):
        logger.info('Epoch %d', epoch)
        # iterate batches

if __name__ == '__main__':
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    train(cfg)

# TODO #