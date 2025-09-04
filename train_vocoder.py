"""Training loop skeleton for vocoder.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


from vocoder import SimpleVocoder
from utils import load_config, get_logger


logger = get_logger(__name__)


class VocoderDataset(Dataset):
    def __init__(self, manifest_path: str):
        # TODO: read manifest
        self.items = []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return None

def train_vocoder(cfg_path: str):
    cfg = load_config(cfg_path)
    model = SimpleVocoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # training loop placeholder
    for epoch in range(1):
        logger.info('Vocoder epoch %d', epoch)


if __name__ == '__main__':
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    train_vocoder(cfg)