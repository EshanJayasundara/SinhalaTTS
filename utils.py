import os
import yaml
import logging
from pathlib import Path
from typing import Iterator

def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def makedirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_logger(name: str = __name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

def read_file(path: str) -> Iterator[str]:
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()
        
# TODO #