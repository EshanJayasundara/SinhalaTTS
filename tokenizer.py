"""Tokenize phonemes to integer ids and back. Save/load vocab.
"""
from typing import List, Dict
import json
from pathlib import Path

class PhonemeTokenizer:
    def __init__(self, tokens: List[str] = None):
        tokens = tokens or []
        self.tokens = tokens
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {i: t for i, t in enumerate(tokens)}

    def encode(self, phonemes: List[str]) -> List[int]:
        return [self.token2id.get(p, self.token2id.get('<unk>', 0)) for p in phonemes]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(i, '<unk>') for i in ids]

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: str):
        tokens = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(tokens=tokens)
    