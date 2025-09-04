"""Simple grapheme-to-phoneme module.
Provide a rule-based fallback and hooks for a dictionary-based G2P.
"""
from typing import List

# Minimal example mapping. Replace with a full mapping for Sinhala.
GRAPHEME_TO_PHONE = {
    'අ': 'a',
    'ඇ': 'ae',
    # TODO: extend
}

def word_to_phonemes(word: str) -> List[str]:
    phones = []
    for ch in word:
        phones.append(GRAPHEME_TO_PHONE.get(ch, ch))
    # simple normalization: collapse repeated symbols etc.
    return phones

def sentence_to_phonemes(sentence: str) -> List[str]:
    words = sentence.strip().split()
    sent_phones = []
    for w in words:
        sent_phones.extend(word_to_phonemes(w) + [' '])
    return sent_phones

# TODO #