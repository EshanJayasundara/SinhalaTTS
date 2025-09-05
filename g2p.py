"""Simple grapheme-to-phoneme module.
Provide a rule-based fallback and hooks for a dictionary-based G2P.
"""
from typing import List
from utils import read_file
from utils import read_file
from typing import Dict

def create_g2p_dict(path: str) -> Dict:
    GRAPHEME_TO_PHONE = {}
    for line in read_file(path):
        row = line.split("|")
        GRAPHEME_TO_PHONE[row[0]] = row[1]
    return GRAPHEME_TO_PHONE

def rough_g2p_conversion(word: str) -> str:
    vowels = create_g2p_dict("g2p-vowels.txt")
    modifiers = create_g2p_dict("g2p-modifiers.txt")
    consonants = create_g2p_dict("g2p-consonents.txt")
    rough_phoneme = ""
    i = 0
    while i < len(word):
        char = word[i]
        # vowels (standalone)
        if char in vowels:
            rough_phoneme += vowels[char]
        # consonants
        elif char in consonants:
            base = consonants[char]
            # if next is a modifier, apply it
            if i+1 < len(word) and word[i+1] in modifiers:
                if modifiers[word[i+1]] == '':
                    rough_phoneme += base
                else:
                    rough_phoneme += base + modifiers[word[i+1]]
                i += 1  # skip modifier
            else:
                # inherent vowel (schwa)
                rough_phoneme += base + "ə"
        # else: unknown char, keep as is
        else:
            raise Exception(f"Unknown character detected: {char!r}")
        i += 1
    return rough_phoneme

def complete_g2p_conversion(word: str) -> List[str]:
    rough_phoneme = rough_g2p_conversion(word)
    ### TODO
    phoneme = [rough_phoneme]
    ### 
    return phoneme

def word_to_phonemes(word: str) -> List[str]:
    phones = complete_g2p_conversion(word)
    return phones

def sentence_to_phonemes(sentence: str) -> List[str]:
    words = sentence.strip().split()
    sent_phones = []
    for w in words:
        sent_phones.extend(word_to_phonemes(w) + [' '])
    return sent_phones

# TODO #