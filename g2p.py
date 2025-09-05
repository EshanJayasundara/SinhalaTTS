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

def rough_g2p_conversion(text: str) -> List[str]:
    vowels = create_g2p_dict("g2p-vowels.txt")
    modifiers = create_g2p_dict("g2p-modifiers.txt")
    consonants = create_g2p_dict("g2p-consonents.txt")

    phonemes: List[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in vowels:
            phonemes.append(vowels[char])
        elif char in consonants:
            base = consonants[char]
            if i + 1 < len(text) and text[i+1] in modifiers:
                mod = modifiers[text[i+1]]
                phonemes.append(base + mod if mod else base)
                i += 1
            else:
                phonemes.append(base + "ə")
        else:
            raise Exception(f"Unknown character detected: {char!r}")
        i += 1
    return phonemes

def complete_g2p_conversion(word: str) -> List[str]:
    rough_phoneme = rough_g2p_conversion(word)

    ### TODO

    # consonent rule 1

    # consonent rule 2

    # vowel rule 1

    # vowel rule 2

    # vowel rule 3

    # vowel rule 4

    # vowel rule 5

    # vowel rule 6

    # vowel rule 7

    # vowel rule 8

    # Rules of the diphthongs

    phoneme = rough_phoneme
    
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