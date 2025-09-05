"""Simple grapheme-to-phoneme module.
Provide a rule-based fallback and hooks for a dictionary-based G2P.
"""
from typing import List
from utils import read_file
from utils import read_file
from typing import Dict
import re


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

from typing import List

def apply_consonant_rule_01(phonemes: List[str]) -> List[str]:
    """
    Rule 01: Duplicate consonant before RSY (rə) or YSY (j)
    Skip duplication if RSY/YSY is in the first syllable or preceded by a cluster.
    """
    result = []
    i = 0

    # Determine index of end of first syllable
    first_syllable_end = 0
    vowels = ["a", "e", "i", "o", "u", "ə", "æ", "rə"]
    for idx, p in enumerate(phonemes):
        if p in vowels:
            first_syllable_end = idx
            break

    while i < len(phonemes):
        p = phonemes[i]

        # RSY duplication
        if p == "rə" and i > first_syllable_end:
            prev = result[-1]
            # crude consonant check
            if prev not in vowels:
                result.append(prev)

        # YSY duplication
        elif p == "j" and i > first_syllable_end:
            prev = result[-1]
            if prev not in vowels:
                result.append(prev)

        result.append(p)
        i += 1

    return result


def apply_consonant_rule_02(phonemes: List[str]) -> List[str]:
    """
    Rule 02: Insert 'k' before /ɲ/ when followed by a vowel
    """
    result = []
    vowels = {"a", "ə", "i", "e", "o", "u", "a:", "i:", "e:", "o:", "u:"}

    for i, p in enumerate(phonemes):
        if p == "ɲ":
            # Check next phoneme exists and is a vowel
            if i + 1 < len(phonemes) and phonemes[i + 1] in vowels:
                result.append("k")  # Insert 'k' before 'ɲ'
        result.append(p)
    return result


def apply_vowel_rules(phonemes: List[str]) -> List[str]:
    """
    Apply Sinhala vowel rules on a phoneme sequence.
    Handles first-syllable schwa, word-final schwa, clusters, and special cases.
    """
    vowels = ["a", "e", "i", "o", "u", "ə", "æ"]

    # Add boundary markers
    text = "# " + " ".join(phonemes) + " #"

    # Vowel Rule 01: first syllable schwa → /a/ (if consonant+schwa)
    def repl_first_syllable(match):
        cons = match.group(1)
        return f"{cons}a"
    text = re.sub(r"# ([^aeiouə\s]+)ə", repl_first_syllable, text)

    # Vowel Rule 02: /[consonant]rəh/ -> /[consonant]rah/
    text = re.sub(r"([^aeiouə])rə h", r"\1ra h", text)

    # Vowel Rule 03: ([a, e, æ, o, ə]hə) -> ([a, e, æ, o, ə]ha)
    text = re.sub(r"([aeæoə]) h ə", r"\1 h a", text)

    # Vowel Rule 04: /əCC/ -> /aCC/
    text = re.sub(r"ə ([^aeiouə]) ([^aeiouə])", r"a \1 \2", text)

    # Vowel Rule 05: word-final schwa before consonant -> a (exceptions)
    text = re.sub(r"ə ([^aeiou#]) #", r"a \1 #", text)
    text = re.sub(r"a (r|b|ɖ|ʈ) #", r"ə \1 #", text)

    # Vowel Rule 06: schwa before 'ji' or 'vu' at word end -> a
    text = re.sub(r"ə (ji|vu) #", r"a \1 #", text)

    # Vowel Rule 07: /kə[r,l]u/ -> /ka[r,l]u/
    text = re.sub(r"k ə (r|l) u", r"k a \1 u", text)

    # Vowel Rule 08: kal special cases
    text = re.sub(r"kal (a:|e:|o:) j", r"kəl \1 j", text)
    text = re.sub(r"kale ([mh]) ([ui])", r"kəle \1 \2", text)
    text = re.sub(r"kal ə h ([ui])", r"kəle h \1", text)
    text = re.sub(r"kal ə", r"kəl ə", text)

    # Remove boundary markers and split back to phonemes
    return text.replace("#", "").strip().split()

DIPHTHONGS = {
    "i vu": "iu",
    "i: vu": "i:u",
    "e vu": "eu",
    "e: vu": "e:u",
    "æ vu": "æu",
    "æ: vu": "æ:u",
    "o vu": "ou",
    "a vu": "au",
    "a: vu": "a:u",
    "u yi": "ui",
    "u: yi": "u:i",
    "o yi": "oi",
    "o: yi": "o:i",
    "a yi": "ai",
    "a: yi": "a:i",
    "e yi": "ei",
    "e: yi": "e:i",
    "æ yi": "æi",
    "æ: yi": "æ:i"
}

def apply_diphthong_rules(phonemes: List[str]) -> List[str]:
    text = " ".join(phonemes)
    for seq, replacement in DIPHTHONGS.items():
        text = text.replace(seq, replacement)
    return text.split()


def complete_g2p_conversion(word: str) -> List[str]:
    # Step 1: Get initial phoneme sequence
    phonemes = rough_g2p_conversion(word)

    # Step 2: Apply consonant rules
    phonemes = apply_consonant_rule_01(phonemes)
    phonemes = apply_consonant_rule_02(phonemes)

    # Step 3: Apply vowel rules
    phonemes = apply_vowel_rules(phonemes)

    # Step 4: Apply diphthong rules
    phonemes = apply_diphthong_rules(phonemes)

    return phonemes

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