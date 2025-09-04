"""End-to-end inference script.
Text -> phonemes -> tokens -> acoustic_model -> mel -> vocoder -> wav
"""
import argparse
import numpy as np
from g2p import sentence_to_phonemes
from tokenizer import PhonemeTokenizer
# from acoustic_model import load_model
# from vocoder import load_vocoder

def synthesize(text: str, tokenizer: PhonemeTokenizer, acoustic_model, vocoder, out_wav: str):
    phonemes = sentence_to_phonemes(text)
    token_ids = tokenizer.encode(phonemes)
    # convert to tensor and call acoustic model
    # mel = acoustic_model.forward(...)
    # wav = vocoder.forward(mel)
    # save wav to out_wav
    print('Synthesize pipeline executed (skeleton)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--out', type=str, default='out.wav')
    args = parser.parse_args()
    synthesize(args.text, None, None, None, args.out)