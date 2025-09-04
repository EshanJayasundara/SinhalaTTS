# **Full Sinhala TTS Training Pipeline**

## **1. Text Processing**

**a. Collect Corpus**

* Large Sinhala text corpus (news, Wikipedia, books, etc.).

**b. Clean Text**

* Remove emojis, Latin letters, numbers (if not required).
* Normalize Sinhala punctuation (e.g., Sinhala-specific periods, commas).

**c. Grapheme-to-Phoneme (G2P)**

* Build or adapt a Sinhala phoneme dictionary.
* If no G2P is available:

  * Use direct grapheme-based modeling (works decently for phonetic scripts like Sinhala).
  * Later refine with a rule-based G2P.

**d. Text → Phoneme Sequence**

* Convert each sentence to phonemes (or graphemes).

  * Example: අයියා → `/a j j a/`

**e. Phonemes → Tokens**

* Map each phoneme to an integer ID (tokenizer).

  * Example: `/a/ → 1`, `/j/ → 2`

---

## **2. Acoustic Model (Phoneme → Mel-Spectrogram)**

**Embeddings**

* Convert token IDs into phoneme embeddings (trainable vectors).

**Model Choices**

* **FastSpeech2** – Transformer + Variance Adaptor
* **Tacotron2** – Seq2Seq + Attention
* **VITS** – End-to-end model (includes vocoder)

**Output**

* Predicts Mel-spectrograms (time-frequency representation of audio).

---

## **3. Audio Preprocessing**

**Dataset**

* Paired Sinhala `(text, speech)` dataset.
* Format: WAV files + transcriptions.

**Preprocessing Steps**

* Resample audio (16kHz or 22.05kHz).
* Normalize volume.
* Extract Mel-spectrograms using STFT (Short-Time Fourier Transform).

**Alignments (Optional)**

* Forced alignment using Montreal Forced Aligner (MFA) for Tacotron2.

---

## **4. Vocoder (Mel-Spectrogram → Waveform)**

**Purpose**

* Convert predicted Mel-spectrograms into natural-sounding speech.

**Vocoder Choices**

* HiFi-GAN (fast, high-quality)
* WaveGlow
* Parallel WaveGAN

**Training**

* Train vocoder separately on Sinhala audio `(mel → wav)`.
* Later plug vocoder into the acoustic model.

---

## **5. Inference Pipeline (Final TTS)**

**Pipeline Steps**

1. Text → Phonemes
2. Phonemes → Tokens
3. Tokens → Embeddings
4. Acoustic Model → Mel-Spectrogram
5. Vocoder → Waveform (WAV)
