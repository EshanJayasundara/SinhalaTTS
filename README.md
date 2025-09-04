**[Architecture](#docs/architecture.md)**

**Quickstart(after complete the development)**
```
pip install -r requirements.txt
```

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```
python data_prep.py
```

```
python train_acoustic.py config.yaml
```

```
python train_vocoder.py config.yaml
```

```
python inference.py --text "බොහෝ දෙනෙක් සුදු ඇඳුම මෙබඳු අවස්ථාවල දී භාවිතයට ගනිති" --out outputs/test.wav
```



**Background Research**

doc: https://docs.google.com/document/d/1MWrTextfkxukDgJf6b0Tx6KQ-6irLEUDZ8rTX22BAa4/edit?usp=drive_link

Dataset(sample): https://huggingface.co/datasets/eshangj/SinhalaTTS

**Available Resources**

- https://github.com/TensorSpeech/TensorFlowTTS
- https://github.com/huggingface/parler-tts
- https://github.com/2noise/ChatTTS
- https://github.com/yl4579/StyleTTS2

**Traditional neural TTS**

- had two stages:
  - Text → Mel-spectrogram (Tacotron2 / FastSpeech2)
  - Mel-spectrogram → Waveform (HiFi-GAN / WaveGlow)

- Problems:
  - Error compounding (bad spectrogram = bad audio)
  - Prosody (intonation, rhythm) is hard to control
  - Two models = harder to train/deploy
 
**VITS**

- has three major blocks:
  - Text Encoder (phoneme embeddings → linguistic features)
  - Stochastic Variational Latent Space (learns natural variation in speech: prosody, style)
  - Generator (HiFi-GAN-like vocoder) (latent → waveform)

- During training, there’s also a Posterior Encoder that learns from real audio to guide the latent space.
- paper: https://arxiv.org/pdf/2106.06103 <br/>
- github: https://github.com/jaywalnut310/vits <br/>
- demo: https://jaywalnut310.github.io/vits-demo/index.html <br/>
