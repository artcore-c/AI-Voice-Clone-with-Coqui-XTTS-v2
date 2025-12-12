# AI Voice Cloning with Coqui XTTS-v2
Free voice cloning for creators using Coqui XTTS-v2 on Google Colab. Clone your voice with just 2-5 minutes of audio for consistent narration. Tutorial + working notebook included. Non-commercial use only.

## Overview

**Coqui XTTS-v2** is a multilingual text-to-speech model with zero-shot voice cloning capabilities. It uses a Transformer architecture similar to GPT-style autoregressive models combined with a VQ-VAE (Vector Quantized Variational AutoEncoder) to generate realistic speech in 16+ languages from just a few seconds of reference audio.

### How It Works

**Voice Cloning Process:**
- **Audio Analysis:** The model extracts acoustic features from your reference audio (pitch, tone, speaking style, cadence)
- **Voice Encoding:** These features are encoded into a speaker embedding vector
- **Text-to-Speech Generation:** Given new text, the model generates speech that matches your voice characteristics
- **Waveform Synthesis:** The output is synthesized into a high-quality audio file

**Technical Stack:**
- **Model:** XTTS-v2 (1.8GB pretrained model from Coqui AI)
- **Framework:** PyTorch 2.1.0 with CUDA support
- **Inference:** Runs on Google Colab's free T4 GPU (16GB VRAM)
- **Sample Rate:** 24kHz output
- **Languages:** Supports 16 languages including English, Spanish, French, German, Japanese, and more

### Why Google Colab?
Google Colab provides free access to GPU-accelerated computing, which is essential for running large neural network models like XTTS-v2. Voice synthesis on CPU would take significantly longer (10-20x slower). The free T4 GPU tier is sufficient for generating voice clones without requiring local hardware or paid cloud services.

### Intended Use Cases

- Consistent narration for storytelling, tutorials, and educational content
- Editing specific audio sections without full re-recording
- Creating voiceovers when recording conditions aren't ideal
- Maintaining voice consistency across multiple recording sessions
- Generating placeholder audio for video editing workflows

___
## Requirements
- Google account (for Google Colab and Google Drive)
- 2-5 minutes of clean audio in WAV format
  - Best results: clear speech, minimal background noise
  - Mix of scripted and natural speaking recommended
- Google Colab with T4 GPU runtime (available with free plan but subject to usage limits)
- No Python installation needed locally (runs in Colab)

## Prerequisites
### 🎤Audio File
- .wav or .mp3 sample audio file uploaded to your Google Drive
- 2-5 minutes in length
- 16-bit or 24-bit, 44.1kHz or 48kHz sample rate recommended

#### Converting Audio to WAV

**macOS (built-in tool):**
```zsh
# afconvert comes pre-installed on macOS
afconvert -f WAVE -d LEI16 input.m4a output.wav
```

**Mac/Linux/Windows (use ffmpeg):**

Install ffmpeg first:
```zsh
# macOS with MacPorts
sudo port install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

Convert audio:
```zsh
ffmpeg -i input.m4a -ar 24000 output.wav
```

Supported input formats: .m4a, .mp3, .mp4, .mov, and most audio/video formats.

#### See **Notes:** section below for Hardware Recommendations

___
## 🎬 Video Tutorial

<table style="border: 0; border-spacing: 0; border-collapse: collapse; background: transparent;">
<tr style="border: 0;">
<td style="vertical-align: top; border: 0; padding: 0; padding-right: 24px; background: transparent;">
  <a href="https://www.youtube.com/@3DCharacterArt/featured">
    <img src="https://img.icons8.com/color/48/youtube-play.png" alt="YouTube" width="64" height="64"/>
  </a>
</td>
<td style="vertical-align: top; border: 0; padding: 0; background: transparent;">
  <h2 style="margin-top: 0;">YouTube Companion Video:</h2>
  <strong>Watch the tutorial on YouTube <a href="https://www.youtube.com/@3DCharacterArt/featured">Free AI Voice Cloning for Creators with Coqui-XTTS-v2</a></strong>
</td>
</tr>
</table>

This repository was created as a companion to the video tutorial covering:
- **Coqui XTTS-v2** setup with Google Colab

___
## 🚀 Quick Start

1. Open the Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artcore-c/AI-Voice-Clone-Coqui-XTTS-v2-Google-Colab/blob/main/Coqui-XTTS-v2-Google-Colab.ipynb)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run cells 1-4 in order (takes ~5 minutes first time)
4. Upload your audio file when prompted
5. Edit the text you want generated in Cell 6
6. Download your cloned voice!

---

### Cell 1 - Install Python 3.11:
```bash
!apt-get update -qq
!apt-get install -y python3.11 python3.11-venv python3.11-dev
```
###### Note: Python 3.11 is the only recommended version tested for compatibility with this notebook. Other versions may trigger runtime errors.

### Cell 2 - Create virtual environment and install TTS:
```bash
!python3.11 -m venv /content/py311env
!/content/py311env/bin/pip install --upgrade pip
!/content/py311env/bin/pip install TTS
```

### Install additional requirements:
> **Transformers with** `BeamSearchScorer`
```bash
!/content/py311env/bin/pip install "transformers<4.50.0"
```

> **PyTorch 2.1.x** 
```bash
!/content/py311env/bin/pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```
___
### Cell 3 - Create a Python Script to Load the Model:
```python
%%writefile /content/load_model.py
import os
os.environ['MPLBACKEND'] = 'Agg'
from TTS.api import TTS
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
print(f'Model loaded on {device}!')
```
___
### Cell 4 - Run the Model with Python 3.11:
```bash
!/content/py311env/bin/python /content/load_model.py
```
**When prompted:**
Type ```y``` and press Enter to agree to the non-commercial license (CPML).

### Cell 5 - Upload your audio file:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 6 - Generate cloned voice:
```python
%%writefile /content/generate_voice.py
import os
os.environ['MPLBACKEND'] = 'Agg'
from TTS.api import TTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)

# Insert your text here to replace the example "..."
text = "He became as good a friend, as good a master, and as good a man as the good old city knew." 

# Generate speech
tts.tts_to_file(
    text=text,
    speaker_wav="/content/drive/MyDrive/Your_Audio_File.wav",  # <-- change this to match your audio file
    language="en",
    file_path="/content/cloned_voice.wav"
)

print("Voice generated: /content/cloned_voice.wav")
```
#### Note: Replace ```Your_Audio_File.wav``` with your own recorded audio sample filename

### Cell 6b - Run the script:
```python
!/content/py311env/bin/python /content/generate_voice.py
```

### Cell 7 - Download your cloned voice:
```python
from google.colab import files
files.download("/content/cloned_voice.wav")
```
___
## Notes:

### Recording Equipment (Minimum Recommended)

**Recommended for best results while recording audio samples:**
- USB audio interface (*we used an Arturia MiniFuse 2*)
- Condenser or shotgun microphone (*we used an Audio-Technica AT875R*)
- Quiet recording environment

**Acceptable minimum:**
- Smartphone (*eg. iPhone 8+*) in a quiet room
- USB microphone with cardioid pattern
- Desktop/Laptop built-in mic in very quiet environment (quality will be lower)

**Background noise:** 
- More important than mic quality. Record in a quiet space.

___
## PyTorch & CUDA Compatibility
##### This notebook uses:
```python
torch==2.1.0
torchaudio==2.1.0
```
> **installed from the CUDA 11.8 wheel index:** `https://download.pytorch.org/whl/cu118`

##### CUDA 11.8 is compatible with Colab’s common T4 GPU hardware. If a different GPU is assigned, PyTorch may fallback to CPU

## Transformers Version Requirement
##### This notebook also pins:
```python
transformers < 4.50.0
```

> **to ensure** ```BeamSearchScorer``` **remains available and XTTS-v2 loads correctly**

___
## License

This repository's code and documentation: MIT License

**However:** The Coqui XTTS-v2 model used in this tutorial is licensed under 
the Coqui Public Model License (CPML), which restricts usage to non-commercial 
purposes only. See https://coqui.ai/cpml for details.

## Acknowledgements

This project builds upon:
- **[Coqui TTS](https://github.com/coqui-ai/TTS)** - The XTTS-v2 model and framework
- **[Google Colab](https://colab.research.google.com)** - Free GPU infrastructure
- **[PyTorch](https://pytorch.org)** - Deep learning framework

We're grateful to the open-source community for making voice cloning accessible to all creators.

## ⚠️ GPU Usage Limits

**Colab Free Plan Limitations:**
- In the free version of Colab notebooks can run for at most 12 hours, depending on availability and usage patterns. 
- Colab Pro and Pay As You Go offer increased compute availability based on your compute unit balance.
- If unavailable, wait 12+ hours or consider Colab Pro ($9.99/month) for increased access.


## Support
Questions? Check the video tutorial or open an issue!
