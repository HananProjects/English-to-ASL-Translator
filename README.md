# English-to-ASL Translator

CME495 Capstone Project (2025–2026)

An offline, end-to-end system that translates spoken or written English into American Sign Language (ASL) gloss tokens using deterministic grammar rules and speech-to-text processing.

---

## 🎯 Project Goal

The goal of this project is to build a **real-time, offline-capable English ↔ ASL translation system** suitable for embedded platforms such as the Raspberry Pi.

This repository implements the **English → ASL** pipeline, including:
- Speech-to-text (STT)
- Text normalization
- ASL grammar transformation
- Confidence scoring
- End-to-end audio demo

---

## 🚀 Key Features

- ✅ Offline speech-to-text (Vosk)
- ✅ Deterministic ASL grammar rules
- ✅ End-to-end audio → ASL translation
- ✅ Composite confidence scoring
- ✅ Latency measurement
- ✅ Clean, modular architecture
- ✅ Python 3.9 compatible (Raspberry Pi friendly)

---

## 🧠 System Overview

**Pipeline flow:**

The system is designed so that **each stage is isolated**, testable, and replaceable.

---

## 🎧 Running the Audio Demo

### 1. Install dependencies

```bash
pip install -r requirements.txt

Download an English model from:
https://alphacephei.com/vosk/models

Extract it to:


3. Prepare a WAV file

The audio file must be:
	•	16 kHz
	•	Mono
	•	PCM WAV

You can convert using ffmpeg:


Run the demo
python -m scripts.run_audio_to_asl samples/demo.wav