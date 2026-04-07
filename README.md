# English-to-ASL Translator

CME495 Capstone Project (2025-2026)

This project is an offline English and ASL translation prototype designed for a Raspberry Pi environment. It supports two user-facing modes:

- English to ASL: converts typed or spoken English into ASL gloss tokens and plays pose-based sign animation clips
- ASL to English: uses live camera landmarks plus a trained recognizer to classify signs and build English output

## Scope

The final application entry point is:

```bash
python -m ui.main
```

This repository also contains training, import, and evaluation scripts, but those are development utilities and are not required to run the main application.

## Main Features

- Offline speech-to-text using Vosk
- Rule-based English to ASL token mapping
- Pose-clip driven ASL animation playback
- Live ASL recognition from camera landmarks
- Reverse translation from recognized ASL tokens to English text
- Raspberry Pi friendly runtime configuration through environment variables

## Project Layout

- `core/`: translation pipelines, recognizers, feature extraction, and shared logic
- `ui/`: PySide6 application, animation playback, and camera worker code
- `scripts/`: helper scripts for running, training, importing clips, and evaluation
- `tests/`: unit tests for translation logic and recognition components
- `data/`: labels and development artifacts
- `models/`: local model files required at runtime but not committed to git

## Requirements

- Python 3.11 recommended for local desktop use
- Python 3.9+ should work for the main project code
- A working microphone for English-to-ASL speech input
- A working camera for ASL-to-English live recognition

Install dependencies:

```bash
pip install -r requirements.txt
```

## Runtime Models

This repository expects local model assets that are intentionally not committed:

1. A Vosk speech model extracted to `models/vosk-en`
2. An ASL recognizer model in `models/`

Recommended recognizer filename:

```text
models/asl_landmark_classifier_v2.npz
```

If `asl_landmark_classifier_v2.npz` is not present, the app falls back to `asl_landmark_classifier_v1.npz` if available.

## Running The Final UI

Start the full application with:

```bash
python -m ui.main
```

The UI contains two modes:

- English -> ASL: record speech or enter text, then view the generated sign sequence
- ASL -> English: use the live camera recognizer to detect signs and convert them to English

## Useful Scripts

Run a simple English-to-ASL text example:

```bash
python -m scripts.run_english_to_asl
```

Run a simple ASL-to-English token example:

```bash
python -m scripts.run_asl_to_english
```

## Testing

Run the test suite with:

```bash
pytest -q
```

## Notes For Submission

- Large local models are expected in `models/` but are not tracked in git
- The repository includes training and import utilities that are not part of the main runtime path
- Generated UI state, evaluation outputs, backup clips, and raw source videos should not be included in the final cleaned submission

## Known Limitations

- Recognition quality depends on lighting, framing, and camera position
- The ASL recognizer is vocabulary-limited and depends on the available trained labels
- The English-to-ASL pipeline is rule-based, so output is constrained by the project dictionary and grammar rules
