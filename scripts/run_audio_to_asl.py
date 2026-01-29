import sys
from core.audio_utils import load_wav_as_pcm
from core.engine import english_to_asl


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_audio_to_asl.py <audio.wav>")
        sys.exit(1)

    wav_path = sys.argv[1]

    audio_bytes, sr = load_wav_as_pcm(wav_path)

    result = english_to_asl(audio=audio_bytes)

    if result.error:
        print(f"ERROR: {result.error}")
        sys.exit(1)

    print("ASL TOKENS:", " ".join(result.asl_tokens))


if __name__ == "__main__":
    main()