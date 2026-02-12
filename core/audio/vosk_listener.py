import json
import queue
import threading

import sounddevice as sd
from vosk import KaldiRecognizer, Model

from core.mic_utils import resolve_input_device_with_rate


class VoskListener:
    def __init__(self, model_path, on_text, input_device=None):
        self.model = Model(model_path)
        self.recognizer = None
        self.audio_queue = queue.Queue()
        self.on_text = on_text
        self.input_device = input_device
        self.running = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(bytes(indata))

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def _run(self):
        device, sample_rate = resolve_input_device_with_rate(self.input_device)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)

        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._callback,
            device=device,
        ):
            print(f"Vosk listening at {sample_rate} Hz...")
            while self.running:
                data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        self.on_text(text)
