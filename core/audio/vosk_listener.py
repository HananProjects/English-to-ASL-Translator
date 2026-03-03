import json
import queue
import threading

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model

from core.mic_utils import _resample_int16_mono, resolve_input_device_with_rate


class VoskListener:
    def __init__(self, model_path=None, on_text=None, input_device=None, model=None):
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = Model(model_path)
        else:
            raise ValueError("model_path or model is required")
        self.recognizer = None
        self.audio_queue = queue.Queue()
        self.on_text = on_text
        self.input_device = input_device
        self.running = False
        self.latest_text = ""
        self.final_segments = []
        self.sample_rate = None
        self.capture_rate = None
        self._thread = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
        chunk = bytes(indata)
        if self.capture_rate and self.capture_rate != self.sample_rate:
            audio = np.frombuffer(chunk, dtype=np.int16)
            chunk = _resample_int16_mono(audio, self.capture_rate, self.sample_rate).tobytes()
        self.audio_queue.put(chunk)

    def start(self):
        if self.running:
            return
        self.audio_queue = queue.Queue()
        self.latest_text = ""
        self.final_segments = []
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, wait: bool = False):
        self.running = False
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass
        if wait and self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.latest_text

    def _append_text(self, text: str):
        clean = (text or "").strip()
        if not clean:
            return
        self.final_segments.append(clean)
        self.latest_text = " ".join(self.final_segments).strip()
        if self.on_text is not None:
            self.on_text(self.latest_text)

    def _run(self):
        try:
            device, capture_rate = resolve_input_device_with_rate(self.input_device)
            self.capture_rate = capture_rate
            self.sample_rate = 16000
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

            with sd.RawInputStream(
                samplerate=capture_rate,
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=self._callback,
                device=device,
            ):
                print(
                    f"Vosk listening at {capture_rate} Hz "
                    f"(recognizer={self.sample_rate} Hz)..."
                )
                while self.running or not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    if data is None:
                        continue
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        self._append_text(text)

            if self.recognizer is not None:
                result = json.loads(self.recognizer.FinalResult())
                text = result.get("text", "").strip()
                self._append_text(text)
        finally:
            self.running = False
            self._thread = None
