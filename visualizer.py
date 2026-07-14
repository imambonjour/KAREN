import threading
import time
import wave

import numpy as np
import plotext as plt


class VoiceScope:
    def __init__(self, width=80):
        self.width = width
        self.history = [0.0] * width
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._mode = "flat"

        self._amplitudes = []
        self._chunk_duration = 0.001
        self._play_start = 0
        self._ref = 1.0
        self._samples_per_frame = 3
        self._frame_buf = []

        plt.theme("pro")

    def _draw(self):
        with self._lock:
            h = list(self.history)
        plt.clf()
        plt.plotsize(None, None)
        plt.ylim(-1.1, 1.1)
        plt.plot(h, marker="braille", color="green")
        plt.axes_color("black")
        plt.canvas_color("black")
        plt.frame(False)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def _update_speaking(self):
        elapsed = time.time() - self._play_start
        idx = int(elapsed / self._chunk_duration)
        if idx >= len(self._amplitudes):
            self._mode = "flat"
            self.history = [0.0] * self.width
            return
        amp = self._amplitudes[idx]
        level = amp / self._ref
        level = max(0.0, min(1.0, level))
        signed = level * (1 if (idx % 2 == 0) else -1)
        self._frame_buf.append(signed)
        if len(self._frame_buf) >= self._samples_per_frame:
            with self._lock:
                self.history.extend(self._frame_buf)
                self.history = self.history[-self.width:]
            self._frame_buf.clear()

    def _run(self):
        while self._running:
            if self._mode == "flat":
                pass
            elif self._mode in ("listening", "thinking"):
                pass
            elif self._mode == "speaking":
                self._update_speaking()
            self._draw()
            time.sleep(0.033)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def set_flat(self):
        with self._lock:
            self.history = [0.0] * self.width
        self._mode = "flat"

    def set_listening(self):
        self._mode = "listening"
        with self._lock:
            self.history = [0.0] * self.width

    def set_thinking(self):
        self._mode = "thinking"
        with self._lock:
            self.history = [0.0] * self.width

    def set_speaking(self, wav_path, chunk_size=256):
        wf = wave.open(wav_path, "rb")
        n_channels = wf.getnchannels()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        wf.close()

        data = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        self._amplitudes = [float(np.abs(c).mean()) if len(c) else 0.0 for c in chunks]
        self._chunk_duration = chunk_size / framerate
        self._ref = np.percentile(self._amplitudes, 95) or 1.0 if len(self._amplitudes) > 0 else 1.0
        self._play_start = time.time()
        with self._lock:
            self.history = [0.0] * self.width
        self._frame_buf = []
        self._mode = "speaking"
