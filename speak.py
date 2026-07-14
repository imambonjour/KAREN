#!/usr/bin/env python3
"""
Karen CLI waveform visualizer (v2 - using plotext).
Plays a WAV file and renders a live terminal waveform synced to its amplitude.

Install dependency first:
    pip install plotext numpy --break-system-packages

Usage:
    python3 karen_wave.py output.wav
    python3 karen_wave.py output.wav --player ffplay   # if aplay not available
"""

import sys
import wave
import time
import threading
import subprocess
import argparse
import numpy as np
import plotext as plt


def get_amplitudes(filename, chunk_size=1024):
    """Read a WAV file and return per-chunk amplitude envelope + chunk duration (s)."""
    wf = wave.open(filename, "rb")
    n_channels = wf.getnchannels()
    framerate = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    wf.close()

    data = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    amplitudes = [float(np.abs(c).mean()) if len(c) else 0.0 for c in chunks]
    chunk_duration = chunk_size / framerate
    return amplitudes, chunk_duration


def draw_wave(history):
    plt.clf()
    plt.plotsize(None, None)   # auto-fit to current terminal size
    plt.theme("pro")           # dark background theme, works well with green line
    plt.ylim(-1.1, 1.1)
    plt.plot(history, marker="braille", color="green")
    plt.axes_color("black")
    plt.canvas_color("black")
    plt.frame(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def render_flat(width=80):
    draw_wave([0.0] * width)


def render_talking(wav_file):
    amplitudes, chunk_duration = get_amplitudes(wav_file, chunk_size=256)
    if not amplitudes:
        render_flat()
        return

    ref = np.percentile(amplitudes, 95) or 1.0

    width = 80
    history = [0.0] * width
    start = time.monotonic()

    samples_per_frame = 3
    buf = []

    for i, amp in enumerate(amplitudes):
        level = amp / ref
        level = max(0.0, min(1.0, level))
        signed = level * (1 if (i % 2 == 0) else -1)
        buf.append(signed)

        if len(buf) >= samples_per_frame:
            history.extend(buf)
            history = history[-width:]
            buf = []
            draw_wave(history)

        target_time = start + (i + 1) * chunk_duration
        sleep_for = target_time - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


def play_audio(wav_file, player):
    try:
        if player == "aplay":
            result = subprocess.run(["aplay", wav_file], capture_output=True, text=True)
        elif player == "ffplay":
            result = subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", wav_file],
                capture_output=True, text=True,
            )
        elif player == "paplay":
            result = subprocess.run(["paplay", wav_file], capture_output=True, text=True)
        else:
            result = subprocess.run([player, wav_file], capture_output=True, text=True)

        if result.returncode != 0:
            sys.stderr.write(f"\n[audio error] {player} exited with code {result.returncode}\n")
            if result.stderr:
                sys.stderr.write(result.stderr + "\n")
    except FileNotFoundError:
        sys.stderr.write(f"\n[warn] player '{player}' not found on this system\n")


def play_and_visualize(wav_file, player="aplay"):
    player_thread = threading.Thread(target=play_audio, args=(wav_file, player))
    player_thread.start()
    render_talking(wav_file)
    player_thread.join()
    render_flat()


def main():
    parser = argparse.ArgumentParser(description="Karen waveform CLI visualizer")
    parser.add_argument("wav_file", help="Path to WAV file to play + visualize")
    parser.add_argument(
        "--player",
        default="aplay",
        help="Command used to play audio (aplay, paplay, ffplay). Default: aplay",
    )
    args = parser.parse_args()

    render_flat()
    time.sleep(1)
    play_and_visualize(args.wav_file, player=args.player)


if __name__ == "__main__":
    main()
