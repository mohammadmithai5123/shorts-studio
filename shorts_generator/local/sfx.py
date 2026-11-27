"""Sound effects, synthesised here (so there is nothing to license): pop, whoosh, ding; plus the sounds of the memes.
One WAV track per clip."""
import wave
from typing import Dict, Sequence, Tuple

import numpy as np  # type: ignore

SR = 48000
LEVEL = 0.30                          # peak of the effects relative to full scale: heard, never louder than speech


def _pop() -> "np.ndarray":
    t = np.arange(int(0.09 * SR)) / SR
    freq = 380 + 620 * np.minimum(t / 0.05, 1.0)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 40)


def _whoosh() -> "np.ndarray":
    n = int(0.32 * SR)
    noise = np.random.default_rng(7).standard_normal(n)
    # a moving band: difference of two running means whose width shrinks (sounds like air rushing past)
    out = np.zeros(n)
    csum = np.concatenate([[0.0], np.cumsum(noise)])
    for i in range(n):
        wide = int(60 - 45 * i / n) + 4
        a, b = max(0, i - wide), max(0, i - wide // 4)
        out[i] = (csum[i + 1] - csum[b]) / (i + 1 - b) - (csum[i + 1] - csum[a]) / (i + 1 - a)
    env = np.sin(np.pi * np.arange(n) / n) ** 2
    out *= env
    return out / (np.abs(out).max() or 1.0) * 0.8


def _ding() -> "np.ndarray":
    t = np.arange(int(0.8 * SR)) / SR
    return (np.sin(2 * np.pi * 1318.5 * t) + 0.55 * np.sin(2 * np.pi * 1975.5 * t)) / 1.55 * np.exp(-t * 6) * np.minimum(t / 0.004, 1.0)


MAKERS = {"pop": _pop, "whoosh": _whoosh, "ding": _ding}


def write_track(events: Sequence[Tuple[float, str]], duration: float, path: str,
                clips: Sequence[Tuple[float, "np.ndarray"]] = ()) -> str:
    """A mono 16-bit WAV, ``duration`` long, with each effect at its time."""
    track = np.zeros(int(duration * SR) + 1)
    cache: Dict[str, np.ndarray] = {}
    for t, name in events:
        if name not in MAKERS:
            continue
        if name not in cache:
            cache[name] = MAKERS[name]()
        sound = cache[name]
        i = int(max(0.0, t) * SR)
        n = min(len(sound), len(track) - i)
        if n > 0:
            track[i:i + n] += sound[:n]
    track *= LEVEL
    for t, samples in clips:                                  # already levelled by their maker (memes.sound_of)
        i = int(max(0.0, t) * SR)
        n = min(len(samples), len(track) - i)
        if n > 0:
            track[i:i + n] += samples[:n]
    pcm = (np.clip(track, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
