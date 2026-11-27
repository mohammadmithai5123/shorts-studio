"""Voice-energy signal for ranking highlights.

The transcript says what was said, not how it was said, and speech recognition on mixed-language or
noisy audio can be close to useless. How loud the speaker gets compared with the rest of the video is
language-independent, and in gaming / streaming / vlog / reaction content the best moments are the
excited ones.

The audio-only track is fetched with yt-dlp (about 30 MB per hour: never the video), reduced to one
loudness value per second, and deleted again; only that tiny series is cached.

(Reading the stream straight through ffmpeg looks tidier but YouTube throttles a continuous read to
about real time: 35+ minutes for a two-hour video, against 49 seconds with yt-dlp's chunked download.)
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import control
from .signals import PctFn, window_pct_fn

SAMPLE_RATE = 16000


def stream_energy_db(stream: Tuple[str, Dict[str, str]], duration: float = 0.0) -> Optional[np.ndarray]:
    """Per-second loudness (dBFS) of an audio/video file (or http stream), or None if it can't be read."""
    target, headers = stream
    cmd = ["ffmpeg", "-v", "error"]
    if target.startswith("http"):
        hdr = "".join(f"{k}: {v}\r\n" for k, v in (headers or {}).items())
        if hdr:
            cmd += ["-headers", hdr]
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    cmd += ["-i", target, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None

    bytes_per_second = SAMPLE_RATE * 2
    per_second = []
    carry = b""
    try:
        while True:
            control.check()
            buf = proc.stdout.read(bytes_per_second * 60)
            if not buf:
                break
            data = carry + buf
            usable = (len(data) // bytes_per_second) * bytes_per_second
            carry = data[usable:]
            if usable:
                x = np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32) / 32768.0
                per_second.extend(np.sqrt((x.reshape(-1, SAMPLE_RATE) ** 2).mean(axis=1) + 1e-12))
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    if len(per_second) < 10:
        return None
    return 20.0 * np.log10(np.asarray(per_second) + 1e-9)


def load_energy_db(get_source, cache_dir: Path, refresh: bool = False) -> Optional[np.ndarray]:
    """Cached per video: ``<cache_dir>/energy.json``. Returns None if the audio can't be fetched or read.

    ``get_source`` (a zero-argument callable returning the Source) is only called on a cache miss.
    """
    path = cache_dir / "energy.json"
    if path.exists() and not refresh:
        try:
            db = json.loads(path.read_text(encoding="utf-8"))["db"]
            if len(db) >= 10:
                print(f"[signals] voice energy cached ({len(db)} s)", flush=True)
                return np.asarray(db, dtype=np.float64)
        except (OSError, ValueError, KeyError):
            pass

    working_copy: Optional[Path] = None
    try:
        source = get_source()
        control.report("audio", None, "fetching the audio track (not the video)")
        audio = source.fetch_audio_file(cache_dir)
        working_copy = audio if source.is_stream else None
        control.report("signals", None, "measuring how loud each second is")
        db = stream_energy_db((str(audio), {}), source.duration)
    except Exception as e:  # noqa: BLE001 — the signal is a bonus: never fail the analysis over it
        print(f"[signals] voice energy unavailable: {e}", flush=True)
        return None
    finally:
        if working_copy is not None:
            working_copy.unlink(missing_ok=True)
    if db is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"db": [round(float(v), 1) for v in db]}), encoding="utf-8")
    return db


def energy_pct_fn(db: Optional[np.ndarray]) -> Optional[PctFn]:
    return window_pct_fn(db) if db is not None and len(db) >= 10 else None
