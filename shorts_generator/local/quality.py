"""Check a rendered clip before anyone sees it (and before it can be uploaded): a broken clip must never reach a channel."""
import json
import os
import subprocess
from typing import List, Optional, Tuple


def probe(path: str) -> dict:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration,size", "-of", "json", path],
                          capture_output=True, text=True)
    try:
        return json.loads(proc.stdout or "{}")
    except ValueError:
        return {}


def _brightness(path: str, at: float) -> Optional[float]:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", path, "-frames:v", "1", "-vf", "scale=64:64,format=gray",
                           "-f", "rawvideo", "-"], capture_output=True)
    data = proc.stdout
    return sum(data) / len(data) if data else None


def verify(path: str, size: Tuple[int, int], duration: float, need_audio: bool = True) -> List[str]:
    """Problems with the file, empty when it is good: size, length, sound, and not an all-black picture."""
    if not os.path.isfile(path) or os.path.getsize(path) < 2_000:
        return ["the file is missing or almost empty"]
    info = probe(path)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return ["there is no picture in the file"]
    problems = []
    if (video.get("width"), video.get("height")) != tuple(size):
        problems.append(f"wrong picture size {video.get('width')}x{video.get('height')} instead of {size[0]}x{size[1]}")
    length = float((info.get("format") or {}).get("duration") or 0)
    if abs(length - duration) > max(1.2, 0.08 * duration):
        problems.append(f"wrong length: {length:.1f}s instead of about {duration:.1f}s")
    if need_audio and not any(s.get("codec_type") == "audio" for s in streams):
        problems.append("the sound is missing")
    samples = [_brightness(path, length * f) for f in (0.2, 0.5, 0.8)] if length > 0 else []
    if samples and all(b is not None and b < 4 for b in samples):
        problems.append("the picture is completely black")
    return problems
