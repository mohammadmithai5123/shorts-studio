"""A timed transcript for a video: the video's own captions when they exist, Whisper otherwise.

YouTube serves captions as a small text file (about 2 MB for a two-hour video, one request). For most
videos they are far better than a CPU Whisper run, which takes longer than the video itself and
struggles with mixed-language speech; they also carry word-level timing (used for karaoke captions
and precise clip edges).

Transcript shape (the contract the rest of the pipeline relies on):
    {"duration": float, "language": str, "source": "youtube-captions:auto:hi-orig" | "whisper:base",
     "segments": [{"start": s, "end": s, "text": str, "words": [{"start", "end", "text"}]?}]}
"""
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

from .. import config
from . import control
from .source import Source, SourceError

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


# --------------------------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------------------------- #
def _finish(lines: List[Dict]) -> List[Dict]:
    """Turn raw caption lines (start, hard_end, words[{start,text}]) into non-overlapping segments."""
    lines = sorted((ln for ln in lines if ln["words"]), key=lambda ln: ln["start"])
    out: List[Dict] = []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else float("inf")
        words = ln["words"]
        end = min(ln["hard_end"], nxt)                      # captions stay on screen after the speech: cut at the next line
        end = max(end, words[-1]["start"] + 0.25)           # but always show the last word...
        if nxt != float("inf"):
            end = min(end, max(nxt, words[0]["start"] + 0.1))   # ...without running into the next line
        if end - words[0]["start"] < 0.1:
            continue
        text = " ".join(w["text"] for w in words)
        if out and out[-1]["text"] == text and ln["start"] < out[-1]["end"] + 0.5:
            continue                                         # rolling captions repeat a line
        timed = []
        for j, w in enumerate(words):
            w_end = words[j + 1]["start"] if j + 1 < len(words) else end
            timed.append({"start": round(w["start"], 3), "end": round(max(w_end, w["start"] + 0.05), 3), "text": w["text"]})
        out.append({"start": round(words[0]["start"], 3), "end": round(end, 3), "text": text, "words": timed})
    return out


def parse_json3(data: Dict) -> List[Dict]:
    """YouTube 'json3' captions -> segments with per-word timing."""
    lines: List[Dict] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        t0 = ev["tStartMs"] / 1000.0
        dur = (ev.get("dDurationMs") or 0) / 1000.0
        words: List[Dict] = []
        for s in segs:
            txt = (s.get("utf8") or "").replace("\n", " ").strip()
            if txt:
                words.append({"start": t0 + (s.get("tOffsetMs") or 0) / 1000.0, "text": txt})
        if not words:
            continue
        if len(words) == 1 and " " in words[0]["text"]:      # hand-made cue: one text blob, spread the words evenly
            toks = words[0]["text"].split()
            span = max(dur, 0.5)
            words = [{"start": t0 + span * k / len(toks), "text": t} for k, t in enumerate(toks)]
        lines.append({"start": words[0]["start"], "hard_end": t0 + dur if dur else words[-1]["start"] + 1.0, "words": words})
    return _finish(lines)


_VTT_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")


def parse_vtt(text: str) -> List[Dict]:
    """WebVTT / SRT cues -> segments (words spread evenly inside each cue)."""
    lines: List[Dict] = []
    cues = re.split(r"\n\s*\n", text.replace("\r", ""))
    for cue in cues:
        rows = [r for r in cue.split("\n") if r.strip()]
        for k, row in enumerate(rows):
            m = _VTT_TIME.search(row)
            if not m:
                continue
            g = m.groups()
            start = int(g[0] or 0) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000.0
            end = int(g[4] or 0) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000.0
            body = re.sub(r"<[^>]+>", "", " ".join(rows[k + 1:]))
            toks = body.split()
            if toks and end > start:
                span = end - start
                lines.append({"start": start, "hard_end": end,
                              "words": [{"start": start + span * i / len(toks), "text": t} for i, t in enumerate(toks)]})
            break
    return _finish(lines)


# --------------------------------------------------------------------------------------------- #
# Sources of transcripts
# --------------------------------------------------------------------------------------------- #
def _usable(segments: List[Dict], duration: float) -> bool:
    if len(segments) < 5:
        return False
    spoken = sum(s["end"] - s["start"] for s in segments)
    return spoken >= 0.05 * max(duration, 1.0)


def fetch_captions(source: Source, language: Optional[str] = None) -> Optional[Dict]:
    """The video's own captions as a transcript, or None when there are none worth using."""
    track = source.caption_track(language)
    if track is None:
        return None
    try:
        proxy = config.env("YTDLP_PROXY")
        r = requests.get(track["url"], timeout=40, headers={"User-Agent": _UA},
                         proxies={"http": proxy, "https": proxy} if proxy else None)
        r.raise_for_status()
        segments = parse_json3(r.json()) if track["ext"] == "json3" else parse_vtt(r.text)
    except (requests.RequestException, ValueError):
        return None
    duration = float(source.duration or (segments[-1]["end"] if segments else 0.0))
    if not _usable(segments, duration):
        return None
    return {
        "duration": duration,
        "language": track["key"].replace("-orig", ""),
        "source": f"youtube-captions:{track['kind']}:{track['key']}",
        "segments": segments,
    }


def whisper_transcript(source: Source, cache_dir: Path, language: Optional[str]) -> Dict:
    """Fallback: transcribe the audio track with faster-whisper (the audio only, tens of MB)."""
    from ..config import whisper_model
    from .transcriber import transcribe_local

    control.report("transcribe", None, "No usable captions: fetching the audio for Whisper")
    audio = source.fetch_audio_file(cache_dir)
    try:
        result = transcribe_local(str(audio), language=language)
    finally:
        if source.is_stream:
            audio.unlink(missing_ok=True)        # it is only a working copy
    result["source"] = f"whisper:{whisper_model()}"
    return result


def get_transcript(
    get_source: Callable[[], Source],
    cache_dir: Path,
    language: Optional[str] = None,
    refresh: bool = False,
    allow_whisper: bool = True,
    duration: Optional[float] = None,
) -> Dict:
    """Cached per video. Captions first; Whisper only if there are none (or allow_whisper is off -> error).

    ``get_source`` is called only when the network is needed, so a cached transcript costs nothing.
    """
    path = cache_dir / "transcript.json"
    want = (language or "").strip().lower() or None
    if path.exists() and not refresh:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("segments") and (not want or str(cached.get("language", "")).lower().split("-")[0] == want):
                print(f"[transcript] cached ({cached.get('source')}, {len(cached['segments'])} segments)", flush=True)
                return cached
        except (OSError, ValueError):
            pass

    control.check()
    source = get_source()
    transcript: Optional[Dict] = None
    if source.is_stream:
        control.report("transcript", None, "Reading the video's captions")
        transcript = fetch_captions(source, want)
        if transcript:
            print(f"[transcript] YouTube captions: {transcript['source']}, {len(transcript['segments'])} segments", flush=True)
    if transcript is None:
        if not allow_whisper:
            raise SourceError("This video has no usable captions.")
        transcript = whisper_transcript(source, cache_dir, want)
    if not transcript.get("segments"):
        raise SourceError("No speech was found in this video, so there is nothing to rank.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    return transcript
