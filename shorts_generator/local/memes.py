"""Your own meme clips, sounds and stickers, dropped in at the right moments.

The files live in ``Memes Assets`` and / or ``memes``. What is known about a file (what it is for, which seconds to play,
how to play it) is in ``assets/meme_catalog.json``, looked up by file name. A file that is not listed there still works:
its mood comes from the folder or the name ("faah_shock.mp3").

How a meme is played ("how"):

  cut     the video waits, the meme fills the screen with its own sound, then the video carries on
  over    over the video while it keeps playing: a sound, a sticker, or a green-screen clip (keyed out)
  end     closes the clip, after the last word
  later   in place of a long wait that was cut out

Which one is used: the AI picks by id from ``menu()`` when it plans the edit; without a pick, laughter / applause in the
captions and loud moments choose by mood. Nothing is downloaded: only what is in the folders is used.
"""
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from ..config import ROOT, SHORT_SECONDS

MOODS = ("laugh", "shock", "fail", "sad", "hype", "awkward", "angry", "win")
MOOD_WORDS = {
    "laugh": ("laugh", "laughing", "funny", "lol", "haha", "hansi", "hasi", "comedy"),
    "shock": ("shock", "wow", "omg", "faah", "fah", "surprise", "wtf", "what"),
    "fail": ("fail", "bruh", "oops", "noob", "galti"),
    "sad": ("sad", "cry", "rona", "emotional", "dukh"),
    "hype": ("hype", "fire", "goat", "applause", "clap", "sigma", "entry"),
    "awkward": ("awkward", "silence", "cricket", "huh", "confused", "kya"),
    "angry": ("angry", "gussa", "rage"),
    "win": ("win", "jeet", "victory", "money", "paisa", "party"),
}
HOWS = ("cut", "over", "end", "later")
INSERTED = ("cut", "end", "later")          # the video stands still while these play
VIDEO = (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".gif")
AUDIO = (".mp3", ".wav", ".m4a", ".ogg", ".aac", ".flac")
IMAGE = (".png", ".jpg", ".jpeg", ".webp")
MAX_BYTES = 60 * 1024 * 1024
VIDEO_SECONDS, SOUND_SECONDS, IMAGE_SECONDS = 2.5, 3.0, 1.4      # for files the catalog says nothing about
LONGEST = 3.5                               # no meme stays longer than this
SAMPLE_RATE = 48000
CATALOG = ROOT / "assets" / "meme_catalog.json"
USER_FOLDER = "Memes Assets"

README = """Apni meme clips yahan daalo (video, sound ya photo), ya "Memes Assets" folder mein.
Editor khud sahi moment pe sahi clip lagayega.

Har mood ka apna folder hai:
  laugh    hansi / funny
  shock    "faah", OMG, surprise
  fail     bruh, oops
  sad      dukh / emotional
  hype     applause, fire, sigma entry
  awkward  silence, cricket, "huh?"
  angry    gussa
  win      jeet, paisa, party

Chalne wali files:  video: mp4 mov webm mkv gif   sound: mp3 wav m4a ogg   photo: png jpg webp
Green screen wali video ka green apne aap hat jata hai.
File ka naam saaf rakho (jaise "kya-bola-tune.mp4"): AI naam padh kar bhi chunta hai.
Chhoti clips (1-3 second) sabse achhi lagti hain. Sirf wahi files daalo jinhe use karne ka haq aapke paas ho
(YouTube copyright claim kar sakta hai).
"""


@dataclass(frozen=True)
class Meme:
    path: str
    mood: str                      # one of MOODS or "any"
    kind: str                      # "video" | "sound" | "image"
    id: str = ""                   # what the AI answers with
    use: str = ""                  # when to use it
    how: str = "over"              # HOWS
    start: float = 0.0             # the part of the file that is played
    end: float = 0.0               # 0 = from ``start``, as long as the kind allows
    auto: bool = True              # False: never picked by itself

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def inserted(self) -> bool:
        return self.how in INSERTED


@dataclass
class Cue:
    t: float                       # output seconds
    dur: float
    meme: Meme
    mood: str
    at: float = 0.0                # seconds in the fetched clip (set by the edit plan)


# --------------------------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------------------------- #
def folders() -> List[Path]:
    custom = os.environ.get("LOCAL_MEME_DIR", "").strip()
    return [Path(custom)] if custom else [ROOT / "memes", ROOT / USER_FOLDER]


def folder() -> Path:
    """The one to show the person."""
    options = folders()
    return next((f for f in reversed(options) if f.is_dir()), options[0])


def ensure_folder() -> Path:
    """Create the folder with its mood sub-folders and a note on how to fill it; returns the folder to show."""
    base = folders()[0]
    for mood in MOODS:
        (base / mood).mkdir(parents=True, exist_ok=True)
    note = base / "README.txt"
    if not note.exists():
        note.write_text(README, encoding="utf-8")
    return folder()


def catalog() -> Dict[str, Dict]:
    """{file name in lower case: what is known about it}. Empty when the list is missing or damaged."""
    try:
        data = json.loads(CATALOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k).lower(): v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")}


def _mood_from_name(name: str) -> str:
    words = set(re.split(r"[^a-z]+", name.lower()))
    for mood, keys in MOOD_WORDS.items():
        if mood in words or words & set(keys):
            return mood
    return "any"


@lru_cache(maxsize=512)
def _green_video(path: str, stamp: float) -> bool:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-vf", "scale=64:64", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                          capture_output=True)
    if len(proc.stdout) < 64 * 64 * 3:
        return False
    return _is_green_screen(np.frombuffer(proc.stdout[:64 * 64 * 3], dtype=np.uint8).reshape(64, 64, 3))


def scan() -> List[Meme]:
    known = catalog()
    found: List[Meme] = []
    taken: set = set()
    for base in folders():
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            ext = path.suffix.lower()
            kind = "video" if ext in VIDEO else "sound" if ext in AUDIO else "image" if ext in IMAGE else ""
            if not kind or not path.is_file() or path.stat().st_size > MAX_BYTES or len(path.relative_to(base).parts) > 3:
                continue
            parts = path.relative_to(base).parts
            parent = parts[0].lower() if len(parts) > 1 else ""
            entry = known.get(path.name.lower(), {})
            mood = str(entry.get("mood") or "") or (parent if parent in MOODS else _mood_from_name(path.stem))
            how = str(entry.get("how") or "")
            if how not in HOWS:
                how = "cut" if kind == "video" and not _green_video(str(path), path.stat().st_mtime) else "over"
            ident = str(entry.get("id") or "") or re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")[:32] or "meme"
            while ident in taken:
                ident += "_2"
            taken.add(ident)
            use = str(entry.get("use") or "") or (re.sub(r"[-_]+", " ", path.stem).strip() + (f" ({mood})" if mood != "any" else ""))
            found.append(Meme(str(path), mood, kind, id=ident, use=use[:110], how=how, start=float(entry.get("start") or 0.0),
                              end=float(entry.get("end") or 0.0), auto=bool(entry.get("auto", True))))
    return found


def summary() -> Dict:
    memes = scan()
    usable = [m for m in memes if m.auto]
    return {"folder": str(folder()), "total": len(memes), "usable": len(usable), "off": [m.name for m in memes if not m.auto],
            "by_mood": {m: sum(1 for x in usable if x.mood == m) for m in MOODS + ("any",)}}


def menu(library: Sequence[Meme]) -> List[Tuple[str, str]]:
    """(id, when to use it) of everything the AI may pick, in a steady order (the AI's answers are saved by question)."""
    return sorted((m.id, m.use) for m in library if m.auto and m.id)


def mood_of_emoji(emoji: str) -> str:
    table = {"😂": "laugh", "🤣": "laugh", "🍕": "laugh", "😮": "shock", "😱": "shock", "🚓": "shock", "✈": "shock", "💀": "fail", "🤦": "fail",
             "🥺": "sad", "😢": "sad", "😭": "sad", "🔥": "hype", "👏": "hype", "💥": "hype", "🚗": "hype", "🎵": "hype", "❤": "hype",
             "🧠": "awkward", "🤔": "awkward", "😡": "angry", "💰": "win", "👑": "win", "🎮": "win", "💯": "win"}
    return next((mood for mark, mood in table.items() if mark in str(emoji or "")), "")


# --------------------------------------------------------------------------------------------- #
# Choosing
# --------------------------------------------------------------------------------------------- #
def pick(library: Sequence[Meme], mood: str, used: set, seed: str) -> Optional[Meme]:
    """A meme of that mood not used yet in this clip (one without a mood as the fallback). Same input, same pick."""
    for wanted in (mood, "any"):
        options = [m for m in library if m.mood == wanted and m.path not in used]
        if options:
            return options[int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(options)]
    return None


def probe_seconds(path: str) -> float:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(proc.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0


def play_seconds(meme: Meme) -> float:
    """How long it plays; 0 when the file cannot be read."""
    if meme.kind == "image":
        return IMAGE_SECONDS
    length = probe_seconds(meme.path)
    if length <= 0:
        return 0.0
    limit = (meme.end - meme.start) if meme.end > meme.start else (VIDEO_SECONDS if meme.kind == "video" else SOUND_SECONDS)
    return max(0.0, min(limit, length - meme.start, LONGEST))


def plan(moments: Sequence[Tuple], library: Sequence[Meme], duration: float, seed: str = "") -> List[Cue]:
    """``moments``: (time, mood, priority; lower is better[, meme id]). A few well spaced memes, never over the opening.

    A moment that names a meme gets that one. Otherwise the mood chooses; only a good reason (priority up to 0.5: the AI,
    or laughter / applause in the captions) may stop the video, weaker ones get a sound or a sticker at most.
    """
    usable = [m for m in library if m.auto]
    if not usable:
        return []
    most = max(2, min(4, int(duration // 8)))                    # sounds included
    most_cuts = max(1, min(3, int(duration // 12)))              # the ones that stop the video
    cut_budget = max(4.0, min(8.0, duration * 0.2))              # the memes must not become the clip
    cut_budget = min(cut_budget, max(0.0, SHORT_SECONDS - duration))      # stops never push a Short over a minute
    cues: List[Cue] = []
    used: set = set()
    cuts, cut_seconds = 0, 0.0
    for moment in sorted(moments, key=lambda m: (m[2], m[0])):
        if len(cues) >= most:
            break
        t, mood, priority = float(moment[0]), moment[1], moment[2]
        wanted = moment[3] if len(moment) > 3 else ""
        if wanted:
            meme = next((m for m in usable if m.id == wanted and m.path not in used), None)
        elif mood == "later":
            meme = next((m for m in usable if m.how == "later" and m.path not in used), None)
        else:
            pool = [m for m in usable if m.how in ("cut", "over") and (priority <= 0.5 or not m.inserted)]
            if cuts >= most_cuts:
                pool = [m for m in pool if not m.inserted]
            meme = pick(pool, mood, used, f"{seed}|{t:.1f}|{mood}")
        if meme is None:
            continue
        if meme.how == "end":
            if any(c.meme.how == "end" for c in cues):
                continue
            t = duration
        elif not (3.2 <= t <= duration - 2.0):
            continue
        if any(abs(t - c.t) < 6.0 for c in cues):
            continue
        length = play_seconds(meme)
        if length < 0.4:
            continue
        if meme.inserted:
            if cuts >= most_cuts or cut_seconds + length > cut_budget:
                continue
            cuts, cut_seconds = cuts + 1, cut_seconds + length
        else:
            length = min(length, max(0.6, duration - t))
        used.add(meme.path)
        cues.append(Cue(t=t, dur=length, meme=meme, mood=mood or meme.mood))
    return sorted(cues, key=lambda c: c.t)


# --------------------------------------------------------------------------------------------- #
# Pixels and sound
# --------------------------------------------------------------------------------------------- #
def _key_green(frame: "np.ndarray") -> "np.ndarray":
    """BGRA with the green screen transparent."""
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))
    greenness = g - np.maximum(r, b)
    alpha = np.clip((60 - greenness) * 255 // 40, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    spill = greenness > 0
    out[:, :, 1] = np.where(spill, np.maximum(r, b), g).astype(np.uint8)          # no green fringe
    out[:, :, 3] = cv2.GaussianBlur(alpha, (3, 3), 0)
    return out


def _is_green_screen(frame: "np.ndarray") -> bool:
    b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))
    return float(((g - np.maximum(r, b) > 50) & (g > 90)).mean()) > 0.25


def _framed(frame: "np.ndarray") -> "np.ndarray":
    """An opaque picture in a white frame, so it reads as a meme card and not as a glitch."""
    border = max(4, frame.shape[1] // 60)
    return cv2.copyMakeBorder(cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA), border, border, border, border, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255, 255))


def _decode(path: str, start: float, seconds: float, fps: float, width: int, height: int) -> List["np.ndarray"]:
    """BGR frames of [start, start + seconds], scaled to fit inside width x height."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
    try:
        w, h = (int(v) for v in probe.stdout.strip().splitlines()[0].split(",")[:2])
    except (ValueError, IndexError):
        return []
    k = min(width / w, height / h)
    w2, h2 = max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2)
    proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path, "-t", f"{seconds:.3f}", "-vf",
                           f"fps={fps:.3f},scale={w2}:{h2}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], capture_output=True)
    size = w2 * h2 * 3
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    return [raw[i * size:(i + 1) * size].reshape(h2, w2, 3) for i in range(len(raw) // size)]


def load_frames(cue: Cue, fps: float, width: int, max_height: int) -> List["np.ndarray"]:
    """BGRA pictures of a meme shown over the video, sized for the output (empty for sounds or unreadable files)."""
    meme = cue.meme
    if meme.kind == "sound":
        return []
    if meme.kind == "image":
        data = np.fromfile(meme.path, dtype=np.uint8)                    # np.fromfile: paths with any letters work
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
        if img is None:
            return []
        k = min(width * 0.8 / img.shape[1], max_height / img.shape[0])
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            return [_framed(img)]
        return [img]
    frames = _decode(meme.path, meme.start, cue.dur, fps, width, max_height)
    if not frames:
        return []
    if _is_green_screen(frames[0]):
        return [_key_green(f) for f in frames]
    return [_framed(f) for f in frames]


def _without_black_bars(frames: List["np.ndarray"]) -> List["np.ndarray"]:
    """Cut away black bars that are baked into the file, so the picture is as big as it can be."""
    sample = np.max(np.stack([frames[i] for i in sorted({0, len(frames) // 2, len(frames) - 1})]), axis=0).max(axis=2)
    cols, rows = np.where(sample.max(axis=0) > 28)[0], np.where(sample.max(axis=1) > 28)[0]
    if not len(cols) or not len(rows) or float(sample.mean()) < 20:      # a mostly black picture (title cards) stays whole
        return frames
    h, w = sample.shape
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
    if (x1 - x0) < w * 0.25 or (y1 - y0) < h * 0.25 or ((x1 - x0) > w * 0.95 and (y1 - y0) > h * 0.95):
        return frames
    return [f[y0:y1, x0:x1] for f in frames]


def load_full_frames(cue: Cue, fps: float, out_w: int, out_h: int) -> List["np.ndarray"]:
    """BGR pictures of a meme that takes over the screen (empty when the file cannot be read)."""
    if cue.meme.kind != "video":
        return []
    frames = _decode(cue.meme.path, cue.meme.start, cue.dur, fps, min(out_w, 1080), min(out_h, 1080))
    return _without_black_bars(frames) if frames else []


def full_frame(picture: "np.ndarray", out_w: int, out_h: int, progress: float) -> "np.ndarray":
    """The meme in the middle of the screen, a blurred dim copy of it behind, pushing in a little as it plays."""
    h, w = picture.shape[:2]
    cover = max(out_w / w, out_h / h)
    cw, ch = max(1, int(out_w / cover)), max(1, int(out_h / cover))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    small = cv2.resize(picture[y0:y0 + ch, x0:x0 + cw], (36, 64), interpolation=cv2.INTER_AREA)
    img = cv2.convertScaleAbs(cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_CUBIC), alpha=0.42)
    k = min(out_w / w, out_h / h) * (1.0 + 0.05 * min(1.0, max(0.0, progress)))
    fw, fh = max(2, int(round(w * k))), max(2, int(round(h * k)))
    front = cv2.resize(picture, (fw, fh), interpolation=cv2.INTER_LINEAR)
    fx, fy = max(0, (fw - out_w) // 2), max(0, (fh - out_h) // 2)
    front = front[fy:fy + out_h, fx:fx + out_w]
    x, y = (out_w - front.shape[1]) // 2, (out_h - front.shape[0]) // 2
    img[y:y + front.shape[0], x:x + front.shape[1]] = front
    return img


def sound_of(cue: Cue) -> Optional["np.ndarray"]:
    """The meme's sound, mono float at 48 kHz, levelled so that a loud file does not blast and a quiet one is heard."""
    if cue.meme.kind == "image":
        return None
    proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{cue.meme.start:.3f}", "-i", cue.meme.path, "-vn", "-t", f"{cue.dur:.3f}",
                           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"], capture_output=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64) / 32768.0
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    if peak < 0.01:
        return None
    pcm = pcm * ((0.7 if cue.meme.inserted else 0.5) / peak)       # alone it may be louder than on top of someone talking
    fade = min(len(pcm) // 4, int(0.08 * SAMPLE_RATE))
    if fade:
        pcm[:fade // 4] *= np.linspace(0.0, 1.0, fade // 4)          # a trimmed file must not start with a click
        pcm[-fade:] *= np.linspace(1.0, 0.0, fade)
    return pcm
