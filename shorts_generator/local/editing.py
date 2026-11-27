"""Automatic editing: decide WHAT happens WHEN. The renderer only draws it.

Two timelines: *source* (seconds from the start of the clip as fetched) and *output* (after the jump cuts). Cut points
sit on the output frame grid, so sound and picture can never drift apart however many pauses are removed.

Effects (each can be switched on its own; "viral" turns all on, "clean" the calm ones):
  jumpcut   pauses are cut out (found in the audio; spoken words are protected)
  zoom      punch-in zooms: the framing alternates at sentence starts, and jumps in on the loudest moments
  memes     your own clips / sounds / stickers (memes.py), at the moments that call for them; a full-screen meme makes
            the video wait for it (an "insert" in the timeline), a sound plays over it
  popups    emoji / short text popping up (off in every preset: they look cheap; switch it on if you like them)
  sfx       soft synthesised "pop" / "whoosh" / "ding" sounds with them
  hook      the clip's headline, big, at the start
  cta       "Follow for more" at the end (off in every preset: the channel's name on the clip is the branding)
  progress  a thin bar showing how much of the clip is left (keeps people watching)
  color     a little more saturation and contrast
  captions  animated word-by-word captions
"""
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore

from . import memes as meme_lib

PRESET_NAMES = ("off", "clean", "viral")
EFFECTS = ("jumpcut", "zoom", "memes", "popups", "sfx", "hook", "cta", "progress", "color", "captions")
PRESETS: Dict[str, Dict[str, bool]] = {
    "off": {k: False for k in EFFECTS},
    "clean": {"jumpcut": True, "zoom": True, "memes": False, "popups": False, "sfx": False, "hook": True, "cta": False, "progress": True, "color": False, "captions": True},
    "viral": {k: k not in ("popups", "cta") for k in EFFECTS},      # no follow / subscribe card unless asked for
}
HOP = 0.02                           # loudness window in seconds
MIN_PAUSE = 0.45                     # a pause shorter than this stays: speech needs to breathe
PAD = 0.12                           # silence kept on each side of a cut
MAX_REMOVED = 0.35                   # never cut more than this share of a clip

Word = Tuple[float, float, str]


def effects_for(preset: str, overrides: Optional[Dict[str, bool]] = None) -> Dict[str, bool]:
    effects = dict(PRESETS.get(preset, PRESETS["off"]))
    for k, v in (overrides or {}).items():
        if k in effects and v is not None:
            effects[k] = bool(v)
    return effects


# --------------------------------------------------------------------------------------------- #
# Sound
# --------------------------------------------------------------------------------------------- #
def audio_levels(path: str) -> Optional["np.ndarray"]:
    """Loudness (dB) every 20 ms, or None when there is no usable audio."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    n = int(HOP * 16000)
    if pcm.size < n * 10:
        return None
    frames = pcm[: pcm.size // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return 20 * np.log10(rms / 32768.0 + 1e-9)


def spoken_span(word: Word) -> Tuple[float, float]:
    """When the word is actually being said. Caption "ends" are the next word's start, so they cover the pause after it."""
    start, end, text = word
    return start, min(end, start + max(0.18, 0.1 + 0.065 * len(text)))


def find_pauses(db: Optional["np.ndarray"], words: Sequence[Word], duration: float) -> List[Tuple[float, float]]:
    """Quiet stretches worth cutting. Empty when the sound never really goes quiet (music, a loud game, applause)."""
    if db is None or len(db) < 50:
        return []
    loud, floor = float(np.percentile(db, 85)), float(np.percentile(db, 5))
    if loud - floor < 12:
        return []
    quiet = db < max(floor + 6.0, loud - 24.0)
    for w in words:
        a, b = spoken_span(w)
        quiet[max(0, int((a - 0.04) / HOP)): int((b + 0.04) / HOP) + 1] = False
    out: List[Tuple[float, float]] = []
    i, n = 0, len(quiet)
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        a, b = i * HOP, min(j * HOP, duration)
        if b - a >= MIN_PAUSE:
            a2, b2 = a + (PAD if a > 0.05 else 0.0), b - (PAD if b < duration - 0.05 else 0.0)
            if b2 - a2 >= 0.2:
                out.append((a2, b2))
        i = j
    return out


def keep_ranges(pauses: Sequence[Tuple[float, float]], duration: float, fps: float, max_removed: float = MAX_REMOVED) -> List[Tuple[float, float]]:
    """The parts of the clip that stay, on the frame grid. Longest pauses go first, never more than ``max_removed``."""
    budget = duration * max_removed
    chosen: List[Tuple[float, float]] = []
    for a, b in sorted(pauses, key=lambda p: p[0] - p[1]):
        if b - a <= budget:
            chosen.append((a, b))
            budget -= b - a
    grid = lambda t: round(t * fps) / fps
    keep: List[Tuple[float, float]] = []
    cursor = 0.0
    for a, b in sorted(chosen):
        a, b = grid(a), grid(b)
        if b <= a:
            continue
        if a - cursor >= 0.2:
            keep.append((cursor, a))
        elif keep:
            keep[-1] = (keep[-1][0], keep[-1][1])        # a stub between two pauses goes with them
        cursor = b
    end = grid(duration)
    if end - cursor >= 0.2 or not keep:
        keep.append((cursor, max(end, cursor + 1 / fps)))
    return keep


class Timeline:
    """Maps source times to output times through the kept ranges.

    ``inserts``: (source time, seconds) where the video waits while something else fills the screen (a meme).
    """

    def __init__(self, keep: Sequence[Tuple[float, float]], inserts: Sequence[Tuple[float, float]] = ()):
        self.keep = [(float(a), float(b)) for a, b in keep]
        self.inserts = sorted((float(t), float(d)) for t, d in inserts)
        self.offsets: List[float] = []                        # where each kept range starts, before any insert
        total = 0.0
        for a, b in self.keep:
            self.offsets.append(total)
            total += b - a
        self.base_duration = total
        self.duration = total + sum(d for _, d in self.inserts)

    def kept(self, t: float) -> bool:
        return any(a - 1e-6 <= t < b - 1e-6 for a, b in self.keep)

    def _base(self, t: float) -> float:
        for (a, b), off in zip(self.keep, self.offsets):
            if t < a:
                return off
            if t <= b:
                return off + (t - a)
        return self.base_duration

    def to_out(self, t: float, end: bool = False) -> float:
        """Output time of source time ``t``. ``end``: of something that stops at ``t``, so an insert right there comes after."""
        edge = -1e-6 if end else 1e-6
        return self._base(t) + sum(d for at, d in self.inserts if at < t + edge)

    def to_source(self, out: float) -> float:
        """The inverse, for a timeline without inserts."""
        for (a, b), off in zip(self.keep, self.offsets):
            if out <= off + (b - a):
                return a + max(0.0, out - off)
        return self.keep[-1][1]

    def joins(self) -> List[float]:
        """Output times where the picture jumps: after a cut, and after an insert."""
        cuts = [self.to_out(a) for a, _ in self.keep[1:]]
        back = [self.to_out(at, end=True) + d for at, d in self.inserts]
        return sorted(set(round(t, 4) for t in cuts + back if t < self.duration - 1e-3))

    def pieces(self) -> List[Tuple[str, float, float]]:
        """The output in order: ("video", source from, source to) and ("wait", seconds, 0)."""
        out: List[Tuple[str, float, float]] = []
        waiting = list(self.inserts)
        for a, b in self.keep:
            while waiting and waiting[0][0] <= a + 1e-6:
                out.append(("wait", waiting.pop(0)[1], 0.0))
            while waiting and waiting[0][0] < b - 1e-6:
                at, d = waiting.pop(0)
                out += [("video", a, at), ("wait", d, 0.0)]
                a = at
            out.append(("video", a, b))
        out += [("wait", d, 0.0) for _, d in waiting]
        return out


def remap_words(words: Sequence[Word], tl: Timeline) -> List[Word]:
    out: List[Word] = []
    for a, b, text in words:
        for at, _ in tl.inserts:                              # a word's "end" is the next word's start: stop it at the wait
            if a < at < b:
                b = at
        s, e = tl.to_out(a), tl.to_out(b, end=True)
        if e - s < 0.05:
            e = min(tl.duration, s + 0.15)
        if e > s:
            out.append((s, e, text))
    return out


# --------------------------------------------------------------------------------------------- #
# Zooms, popups, sounds
# --------------------------------------------------------------------------------------------- #
def emphasis_peaks(db: Optional["np.ndarray"], tl: Timeline, spacing: float = 3.5, limit: int = 8) -> List[float]:
    """Output times of the loudest reactions (shouts, laughs), at least ``spacing`` apart."""
    if db is None:
        return []
    win = int(0.4 / HOP)
    smooth = np.convolve(db, np.ones(win) / win, mode="same")
    if float(np.percentile(smooth, 95) - np.percentile(smooth, 50)) < 3:
        return []
    threshold = float(np.percentile(smooth, 90))
    peaks: List[float] = []
    for i in np.argsort(smooth)[::-1]:
        if smooth[i] < threshold or len(peaks) >= limit:
            break
        t = i * HOP
        if not tl.kept(t):
            continue
        out = tl.to_out(t)
        if 2.0 <= out <= tl.duration - 1.5 and all(abs(out - p) >= spacing for p in peaks):
            peaks.append(out)
    return sorted(peaks)


def sentence_starts(words: Sequence[Word]) -> List[float]:
    out: List[float] = []
    prev_end, prev_text = None, ""
    for a, b, text in words:
        if prev_end is not None and (a - prev_end > 0.35 or prev_text.rstrip().endswith((".", "?", "!", "।"))):
            out.append(a)
        prev_end, prev_text = spoken_span((a, b, text))[1], text
    return out


def zoom_plan(tl: Timeline, words: Sequence[Word], peaks: Sequence[float], strong: bool) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """(steps [(from time, level)], punches [(start, end, level)]).

    Steps are instant "jump" zooms alternating at cuts and sentence starts (what an editor does to hide a cut and keep
    the eye busy); punches ease in on the loudest moments.
    """
    wide, close = 1.0, 1.12 if strong else 1.07
    marks = sorted(set(round(t, 3) for t in list(tl.joins()) + sentence_starts(words)))
    steps: List[Tuple[float, float]] = [(0.0, wide)]
    level = wide
    for t in marks:
        if t - steps[-1][0] >= 2.6 and t < tl.duration - 1.0:
            level = close if level == wide else wide
            steps.append((t, level))
    if len(steps) == 1:                                  # one long sentence: still change the framing every few seconds
        t = 4.0
        while t < tl.duration - 2.0:
            level = close if level == wide else wide
            steps.append((t, level))
            t += 4.0
    punches = [(max(0.0, p - 0.15), min(tl.duration, p + 1.0), 1.22) for p in peaks[:6]] if strong else []
    return steps, punches


KEYWORD_EMOJI: List[Tuple[str, str]] = [
    (r"\b(money|paisa|paise|rupee|rupay|crore|lakh|dollar|rich)\b|पैस|रुपय|करोड़|लाख", "💰"),
    (r"\b(laugh|funny|joke|haha|hasna|hansi|mazaa?k)\w*|हंस|हँस|मज़ाक|मजाक", "😂"),
    (r"\b(love|pyaar|pyar|dil|crush|girlfriend|boyfriend|shaadi|marry)\w*|प्यार|दिल|शादी", "❤️"),
    (r"\b(fire|aag|lit|hot|op)\b|आग", "🔥"),
    (r"\b(wow|omg|unbelievable|crazy|pagal|kamaal|insane)\w*|कमाल|पागल", "😮"),
    (r"\b(darr?\b|scared|scary|bhoot|ghost|horror)\w*|डर|भूत", "😱"),
    (r"\b(sad\b|cry|rona|emotional|dukh)\w*|रोना|रोने|दुख", "🥺"),
    (r"\b(police|jail|arrest|chor)\b|पुलिस|जेल|चोर", "🚓"),
    (r"\b(car|gaadi|gadi|bike|drive|race)\w*|गाड़ी|कार", "🚗"),
    (r"\b(plane|flight|jahaa?z|fly)\w*|जहाज", "✈️"),
    (r"\b(gun|shoot|goli|fight|maar|punch|attack)\w*|गोली|मार", "💥"),
    (r"\b(game|gaming|level|mission|win|won|jeet)\w*|गेम|जीत", "🎮"),
    (r"\b(food|khaa?na|eat|pizza|biryani|chai)\b|खाना|चाय", "🍕"),
    (r"\b(song|gaana|music|dance|naa?ch|sing)\w*|गाना|डांस|नाच", "🎵"),
    (r"\b(idea|think|soch|mind|brain|dimaa?g)\w*|दिमाग|सोच", "🧠"),
    (r"\b(boss|king|legend|best|number one)\b|राजा|बादशाह", "👑"),
]


@dataclass
class Popup:
    t: float
    dur: float
    emoji: str = ""
    text: str = ""
    side: int = 0                    # 0 left, 1 right (alternates)


def popups_from(hints: Optional[List[Dict]], words_src: Sequence[Word], peaks: Sequence[float], tl: Timeline,
                reactions: Sequence[Tuple[float, str]] = ()) -> List[Popup]:
    """AI-chosen popups first (their times are source times), then the captions' own laughter / applause notes, keyword
    matches and reactions to loud moments."""
    wanted: List[Tuple[float, str, str, int]] = []                        # (output time, emoji, text, priority)
    for h in hints or []:
        try:
            at = float(h.get("at"))
        except (TypeError, ValueError):
            continue
        emoji, text = str(h.get("emoji") or "").strip()[:8], str(h.get("text") or "").strip()[:24]
        if emoji or text:
            wanted.append((tl.to_out(at), emoji, text, 0))
    for t, kind in reactions:
        wanted.append((tl.to_out(t), "😂" if kind == "laugh" else "👏", "", 0.5))
    for a, _, token in words_src:
        low = token.lower()
        for pattern, emoji in KEYWORD_EMOJI:
            if re.search(pattern, low):
                wanted.append((tl.to_out(a), emoji, "", 1))
                break
    for k, p in enumerate(peaks):
        wanted.append((p, ("😂", "🔥", "😮", "💯")[k % 4], "", 2))

    most = max(2, min(8, int(tl.duration // 5)))
    gap = 2.4
    chosen: List[Tuple[float, str, str]] = []
    for t, emoji, text, _ in sorted(wanted, key=lambda w: (w[3], w[0])):
        if len(chosen) >= most:
            break
        if 2.8 <= t <= tl.duration - 2.0 and all(abs(t - c[0]) >= gap for c in chosen):   # not over the hook or the end card
            chosen.append((t, emoji, text))
    return [Popup(t=t, dur=1.4, emoji=e, text=x, side=i % 2) for i, (t, e, x) in enumerate(sorted(chosen))]


EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
LONG_WAIT = 3.0                      # this much cut out in one go deserves "a few moments later"


def meme_moments(hints: Optional[List[Dict]], words_src: Sequence[Word], peaks: Sequence[float], tl: Timeline,
                 reactions: Sequence[Tuple[float, str]] = (), picks: Optional[List[Dict]] = None) -> List[Tuple]:
    """Moments that call for a meme, as (output time, mood, priority[, meme id]).

    When the AI named memes (``picks``: {"at", "id"}) only those are used: it read what is said. Otherwise its popup
    moods (``hints``, only when it was not asked about memes), laughter / applause in the captions, long cut-out waits,
    telling words and loud reactions.
    """
    named: List[Tuple] = []
    for p in picks or []:
        try:
            at = float(p.get("at"))
        except (TypeError, ValueError, AttributeError):
            continue
        if p.get("id"):
            named.append((tl.to_out(at, end=True), "", 0.0, str(p["id"])))
    if named:
        return named
    out: List[Tuple] = []
    for (_, b), (a, _) in zip(tl.keep, tl.keep[1:]):
        if a - b >= LONG_WAIT:
            out.append((tl.to_out(b, end=True), "later", 0.4))
    for h in (hints or []) if picks is None else []:
        try:
            at = float(h.get("at"))
        except (TypeError, ValueError):
            continue
        mood = str(h.get("mood") or "").strip().lower()
        mood = mood if mood in meme_lib.MOODS else meme_lib.mood_of_emoji(h.get("emoji"))
        if mood:
            out.append((tl.to_out(at), mood, 0.0))
    out += [(tl.to_out(t), "laugh" if kind == "laugh" else "hype", 0.5) for t, kind in reactions]
    for a, _, token in words_src:
        low = token.lower()
        emoji = next((e for pattern, e in KEYWORD_EMOJI if re.search(pattern, low)), "")
        mood = meme_lib.mood_of_emoji(emoji)
        if mood:
            out.append((tl.to_out(a), mood, 1.0))
    out += [(p, ("shock", "laugh", "hype")[k % 3], 2.0) for k, p in enumerate(peaks)]
    return out


def phrase_end(at: float, words_src: Sequence[Word], db: Optional["np.ndarray"]) -> float:
    """The end of the sentence around ``at`` (source time): a meme must not cut into a word."""
    best: Optional[float] = None
    for i, word in enumerate(words_src):
        stop = spoken_span(word)[1]
        if not at - 0.8 <= stop <= at + 2.0:
            continue
        following = words_src[i + 1][0] if i + 1 < len(words_src) else None
        pause = following is None or following - stop >= 0.15 or word[2].rstrip().endswith((".", "?", "!", "।", "۔"))
        score = abs(stop - at) + (0.0 if pause else 1.5)
        if best is None or score < best[0]:
            best = (score, stop)
    t = best[1] if best is not None else at
    if db is not None:                                        # the quietest instant just after it
        i, j = max(0, int(t / HOP)), min(len(db), int((t + 0.25) / HOP))
        if j > i:
            t = (i + int(np.argmin(db[i:j]))) * HOP
    return t


def place_memes(cues: List["meme_lib.Cue"], tl: Timeline, words_src: Sequence[Word], db: Optional["np.ndarray"], fps: float) -> Timeline:
    """Give every cue its place and return the timeline with the waits in it. Updates the cues' times (output) and ``at``."""
    grid = lambda t: round(t * fps) / fps
    inserts: List[Tuple[float, float]] = []
    for cue in cues:
        at = tl.to_source(cue.t)
        if cue.meme.how == "end":
            at = tl.keep[-1][1]
        elif cue.meme.how == "cut":
            at = phrase_end(at, words_src, db)
            if tl.to_out(at) < 3.0:                           # never over the opening
                at = tl.to_source(3.0)
        at = grid(min(max(at, tl.keep[0][0]), tl.keep[-1][1]))
        for (_, b), (a, _) in zip(tl.keep, tl.keep[1:]):      # inside a cut-out pause: the cut is the place
            if b < at < a:
                at = b
        cue.at = at
        if cue.meme.inserted:
            cue.dur = max(1, int(round(cue.dur * fps))) / fps
            inserts.append((at, cue.dur))
    final = Timeline(tl.keep, inserts)
    for cue in cues:
        cue.t = final.to_out(cue.at, end=cue.meme.inserted)
    cues.sort(key=lambda c: c.t)
    return final


def sound_events(popups: Sequence[Popup], punches, hook: str, cta: str, duration: float) -> List[Tuple[float, str]]:
    events: List[Tuple[float, str]] = []
    if hook:
        events.append((0.05, "whoosh"))
    events += [(p.t, "pop") for p in popups]
    events += [(a, "whoosh") for a, _, _ in punches]
    if cta:
        events.append((max(0.0, duration - 1.7), "ding"))
    out: List[Tuple[float, str]] = []
    for t, name in sorted(events):
        if not out or t - out[-1][0] >= 0.5:
            out.append((t, name))
    return out


# --------------------------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------------------------- #
@dataclass
class EditPlan:
    effects: Dict[str, bool]
    timeline: Timeline
    source_duration: float
    words: List[Word] = field(default_factory=list)             # output time
    steps: List[Tuple[float, float]] = field(default_factory=list)
    punches: List[Tuple[float, float, float]] = field(default_factory=list)
    popups: List[Popup] = field(default_factory=list)
    memes: List["meme_lib.Cue"] = field(default_factory=list)
    sounds: List[Tuple[float, str]] = field(default_factory=list)
    hook: str = ""
    cta: str = ""
    keywords: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.timeline.duration

    def zoom_at(self, t: float) -> float:
        z = 1.0
        for start, level in self.steps:
            if t >= start:
                z = level
            else:
                break
        for a, b, level in self.punches:
            if a <= t < b:
                ramp = min(1.0, (t - a) / 0.15, (b - t) / 0.25)
                ease = ramp * ramp * (3 - 2 * ramp)
                z = max(z, 1.0 + (level - 1.0) * ease)
        return z

    def summary(self) -> Dict:
        return {"cuts": len(self.timeline.keep) - 1, "removed_seconds": round(self.source_duration - self.timeline.base_duration, 1),
                "zooms": max(0, len(self.steps) - 1) + len(self.punches), "popups": len(self.popups), "memes": len(self.memes), "sounds": len(self.sounds),
                "hook": bool(self.hook), "end_card": bool(self.cta)}


def build_plan(duration: float, fps: float, words: Sequence[Word], db: Optional["np.ndarray"], effects: Dict[str, bool],
               hints: Optional[Dict] = None, title: str = "", reactions: Sequence[Tuple[float, str]] = (),
               library: Sequence["meme_lib.Meme"] = ()) -> EditPlan:
    """``words``: (start, end, text) in source time, relative to the clip. ``hints``: the AI's {hook, popups, keywords}."""
    hints = hints or {}
    pauses = find_pauses(db, words, duration) if effects.get("jumpcut") else []
    tl = Timeline(keep_ranges(pauses, duration, fps))
    cues: List["meme_lib.Cue"] = []
    if effects.get("memes") and library:
        moments = meme_moments(hints.get("popups"), words, emphasis_peaks(db, tl), tl, reactions, picks=hints.get("memes"))
        cues = meme_lib.plan(moments, library, tl.duration, seed=f"{title}|{duration:.1f}")
        tl = place_memes(cues, tl, words, db, fps)             # from here on the timeline includes the waits
    out_words = remap_words(words, tl)
    peaks = emphasis_peaks(db, tl)
    plan = EditPlan(effects=effects, timeline=tl, source_duration=duration, words=out_words, memes=cues)
    if effects.get("zoom"):
        plan.steps, plan.punches = zoom_plan(tl, out_words, peaks, strong=effects.get("sfx", False))
    if effects.get("popups"):
        plan.popups = popups_from(hints.get("popups"), words, peaks, tl, reactions)
        if plan.memes:                                           # never both on the same moment
            plan.popups = [p for p in plan.popups if all(abs(p.t - c.t) > 2.0 for c in plan.memes)]
    if effects.get("hook"):
        hook = EMOJI.sub("", str(hints.get("hook") or title or "")).strip()      # the text drawer cannot draw them
        plan.hook = (hook[:46].rsplit(" ", 1)[0] if len(hook) > 46 else hook)
    if effects.get("cta") and tl.duration >= 8 and not any(c.meme.how == "end" for c in cues):
        plan.cta = "FOLLOW FOR MORE"                           # a closing meme is the ending: no card over it
    if effects.get("sfx"):
        plan.sounds = [(t, name) for t, name in sound_events(plan.popups, plan.punches, plan.hook, plan.cta, tl.duration)
                       if not any(c.meme.inserted and c.t - 0.2 <= t < c.t + c.dur for c in cues)]      # not over a meme
    plan.keywords = [str(k).strip().lower() for k in hints.get("keywords") or [] if str(k).strip()][:12]
    return plan
