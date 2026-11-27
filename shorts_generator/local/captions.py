"""Burned-in text for a clip, as one ASS subtitle file: captions, an optional title card and a watermark.

Captions are short (a few words at a time, like the big-type captions on Shorts and Reels) and sized
relative to the output so they look the same at 1080x1920 or 1080x1080. When the transcript has word
timings (YouTube captions do) the cards follow the speech exactly; otherwise the words are spread evenly
over each segment. ffmpeg's libass renders the file onto the final video during the encode.

Styles: pop (big words, the spoken one highlighted, bouncing in: the Reels look), classic (white), yellow,
box (white on a dark box), karaoke (white words turn yellow as they are said).
Edited clips also get an animated hook headline at the start and a "follow" card at the end.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import LOCAL_CAPTION_FONT

STYLES = ("pop", "classic", "yellow", "box", "karaoke")
POP_WORDS, POP_CHARS = 3, 18
MAX_WORDS = 4          # words per caption card
MAX_CHARS = 26         # ...or this many characters, whichever comes first
MIN_DURATION = 0.35    # seconds a card stays up at the very least

Word = Tuple[float, float, str]      # start, end (seconds, absolute), text


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _clean(text: str) -> str:
    text = re.sub(r"[{}\\]", "", text)               # ASS override syntax
    return re.sub(r"\s+", " ", text).strip()


def _font_for(text: str) -> str:
    if LOCAL_CAPTION_FONT:
        return LOCAL_CAPTION_FONT
    if re.search(r"[ऀ-ॿ]", text):          # Devanagari
        return "Nirmala UI"
    return "Arial"                                   # Latin and Arabic/Urdu script


# YouTube's automatic captions write sounds in brackets, e.g. "[Laughter]", "[हंसी]", "(Music)". They are not speech.
_REACTIONS = (("laugh", r"laugh|haha|हंसी|हँसी|हसी|ठहाके"), ("applause", r"applause|clap|cheer|ताली|तालियां|तालियाँ|हौसला|अफ़ज़ाई|अफजाई"))


def split_annotations(words: List[Word]) -> Tuple[List[Word], List[Tuple[float, str]]]:
    """(spoken words, [(time, "laugh" | "applause")]): bracketed sound notes are taken out, reactions are remembered."""
    spoken: List[Word] = []
    reactions: List[Tuple[float, str]] = []
    inside: List[Word] = []
    closer = ""
    # a note that began in the previous caption line: "... आवाज़]" with no opening bracket
    first_close = next((i for i, w in enumerate(words) if re.search(r"[\])]", w[2])), None)
    first_open = next((i for i, w in enumerate(words) if re.search(r"[\[(]", w[2])), None)
    if first_close is not None and (first_open is None or first_close < first_open):
        words = words[first_close + 1:]

    def note(group: List[Word]) -> None:
        text = " ".join(w[2] for w in group).lower()
        for kind, pattern in _REACTIONS:
            if re.search(pattern, text):
                reactions.append((group[0][0], kind))
                return

    for word in words:
        text = word[2]
        if not closer and (text.startswith("[") or text.startswith("(")):
            closer = "]" if text.startswith("[") else ")"
        if closer:
            inside.append(word)
            if closer in text:
                note(inside)
                inside, closer = [], ""
            continue
        spoken.append(word)
    if inside:                                    # a note that runs past the end of the line
        note(inside)
    return spoken, reactions


def _segment_words(seg: Dict) -> List[Word]:
    return split_annotations(_raw_words(seg))[0]


def _raw_words(seg: Dict) -> List[Word]:
    words = seg.get("words")
    if words:
        out = [(float(w["start"]), float(w["end"]), _clean(str(w["text"]))) for w in words]
        return [w for w in out if w[2]]
    toks = _clean(str(seg.get("text", ""))).split(" ")
    toks = [t for t in toks if t]
    if not toks:
        return []
    span = max(0.1, seg["end"] - seg["start"])
    weights = [max(1, len(t)) for t in toks]
    total = float(sum(weights))
    out, t = [], float(seg["start"])
    for tok, w in zip(toks, weights):
        out.append((t, t + span * w / total, tok))
        t += span * w / total
    return out


def _group(words: List[Word], max_words: int = MAX_WORDS, max_chars: int = MAX_CHARS) -> List[List[Word]]:
    cards: List[List[Word]] = []
    cur: List[Word] = []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x[2] for x in cur + [w])) > max_chars):
            cards.append(cur)
            cur = []
        cur.append(w)
    if cur:
        cards.append(cur)
    return cards


def build_events(segments: List[Dict], clip_start: float, clip_end: float, max_words: int = MAX_WORDS,
                 max_chars: int = MAX_CHARS) -> List[Tuple[float, float, List[Word]]]:
    """Caption cards as (start, end, words), times relative to the clip start."""
    duration = clip_end - clip_start
    raw: List[Tuple[float, float, List[Word]]] = []
    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        words = _segment_words(seg)
        if not any(re.search(r"\w", w[2]) for w in words):      # punctuation-only / empty
            continue
        for card in _group(words, max_words, max_chars):
            raw.append((card[0][0] - clip_start, card[-1][1] - clip_start,
                        [(a - clip_start, b - clip_start, t) for a, b, t in card]))

    raw.sort(key=lambda e: e[0])
    out: List[Tuple[float, float, List[Word]]] = []
    for i, (s, e, words) in enumerate(raw):
        nxt = raw[i + 1][0] if i + 1 < len(raw) else duration
        e = max(e, min(s + MIN_DURATION, nxt))       # keep short cards readable...
        e = min(e, nxt, duration)                    # ...but never overlap the next one
        s = max(0.0, s)
        if e - s >= 0.1 and s < duration:
            out.append((s, e, words))
    return out


def _card_text(words: List[Word], start: float, end: float, karaoke: bool) -> str:
    if not karaoke:
        return " ".join(w[2] for w in words)
    parts = []
    for i, (a, b, tok) in enumerate(words):
        nxt = words[i + 1][0] if i + 1 < len(words) else end
        parts.append(f"{{\\k{max(1, round((max(nxt, a + 0.05) - max(a, start)) * 100))}}}{tok}")
    return " ".join(parts)


def _pop_events(events, keywords: Sequence[str], width: int) -> List[str]:
    """One line per spoken word: the card of up to three words, the current one big and yellow, key words green."""
    keys = {re.sub(r"\W", "", k.lower()) for k in keywords if k}
    lines: List[str] = []
    for start, end, words in events:
        font = _font_for(" ".join(w[2] for w in words))
        if font == "Arial" and not LOCAL_CAPTION_FONT:
            font = "Arial Black"
        size = f"\\fs{round(width * 0.098)}" if font == "Nirmala UI" else ""          # Devanagari reads small at the same size
        for i, (a, _, _tok) in enumerate(words):
            t0 = start if i == 0 else max(start, a)
            t1 = end if i + 1 == len(words) else min(end, words[i + 1][0])
            if t1 - t0 < 0.04:
                continue
            parts = []
            for j, (_, _, tok) in enumerate(words):
                bare = re.sub(r"\W", "", tok.lower())
                color = "&H00FFFF&" if j == i else ("&H40FF40&" if bare and (bare in keys or bare.isdigit()) else "&HFFFFFF&")
                scale = r"\fscx112\fscy112" if j == i else r"\fscx100\fscy100"
                parts.append("{\\c" + color + scale + "}" + tok.upper())
            bounce = r"\fscx80\fscy80\t(0,110,\fscx100\fscy100)" if i == 0 else ""
            lines.append(f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Pop,,0,0,0,," + "{\\fn" + font + size + bounce + "}" + " ".join(parts))
    return lines


def _style_line(name: str, font: str, size: int, primary: str, secondary: str, outline_c: str, back: str,
                border_style: int, outline: int, alignment: int, margin_lr: int, margin_v: int) -> str:
    return (f"Style: {name},{font},{size},{primary},{secondary},{outline_c},{back},-1,0,0,0,100,100,0,0,"
            f"{border_style},{outline},0,{alignment},{margin_lr},{margin_lr},{margin_v},1")


def build_ass(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    style: str = "classic",
    captions: bool = True,
    title: str = "",
    title_seconds: float = 3.0,
    title_top: Optional[int] = None,
    watermark: str = "",
    keywords: Sequence[str] = (),
    hook: str = "",
    cta: str = "",
    margin_bottom: Optional[float] = None,
    cta_y: float = 0.40,
) -> Optional[str]:
    """ASS document for the clip, or None when there is nothing to draw.

    ``hook``: an animated headline for the first seconds; ``cta``: a "follow" card at the end; ``margin_bottom``: how far
    up from the bottom the captions sit, as a share of the height (the platform's own buttons cover the bottom edge).
    """
    style = style if style in STYLES else "classic"
    if not captions:
        events = []
    elif style == "pop":
        events = build_events(segments, clip_start, clip_end, POP_WORDS, POP_CHARS)
    else:
        events = build_events(segments, clip_start, clip_end)
    title = _clean(title)[:90]
    watermark = _clean(watermark)[:40]
    hook, cta = _clean(hook)[:60], _clean(cta)[:40]
    if not events and not title and not watermark and not hook and not cta:
        return None

    duration = clip_end - clip_start
    size = round(width * 0.064)
    outline = max(3, round(width * 0.0055))
    margin_v = round(height * (margin_bottom if margin_bottom is not None else (0.23 if style == "pop" else 0.16)))
    margin_lr = round(width * 0.06)
    white, black, gold = "&H00FFFFFF", "&H00000000", "&H0000D5FF"
    if style == "yellow":
        primary, secondary, outline_c, back, border = gold, gold, black, black, 1
    elif style == "box":
        primary, secondary, outline_c, back, border, outline = white, white, "&H99000000", "&H99000000", 3, max(8, round(width * 0.012))
    elif style == "karaoke":
        primary, secondary, outline_c, back, border = gold, white, black, black, 1       # \k sweeps secondary -> primary
    else:
        primary, secondary, outline_c, back, border = white, white, black, black, 1

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        _style_line("Default", "Arial", size, primary, secondary, outline_c, back, border, outline, 2, margin_lr, margin_v),
        _style_line("Title", "Arial", round(width * 0.072), white, white, black, black, 1, max(5, outline + 2), 8,
                    round(width * 0.07), title_top if title_top is not None else round(height * 0.07)),
        _style_line("Mark", "Arial Black", round(width * 0.042), white, white, "&H60000000", black, 3, max(6, round(width * 0.008)), 7,
                    round(width * 0.04), round(height * 0.03)),                  # the channel's badge, top left
        _style_line("Pop", "Arial Black", round(width * 0.080), white, white, black, "&H80000000", 1, max(6, round(width * 0.0075)), 2,
                    margin_lr, margin_v),
        _style_line("Hook", "Arial Black", round(width * 0.072), black, black, "&H0000DDFF", "&H0000DDFF", 3, max(10, round(width * 0.014)), 8,
                    round(width * 0.08), title_top if title_top is not None else round(height * 0.085)),
        _style_line("Cta", "Arial Black", round(width * 0.074), white, white, black, black, 1, max(7, round(width * 0.008)), 5,
                    margin_lr, 0),
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if style == "pop":
        lines += _pop_events(events, keywords, width)
    for s, e, words in (events if style != "pop" else []):
        text = _card_text(words, s, e, karaoke=(style == "karaoke"))
        font = _font_for(text)
        override = f"{{\\fn{font}}}" if font != "Arial" else ""
        lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Default,,0,0,0,,{override}{text}")
    if title:
        end = min(duration, max(1.0, title_seconds))
        font = _font_for(title)
        override = f"{{\\fad(250,450)\\fn{font}}}" if font != "Arial" else "{\\fad(250,450)}"
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(end)},Title,,0,0,0,,{override}{title}")
    if hook:
        font = _font_for(hook)
        face = "" if font == "Arial" else "\\fn" + font
        pop = r"\fscx60\fscy60\t(0,160,\fscx108\fscy108)\t(160,280,\fscx100\fscy100)\fad(0,250)"
        lines.append(f"Dialogue: 2,{_ass_time(0)},{_ass_time(min(duration, 3.0))},Hook,,0,0,0,," + "{" + pop + face + "}" + hook.upper())
    if cta:
        start = max(0.0, duration - 1.6)
        pop = "\\an5\\pos(" + f"{width // 2},{round(height * cta_y)}" + r")\fscx50\fscy50\t(0,150,\fscx110\fscy110)\t(150,260,\fscx100\fscy100)"
        lines.append(f"Dialogue: 3,{_ass_time(start)},{_ass_time(duration)},Cta,,0,0,0,," + "{" + pop + "}" + cta.upper())
    if watermark:
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(duration)},Mark,,0,0,0,,{watermark}")
    return "\n".join(lines) + "\n"
