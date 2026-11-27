"""Pictures drawn on top of a frame: emoji / short-text popups with a bouncy entrance, and the progress bar.

Colour emoji come from the system's emoji font (Segoe UI Emoji on Windows) through Pillow. libass (used for captions)
cannot draw colour emoji, which is why popups are drawn here. Without Pillow or an emoji font the popups are simply
skipped (text popups still work where a font exists); nothing fails.
"""
import math
import os
import re
from functools import lru_cache
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except ImportError:  # pragma: no cover - optional
    Image = ImageDraw = ImageFont = None

_WIN = os.environ.get("WINDIR", "C:/Windows") + "/Fonts/"
EMOJI_FONTS = ((_WIN + "seguiemj.ttf", 220), ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109),
               ("/System/Library/Fonts/Apple Color Emoji.ttc", 160))
TEXT_FONTS = (_WIN + "ariblk.ttf", _WIN + "arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Black.ttf")
_LATIN = re.compile(r"^[\x20-\x7eÀ-ɏ‘’“”…!?]+$")


def _font(paths_sizes):
    if ImageFont is None:
        return None
    for path, size in paths_sizes:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def _bgra(img) -> "np.ndarray":
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def _sticker(sprite: "np.ndarray", rim: int) -> "np.ndarray":
    """A white rim around the shape (a "sticker"), so a dark emoji still shows on a dark picture."""
    pad = rim + 2
    h, w = sprite.shape[:2]
    canvas = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = sprite
    solid = (canvas[:, :, 3] > 40).astype(np.uint8) * 255
    border = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1, 2 * rim + 1)))
    border = cv2.GaussianBlur(border, (3, 3), 0)
    out = np.empty_like(canvas)
    out[:, :, :3] = 255
    out[:, :, 3] = border
    a = canvas[:, :, 3:4].astype(np.float32) / 255.0
    out[:, :, :3] = (canvas[:, :, :3] * a + out[:, :, :3] * (1 - a)).astype(np.uint8)
    out[:, :, 3] = np.maximum(out[:, :, 3], canvas[:, :, 3])
    return out


@lru_cache(maxsize=64)
def emoji_sprite(char: str, size: int) -> Optional["np.ndarray"]:
    """BGRA picture of one emoji, ``size`` pixels high (None when it cannot be drawn in colour)."""
    font = _font(EMOJI_FONTS)
    if font is None or not char:
        return None
    side = int(font.size * 1.6)
    img = Image.new("RGBA", (side * 2, side), (0, 0, 0, 0))
    try:
        ImageDraw.Draw(img).text((side, side // 2), char, font=font, embedded_color=True, anchor="mm")
    except (OSError, ValueError):
        return None
    box = img.getbbox()
    if not box or box[3] - box[1] < 8:
        return None
    img = img.crop(box)
    w, h = img.size
    img = img.resize((max(1, round(w * size / h)), size), Image.LANCZOS)
    return _sticker(_bgra(img), max(3, size // 28))


@lru_cache(maxsize=64)
def text_sprite(text: str, size: int) -> Optional["np.ndarray"]:
    """A short word or two on a rounded yellow badge. Latin script only (other scripts need a shaping engine)."""
    text = " ".join(text.split()).upper()
    font = _font([(p, size) for p in TEXT_FONTS])
    if font is None or not text or not _LATIN.match(text):
        return None
    pad_x, pad_y = int(size * 0.45), int(size * 0.28)
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    img = Image.new("RGBA", (w + 12, h + 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((6, 9, w + 6, h + 9), radius=h // 3, fill=(0, 0, 0, 130))            # soft shadow
    draw.rounded_rectangle((6, 6, w + 6, h + 6), radius=h // 3, fill=(255, 221, 0, 255), outline=(0, 0, 0, 255), width=4)
    draw.text((6 + pad_x - left, 6 + pad_y - top), text, font=font, fill=(15, 15, 15, 255))
    return _bgra(img)


def popup_sprite(emoji: str, text: str, width: int) -> Optional["np.ndarray"]:
    """Emoji with an optional badge under it, sized for an output ``width`` wide."""
    badge = text_sprite(text, int(width * 0.055)) if text else None
    if badge is not None and badge.shape[1] > width * 0.7:            # a long phrase: shrink it to fit
        k = width * 0.7 / badge.shape[1]
        badge = cv2.resize(badge, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    parts = [p for p in (emoji_sprite(emoji, int(width * 0.17)) if emoji else None, badge) if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    w = max(p.shape[1] for p in parts)
    h = sum(p.shape[0] for p in parts)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    y = 0
    for p in parts:
        x = (w - p.shape[1]) // 2
        out[y:y + p.shape[0], x:x + p.shape[1]] = p
        y += p.shape[0]
    return out


def pop_scale(age: float) -> float:
    """Entrance: grows from small, overshoots, settles ("ease out back")."""
    if age >= 0.28:
        return 1.0
    x = max(0.0, age / 0.28) - 1.0
    return 0.3 + 0.7 * (1 + 2.7 * x ** 3 + 1.7 * x ** 2)


def pop_alpha(age: float, duration: float) -> float:
    return max(0.0, min(1.0, age / 0.08, (duration - age) / 0.2))


def blend(frame: "np.ndarray", sprite: "np.ndarray", cx: float, cy: float, scale: float = 1.0, alpha: float = 1.0) -> None:
    """Draw a BGRA sprite centred at (cx, cy) onto a BGR frame, in place, clipped to the frame."""
    if alpha <= 0 or scale <= 0.02:
        return
    if abs(scale - 1.0) > 0.01:
        sprite = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = sprite.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    part = sprite[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = (part[:, :, 3:4].astype(np.uint16) * int(alpha * 256)) >> 8
    region = frame[fy0:fy1, fx0:fx1].astype(np.uint16)
    frame[fy0:fy1, fx0:fx1] = ((region * (255 - a) + part[:, :, :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)


def draw_popup(frame: "np.ndarray", sprite: "np.ndarray", age: float, duration: float, side: int, y_frac: float = 0.24) -> None:
    h, w = frame.shape[:2]
    half = sprite.shape[1] * 1.15 / 2                       # the entrance overshoots by 15%
    cx = w * (0.26 if side == 0 else 0.74)
    cx = max(half + 8, min(w - half - 8, cx)) if 2 * half + 16 < w else w / 2      # always fully on screen
    cy = h * y_frac + math.sin(age * 7.0) * h * 0.004
    blend(frame, sprite, cx, cy, pop_scale(age), pop_alpha(age, duration))


def draw_progress(frame: "np.ndarray", fraction: float) -> None:
    h, w = frame.shape[:2]
    bar = max(6, int(h * 0.0065))
    frame[:bar] = (frame[:bar].astype(np.uint16) * 2 // 5).astype(np.uint8)
    frame[:bar, : int(w * max(0.0, min(1.0, fraction)))] = (0, 213, 255)         # BGR amber
