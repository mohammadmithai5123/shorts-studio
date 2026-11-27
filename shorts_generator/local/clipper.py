"""Local clipping: stable, layout-aware vertical reframing of one clip.

Per clip:
  1. Get the clip's pixels. A streamed video arrives already trimmed to the clip (see source.py), so
     it is used as is; a local file is cut with ffmpeg (fast input seek).
  2. Analyse it: sample frames (~4 fps), find baked-in letterbox bars and
     detect faces.
  3. Pick a layout (``--layout auto`` decides):
       cutout a streamer laid over the game with no box (green screen): the person goes
              on top on a plain colour, the game below (see cutout.py).
       split  streamer / gameplay with a facecam overlay. The overlay is a face
              that stays put in a corner for the whole clip: facecam on top,
              gameplay below. Game characters' faces never steer the crop.
       smart  everything else (shows, podcasts, panels, vlogs): every camera shot
              gets its own steady framing (see framing.py).
       track  (on request) ONE crop window that follows the speaker slowly.
       fit    whole frame, undistorted, centred over a blurred copy of itself.
  4. Optionally plan automatic edits (editing.py): jump cuts, zooms, popups, ...
  5. Render at a real output size (1080x1920 for 9:16): a few threads compose
     frames while ffmpeg encodes them (on the graphics chip when it can, see
     encoder.py) with the clip's audio, cut and mixed the same way.

Aspect ratio is never stretched: every layout crops or letterboxes, it never
resizes with a different x/y scale.
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "opencv-python is required for --mode local. Install it with:\n"
        "    pip install -r requirements-local.txt"
    ) from e

from .. import config
from . import control, cutout, editing, encoder, framing, memes, overlays, sfx
from .captions import STYLES, _raw_words, _segment_words, build_ass, split_annotations
from .faces import FaceDetector

LAYOUTS = ("auto", "track", "split", "fit", "cutout")   # "auto" resolves to cutout, split or smart (per camera shot)

SAMPLE_FPS = 4.0             # face-analysis sampling rate
OUT_FPS_MAX = 30.0           # output frame-rate cap (a 60 fps source is halved)
FACECAM_PANEL_FRAC = 0.38    # split: share of the output height used by the facecam
FACECAM_WIDTH_X_FACE = 2.8   # split fallback: facecam width as a multiple of the face width


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _out_size(aspect_ratio: str, base: int = 1080) -> Tuple[int, int]:
    """Real output size: 9:16 → 1080x1920, 1:1 → 1080x1080, 16:9 → 1920x1080."""
    r = _ratio(aspect_ratio)
    if r <= 1:
        w, h = base, base / r
    else:
        w, h = base * r, base
    return int(round(w / 2) * 2), int(round(h / 2) * 2)


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """Cut [start, end] out of a LOCAL file with ffmpeg (streamed videos arrive pre-trimmed).

    ``-ss`` goes *before* ``-i`` (fast seek; still frame-accurate because we
    re-encode). Putting it after ``-i`` makes ffmpeg decode the whole video up to
    ``start`` — minutes for a clip an hour into a long video. The intermediate is
    30 fps (the output is 30 fps anyway) and lightly compressed: re-encoding 1080p is
    slow on a CPU, and the final quality comes from the render step.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{max(0.1, end - start):.3f}",
        "-vf", "fps=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _find_bars(maxes: List["np.ndarray"], thresh: int = 40, need: float = 0.9, min_px: int = 4) -> Tuple[int, int]:
    """Leading/trailing lines that are black in (nearly) every sampled frame."""
    m = np.stack(maxes)
    dark = (m < thresh).mean(axis=0) >= need
    n = len(dark)
    lead = 0
    while lead < n and dark[lead]:
        lead += 1
    trail = 0
    while trail < n - lead and dark[n - 1 - trail]:
        trail += 1
    if lead < min_px:
        lead = 0
    if trail < min_px:
        trail = 0
    if n - lead - trail < 0.3 * n:
        return 0, 0
    # 2px safety margin so bicubic taps never reach into the bar
    return (lead + 2 if lead else 0), (trail + 2 if trail else 0)


def _analyse(path: str) -> Dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / SAMPLE_FPS)))

    detector = FaceDetector(width, height)

    sidx: List[int] = []
    boxes: List[List[Tuple[int, int, int, int]]] = []
    row_max: List["np.ndarray"] = []
    col_max: List["np.ndarray"] = []
    prev = None
    motion_sum = None
    n_diff = 0
    idx = 0
    try:
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                row_max.append(gray.max(axis=1))
                col_max.append(gray.max(axis=0))
                # Mean |frame - previous sample|: a facecam overlay barely moves
                # next to the game picture, which exposes its rectangle.
                g32 = gray.astype(np.float32)
                if prev is not None:
                    d = np.abs(g32 - prev)
                    motion_sum = d if motion_sum is None else motion_sum + d
                    n_diff += 1
                prev = g32
                sidx.append(idx)
                boxes.append(detector.detect(frame))
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()

    if not sidx:
        raise RuntimeError(f"no frames decoded from {path}")

    top, bottom = _find_bars(row_max)
    left, right = _find_bars(col_max)
    roi = (left, top, width - left - right, height - top - bottom)
    motion = (motion_sum / n_diff) if motion_sum is not None and n_diff else None
    return {
        "fps": fps, "n_frames": idx, "size": (width, height), "roi": roi,
        "sidx": sidx, "boxes": boxes, "motion": motion,
    }


def _smooth(a: "np.ndarray", k: int = 5) -> "np.ndarray":
    return np.convolve(np.pad(a, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _step_edge(profile: "np.ndarray", start: int, direction: int, lo: int, hi: int,
               min_step: float, win: int = 8) -> Optional[int]:
    """Sharpest *rise* of ``profile`` walking from ``start`` outward.

    Inside a facecam the motion is low, just outside it jumps up to the game's
    level. Returns the first pixel of the inner side of the best step between
    ``lo`` and ``hi`` pixels away, or None if no step is at least ``min_step``.
    """
    n = len(profile)
    best_score, best_pos = 0.0, None
    for dist in range(max(win, lo), hi):
        p = start + direction * dist
        a, b = p + direction * win, p - direction * win
        if min(p, a, b) < 0 or max(p, a, b) > n:
            continue
        outer = profile[min(p, a):max(p, a)].mean()
        inner = profile[min(p, b):max(p, b)].mean()
        if outer - inner > best_score:
            best_score, best_pos = outer - inner, p
    return best_pos if best_score >= min_step else None


def _overlay_rect(motion: Optional["np.ndarray"], face: Tuple[float, float, float, float],
                  roi: Tuple[int, int, int, int]) -> Optional[Tuple[float, float, float, float]]:
    """Rectangle (x, y, w, h; full-frame coords) of the facecam around ``face``.

    Uses the temporal motion map: the facecam stays calm while the game picture
    changes. Returns None when the evidence is weak, so the caller falls back to
    a face-proportion guess.
    """
    if motion is None:
        return None
    rx, ry, rw, rh = roi
    fx, fy, fw, fh = face
    cx, cy = int(fx + fw / 2), int(fy + fh / 2)
    game = float(np.percentile(motion[ry:ry + rh, rx:rx + rw], 80))
    if game < 6.0:                      # (nearly) static footage: no contrast to work with
        return None
    min_step = max(3.0, 0.2 * game)

    x_lo, x_hi, y_lo, y_hi = rx, rx + rw, ry, ry + rh
    band_y = slice(int(max(y_lo, cy - 0.35 * fh)), int(min(y_hi, cy + 0.35 * fh)))
    col = _smooth(motion[band_y, :].mean(axis=0), 7)
    left = _step_edge(col, cx, -1, int(0.6 * fw), int(min(6 * fw, cx - x_lo)), min_step)
    right = _step_edge(col, cx, +1, int(0.6 * fw), int(min(6 * fw, x_hi - cx)), min_step)
    if left is None:                    # no step: overlay is flush with the frame edge (or lost)
        left = x_lo if cx - x_lo <= 6 * fw else None
    if right is None:
        right = x_hi if x_hi - cx <= 6 * fw else None
    if left is None or right is None or right - left < 1.4 * fw or right - left > 8 * fw:
        return None

    inset_x = int(0.12 * (right - left))
    row = _smooth(motion[:, left + inset_x:right - inset_x].mean(axis=1), 7)
    top = _step_edge(row, cy, -1, int(0.6 * fh), int(min(6 * fh, cy - y_lo)), min_step)
    bottom = _step_edge(row, cy, +1, int(0.6 * fh), int(min(6 * fh, y_hi - cy)), min_step)
    if top is None:
        top = y_lo if cy - y_lo <= 6 * fh else None
    if bottom is None:
        bottom = y_hi if y_hi - cy <= 6 * fh else None
    if top is None or bottom is None or bottom - top < 1.2 * fh or bottom - top > 8 * fh:
        return None

    m = 2   # stay clear of the border / rounded corners
    return float(left + m), float(top + m), float(right - left - 2 * m), float(bottom - top - 2 * m)


def _primary_cluster(sample_boxes, roi) -> Optional[Dict]:
    """Group per-sample detections into tracks; return the one seen most often.

    Detections are associated by position, so a face that stays put (a facecam)
    forms one dense track, while characters moving through a game scene form
    short, scattered ones.
    """
    rx, ry, rw, rh = roi
    clusters: List[Dict] = []
    for si, boxes in enumerate(sample_boxes):
        taken = set()
        for (x, y, w, h) in sorted(boxes, key=lambda b: -b[2] * b[3]):
            x, y = x - rx, y - ry
            cx, cy = x + w / 2.0, y + h / 2.0
            if not (0 <= cx < rw and 0 <= cy < rh):
                continue
            best, best_d = None, float("inf")
            for ci, c in enumerate(clusters):
                if ci in taken:
                    continue
                d = ((cx - c["cx"]) ** 2 + (cy - c["cy"]) ** 2) ** 0.5
                if d < 0.6 * max(w, c["w"]) and d < best_d:
                    best, best_d = ci, d
            if best is None:
                clusters.append({"cx": cx, "cy": cy, "w": float(w), "hits": {si: (x, y, w, h)}})
                taken.add(len(clusters) - 1)
            else:
                c = clusters[best]
                c["hits"][si] = (x, y, w, h)
                taken.add(best)
                c["cx"] = 0.7 * c["cx"] + 0.3 * cx
                c["cy"] = 0.7 * c["cy"] + 0.3 * cy
                c["w"] = 0.7 * c["w"] + 0.3 * w
    if not clusters:
        return None
    return max(clusters, key=lambda c: len(c["hits"]))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _track_x0s(an: Dict, cluster: Dict, roi, crop_w: float) -> "np.ndarray":
    """Per-source-frame left edge of a crop window that follows the speaker calmly."""
    _, _, rw, _ = roi
    sidx = np.asarray(an["sidx"], dtype=float)
    n = len(sidx)
    centers = np.full(n, np.nan)
    rx = roi[0]
    for si, boxes in enumerate(an["boxes"]):
        if si in cluster["hits"]:                       # the speaker we have been following
            b = cluster["hits"][si]
            centers[si] = b[0] + b[2] / 2.0
        elif boxes:                                     # a camera cut to someone else: follow the biggest face there
            x, _, w, h = max(boxes, key=lambda b: b[2] * b[3])
            centers[si] = (x - rx) + w / 2.0
    ids = np.arange(n)
    good = ~np.isnan(centers)
    centers = np.interp(ids, ids[good], centers[good])          # fill gaps

    pad = np.pad(centers, 2, mode="edge")                          # kill outliers
    centers = np.array([np.median(pad[i:i + 5]) for i in range(n)])

    dead = 0.08 * crop_w                                           # ignore small movements
    max_step = 0.35 * crop_w / SAMPLE_FPS                          # ≤ 0.35 crop-widths per second
    cur = centers[0]
    path = np.empty(n)
    for i, t in enumerate(centers):
        err = t - cur
        if abs(err) > dead:
            cur += float(np.clip(err - np.sign(err) * dead, -max_step, max_step))
        path[i] = cur
    path = np.convolve(np.pad(path, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")

    per_frame = np.interp(np.arange(an["n_frames"]), sidx, path)
    return np.clip(per_frame - crop_w / 2.0, 0, rw - crop_w)


def _plan(an: Dict, layout: str, ratio: float, out_w: int, out_h: int) -> Dict:
    rx, ry, rw, rh = an["roi"]
    n = len(an["sidx"])
    cl = _primary_cluster(an["boxes"], an["roi"])
    presence = (len(cl["hits"]) / n) if cl else 0.0

    if layout == "fit":
        return {"layout": "fit", "why": "requested"}
    if cl is None or presence < 0.25:
        why = "no face" if cl is None else f"face in only {presence:.0%} of samples"
        return {"layout": "fit", "why": why}

    hits = list(cl["hits"].values())
    mx, my, mw, mh = (float(np.median([b[k] for b in hits])) for k in range(4))
    cx, cy = mx + mw / 2.0, my + mh / 2.0
    spread_x = float(np.std([b[0] + b[2] / 2.0 for b in hits]))
    spread_y = float(np.std([b[1] + b[3] / 2.0 for b in hits]))
    static = spread_x < 0.03 * rw and spread_y < 0.03 * rh
    corner = abs(cx - rw / 2.0) > 0.28 * rw and (cy < 0.38 * rh or cy > 0.62 * rh) and mh < 0.26 * rh

    if layout == "auto":
        if corner and static:
            chosen, why = "split", f"static facecam in a corner (face in {presence:.0%} of samples)"
        elif presence >= 0.5:
            chosen, why = "track", f"speaker in {presence:.0%} of samples"
        else:
            return {"layout": "fit", "why": f"face in only {presence:.0%} of samples"}
    else:
        chosen, why = layout, "requested"

    if chosen == "track":
        ch = float(rh)
        cw = ch * ratio
        if cw > rw:
            cw = float(rw)
            ch = cw / ratio
        return {
            "layout": "track", "why": why, "cw": cw, "ch": ch,
            "y0": ry + (rh - ch) / 2.0, "x0s": rx + _track_x0s(an, cl, an["roi"], cw),
        }

    # split: facecam on top, gameplay below. Everything below is in full-frame coords.
    face_cx, face_cy = rx + cx, ry + cy
    overlay = _overlay_rect(an.get("motion"), (rx + mx, ry + my, mw, mh), an["roi"])
    if overlay is not None:
        ox, oy, ow, oh = overlay
        # Let the panel follow the overlay's own shape so nothing is cut off.
        top_h = int(round(_clamp(out_w * oh / ow, 0.28 * out_h, 0.45 * out_h) / 2) * 2)
        how = f"overlay rect {ow:.0f}x{oh:.0f}"
    else:
        # No reliable rectangle: guess from the face size (works for the common ~3x layout).
        ow = min(float(rw), mw * FACECAM_WIDTH_X_FACE)
        oh = ow * 0.66
        ox, oy = face_cx - ow / 2.0, face_cy + 0.05 * mh - oh / 2.0
        top_h = int(round(out_h * FACECAM_PANEL_FRAC / 2) * 2)
        how = "face-size guess"
    bot_h = out_h - top_h
    ap = out_w / top_h

    # crop with the panel's aspect ratio, inside the overlay, centred on the face
    if ow / oh >= ap:
        fch, fcw = oh, oh * ap
    else:
        fcw, fch = ow, ow / ap
    fcx = _clamp(face_cx, ox + fcw / 2.0, ox + ow - fcw / 2.0)
    fcy = _clamp(face_cy, oy + fch / 2.0, oy + oh - fch / 2.0)
    fx0 = _clamp(fcx - fcw / 2.0, 0, an["size"][0] - fcw)
    fy0 = _clamp(fcy - fch / 2.0, 0, an["size"][1] - fch)

    ag = out_w / bot_h
    gh = float(rh)
    gw = gh * ag
    if gw > rw:
        gw = float(rw)
        gh = gw / ag
    gx0 = rx + (rw - gw) / 2.0
    gy0 = ry + (rh - gh) / 2.0
    if gx0 < ox + ow and gx0 + gw > ox:         # keep the facecam out of the gameplay panel
        if ox + ow / 2.0 > rx + rw / 2.0:
            gx0 = max(float(rx), ox - gw)
        else:
            gx0 = min(rx + rw - gw, ox + ow)

    return {
        "layout": "split", "why": f"{why}; {how}", "top_h": top_h,
        "top": (fx0, fy0, fcw, fch),
        "bot": (gx0, gy0, gw, gh),
    }


# --------------------------------------------------------------------------- #
# Encode
# --------------------------------------------------------------------------- #
def _fonts_dir_option() -> str:
    """libass option pointing at the Windows fonts folder.

    The value is single-quoted and its drive-letter ':' escaped — both are required inside an
    ffmpeg filter graph (unquoted, ffmpeg fails with "No option name near '/WINDOWS/Fonts'").
    """
    windir = os.environ.get("WINDIR")
    if not windir:
        return ""
    return ":fontsdir='" + (windir.replace("\\", "/") + "/Fonts").replace(":", "\\:") + "'"


_CAPTIONS_OK: Optional[bool] = None


def _captions_supported() -> bool:
    """One-off check that this ffmpeg can burn ASS subtitles (needs libass), so a missing
    filter shows up as a warning instead of a clip that fails after minutes of rendering."""
    global _CAPTIONS_OK
    if _CAPTIONS_OK is None:
        tmp = tempfile.mkdtemp(prefix="shorts_caps_")
        try:
            with open(os.path.join(tmp, "t.ass"), "w", encoding="utf-8") as f:
                f.write(build_ass([{"start": 0.0, "end": 1.0, "text": "test"}], 0.0, 1.0, 320, 240) or "")
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2",
                 "-vf", "ass=t.ass" + _fonts_dir_option(), "-frames:v", "1", "-f", "null", "-"],
                cwd=tmp, capture_output=True,
            )
            _CAPTIONS_OK = r.returncode == 0
        except OSError:
            _CAPTIONS_OK = False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return _CAPTIONS_OK


LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"      # -14 LUFS: the loudness YouTube / Reels / TikTok aim for
COLOR_BOOST = "eq=saturation=1.16:contrast=1.04:brightness=0.01"


def _has_audio(path: str) -> bool:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                          capture_output=True, text=True)
    return bool(proc.stdout.strip())


def _filter_graph(ass: bool, color: bool, keep, has_audio: bool, has_sfx: bool, normalize: bool, pieces=None) -> Tuple[str, List[str]]:
    """The ffmpeg filter_complex (picture effects + the sound cut, mixed and levelled) and the -map arguments.

    ``pieces`` (Timeline.pieces) replaces ``keep`` when the video waits for memes: those stretches are silent here and
    the meme's own sound comes in with the effects track.
    """
    video = [f for f in (COLOR_BOOST if color else "", "ass=captions.ass" + _fonts_dir_option() if ass else "") if f]
    chains = ["[0:v]" + (",".join(video) or "null") + "[v]"]
    maps = ["-map", "[v]"]
    if not has_audio:
        return ";".join(chains), maps
    parts = pieces or [("video", a, b) for a, b in keep or []]
    if parts:                                                     # cut the sound exactly where the picture was cut
        n = len(parts)
        sources = sum(1 for kind, _, _ in parts if kind == "video")
        same = ",aformat=sample_rates=48000:channel_layouts=stereo" if sources < n else ""      # silence and speech must match
        if sources > 1:
            chains.append("[1:a]asplit=" + str(sources) + "".join(f"[s{i}]" for i in range(sources)))
        used = 0
        for i, (kind, a, b) in enumerate(parts):
            if kind != "video":
                chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={a:.4f},asetpts=PTS-STARTPTS{same}[a{i}]")
                continue
            source = f"[s{used}]" if sources > 1 else "[1:a]"
            used += 1
            fades = f",afade=t=in:d=0.015,afade=t=out:st={max(0.0, b - a - 0.015):.4f}:d=0.015"     # no clicks at the joins
            chains.append(f"{source}atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS{fades}{same}[a{i}]")
        chains.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[cut]")
    else:
        chains.append("[1:a]anull[cut]")
    last = "[cut]"
    if has_sfx:
        chains.append(f"{last}aresample=48000,aformat=channel_layouts=stereo[speech]")
        chains.append("[2:a]aformat=channel_layouts=stereo[fx]")
        chains.append("[speech][fx]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]")
        last = "[mix]"
    chains.append(f"{last}{LOUDNORM + ',' if normalize else ''}aresample=48000[a]")
    return ";".join(chains), maps + ["-map", "[a]"]


def _meme_height(plan: Dict, out_h: int) -> float:
    """Where a meme sits: over the game or the lower panel, otherwise above the people."""
    if plan["layout"] == "cutout":
        return (plan["top_h"] + plan["game_h"] * 0.58) / out_h
    if plan["layout"] == "split":
        return (plan["top_h"] + (out_h - plan["top_h"]) * 0.45) / out_h
    return 0.27


def _composer(an: Dict, plan: Dict, out_w: int, out_h: int, out_fps: float, edit, sprites: List, meme_frames: List = ()):
    """A function (source frame, source frame number, output frame number) -> yuv420p bytes. Safe to run in threads."""
    layout = plan["layout"]
    zooming = bool(edit and edit.effects.get("zoom"))
    total = max(1.0, edit.duration) if edit else 1.0

    def compose(frame, i: int, k: int) -> bytes:
        t = k / out_fps
        z = edit.zoom_at(t) if zooming else 1.0
        popup_y = 0.24
        if layout == "cutout":
            img = cutout.compose(frame, plan, i, an["roi"], out_w, out_h, z)
            popup_y = min(0.62, plan["top_h"] / out_h + 0.16)             # over the game, under the captions
        elif layout == "smart":
            img = framing.compose_smart(frame, plan, i, an["roi"], out_w, out_h, z)
            if framing.shot_for(plan, i)["kind"] in ("group", "fit"):
                popup_y = 0.17                                    # above the video block, not on the faces
        elif layout == "track":
            x0 = float(plan["x0s"][min(i, len(plan["x0s"]) - 1)])
            img = framing.warp(frame, *framing.zoom_rect(x0, plan["y0"], plan["cw"], plan["ch"], z, fy=plan["y0"] + plan["ch"] * 0.35),
                               out_w, out_h)
        elif layout == "split":
            img = np.empty((out_h, out_w, 3), dtype=np.uint8)
            th = plan["top_h"]
            img[:th] = framing.warp(frame, *framing.zoom_rect(*plan["top"], z), out_w, th)
            img[th:] = framing.warp(frame, *framing.zoom_rect(*plan["bot"], 1.0 + (z - 1.0) * 0.5), out_w, out_h - th)
            popup_y = min(0.62, th / out_h + 0.12)
        else:
            img = framing.compose_block(frame, framing.zoom_rect(*an["roi"], 1.0 + (z - 1.0) * framing.GROUP_ZOOM), an["roi"], out_w, out_h, top_frac=0.5)
            popup_y = 0.17
        if edit:
            for popup, sprite in sprites:
                age = t - popup.t
                if 0 <= age < popup.dur:
                    overlays.draw_popup(img, sprite, age, popup.dur, popup.side, popup_y)
            for cue, frames in meme_frames:
                age = t - cue.t
                if 0 <= age < cue.dur:
                    picture = frames[min(int(age * out_fps), len(frames) - 1)]
                    overlays.blend(img, picture, out_w / 2, out_h * _meme_height(plan, out_h), overlays.pop_scale(age),
                                   overlays.pop_alpha(age, cue.dur))
            if edit.effects.get("progress"):
                overlays.draw_progress(img, (t + 1.0 / out_fps) / total)
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2YUV_I420).tobytes()

    return compose


def _render(in_path: str, out_path: str, an: Dict, plan: Dict, out_w: int, out_h: int,
            ass_text: Optional[str] = None, normalize_audio: bool = True, edit=None) -> str:
    """Encode the clip. Returns the encoder used. A hardware encoder that fails is retried once on the processor."""
    name = encoder.choose()
    try:
        _encode(in_path, out_path, an, plan, out_w, out_h, ass_text, normalize_audio, edit, name)
    except EncodeFailed as e:
        if name == "cpu":
            raise RuntimeError(str(e)) from None
        print(f"[clip/local]   {encoder.ENCODERS[name]['label']} failed; encoding on the processor instead", flush=True)
        encoder.disable_hardware(str(e))
        name = "cpu"
        _encode(in_path, out_path, an, plan, out_w, out_h, ass_text, normalize_audio, edit, name)
    return name


class EncodeFailed(RuntimeError):
    pass


def _encode(in_path, out_path, an, plan, out_w, out_h, ass_text, normalize_audio, edit, encoder_name) -> None:
    src_fps = an["fps"]
    skip = max(1, int(round(src_fps / OUT_FPS_MAX)))
    out_fps = src_fps / skip
    n_total = max(1, int(an.get("n_frames", 1)))
    # ffmpeg runs from a scratch folder (subtitle and sound files are referenced by relative names, which sidesteps the
    # drive-letter ':' escaping), so every path must be absolute.
    in_path, out_path = os.path.abspath(in_path), os.path.abspath(out_path)
    work = tempfile.mkdtemp(prefix="shorts_clip_")
    try:
        if ass_text:
            with open(os.path.join(work, "captions.ass"), "w", encoding="utf-8") as f:
                f.write(ass_text)
        has_audio = _has_audio(in_path)
        keep = edit.timeline.keep if edit else None
        meme_sounds = []
        if has_audio and edit:
            meme_sounds = [(c.t, s) for c in edit.memes for s in [memes.sound_of(c)] if s is not None]
        has_sfx = bool(has_audio and edit and (edit.sounds or meme_sounds))
        if has_sfx:
            sfx.write_track(edit.sounds, edit.duration, os.path.join(work, "sfx.wav"), clips=meme_sounds)
        graph, maps = _filter_graph(bool(ass_text), bool(edit and edit.effects.get("color")), keep, has_audio, has_sfx, normalize_audio,
                                    pieces=edit.timeline.pieces() if edit and edit.timeline.inserts else None)
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{out_w}x{out_h}", "-r", f"{out_fps:.3f}", "-i", "-",
               "-i", in_path, *(["-i", "sfx.wav"] if has_sfx else []),
               "-filter_complex", graph, *maps,
               *encoder.video_args(encoder_name),
               "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
               "-shortest", "-movflags", "+faststart", out_path]
        sprites = []
        if edit:
            for popup in edit.popups:
                sprite = overlays.popup_sprite(popup.emoji, popup.text, out_w)
                if sprite is not None:
                    sprites.append((popup, sprite))
        meme_frames, waits = [], []
        for cue in edit.memes if edit else []:
            if cue.meme.inserted:
                frames = memes.load_full_frames(cue, out_fps, out_w, out_h)
                waits.append((cue.at, int(round(cue.dur * out_fps)), frames))
            else:
                frames = memes.load_frames(cue, out_fps, int(out_w * 0.72), int(out_h * 0.32))
                if frames:
                    meme_frames.append((cue, frames))
            if not frames and cue.meme.kind != "sound":
                print(f"[clip/local]   meme could not be read: {cue.meme.name}", flush=True)
        compose = _composer(an, plan, out_w, out_h, out_fps, edit, sprites, meme_frames)
        _pump(cmd, work, out_path, in_path, compose, skip, src_fps, n_total, edit,
              sorted(waits, key=lambda w: w[0]), _meme_screen(out_w, out_h, out_fps, edit))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _meme_screen(out_w: int, out_h: int, out_fps: float, edit):
    """A function (meme picture, how far through it 0-1, output frame number) -> yuv420p bytes."""
    total = max(1.0, edit.duration) if edit else 1.0

    def compose(picture, progress: float, k: int) -> bytes:
        img = memes.full_frame(picture, out_w, out_h, progress)
        if edit and edit.effects.get("progress"):
            overlays.draw_progress(img, (k + 1) / out_fps / total)
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2YUV_I420).tobytes()

    return compose


def _pump(cmd, work, out_path, in_path, compose, skip, src_fps, n_total, edit, waits=(), meme_screen=None) -> None:
    """Read frames in order, compose them on a few threads, feed ffmpeg in order.

    ``waits``: (source time, number of frames, meme pictures) where the video stands still and the meme is shown.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")
    err_path = os.path.join(work, "ffmpeg.log")
    workers = max(2, min(4, (os.cpu_count() or 4) // 2))
    cv_threads = cv2.getNumThreads()
    ok_run = False
    proc = None
    with open(err_path, "wb") as err_file:
        try:
            cv2.setNumThreads(1)                                  # the parallelism is ours: one frame per thread
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file, cwd=work)
            pending: "deque" = deque()
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compose") as pool:
                i = k = 0
                waiting = deque(waits)
                last = None                                        # the last picture used: (frame, source frame number)

                def send(job) -> None:
                    pending.append(job)
                    while len(pending) > workers * 2:
                        proc.stdin.write(pending.popleft().result())

                def show_meme() -> None:
                    nonlocal k
                    _, count, pictures = waiting.popleft()
                    for j in range(count):
                        if pictures:
                            send(pool.submit(meme_screen, pictures[min(j, len(pictures) - 1)], j / max(1, count - 1), k))
                        elif last is not None:                     # unreadable meme: hold the picture, stay in step with the sound
                            send(pool.submit(compose, last[0], last[1], k))
                        else:
                            send(pool.submit(meme_screen, np.zeros((16, 16, 3), np.uint8), 0.0, k))
                        k += 1

                try:
                    while True:
                        if not cap.grab():
                            break
                        used = i % skip == 0
                        while used and waiting and i / src_fps >= waiting[0][0] - 1e-6:
                            show_meme()
                        if used and (edit is None or edit.timeline.kept(i / src_fps)):
                            ok, frame = cap.retrieve()             # converting a frame costs: only the ones that are used
                            if not ok:
                                break
                            last = (frame, i)
                            send(pool.submit(compose, frame, i, k))
                            k += 1
                        elif edit is not None and i / src_fps >= edit.timeline.keep[-1][1]:
                            break
                        i += 1
                        if i % 30 == 0:
                            control.check()
                            control.report("render", min(1.0, i / n_total), "encoding")
                    while waiting:                                 # a closing meme, or a source that ran a frame short
                        show_meme()
                    while pending:
                        proc.stdin.write(pending.popleft().result())
                finally:
                    for f in pending:
                        f.cancel()
            ok_run = True
        except (BrokenPipeError, OSError) as e:
            raise EncodeFailed(f"ffmpeg stopped accepting frames ({e.__class__.__name__}): {_tail(err_path)}") from None
        finally:
            cap.release()
            cv2.setNumThreads(cv_threads)
            if proc is not None:
                if not ok_run and proc.poll() is None:            # error or Cancel: stop ffmpeg, leave no half-written file
                    proc.kill()
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                rc = proc.wait()
                if not ok_run:
                    _remove_quietly(out_path)
    if rc != 0:
        _remove_quietly(out_path)
        raise EncodeFailed(f"ffmpeg encode failed (exit {rc}): {_tail(err_path)}")


def _tail(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return f.read()[-600:].decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


@dataclass
class RenderOptions:
    """How a clip should look. Defaults give the classic Shorts look without any text or editing."""
    aspect_ratio: str = "9:16"
    layout: str = "auto"                  # auto | split | track | fit
    captions: bool = False
    caption_style: str = "pop"            # pop | classic | yellow | box | karaoke
    title_card: bool = False              # the clip's title, big, for its first seconds
    facecam_bg: str = ""                  # colour behind a cut-out streamer ("" = the saved setting)
    watermark: str = ""                   # small text in a corner, e.g. "@mychannel"
    normalize_audio: bool = True          # -14 LUFS so clips are not quieter than everything else in the feed
    edit: str = "off"                     # automatic editing: off | clean | viral (editing.py)
    effects: Optional[Dict[str, bool]] = None     # single effects switched on/off on top of the preset

    def validate(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"Unknown layout {self.layout!r}. Use one of {', '.join(LAYOUTS)}.")
        if self.facecam_bg and self.facecam_bg not in cutout.BACKGROUNDS:
            raise ValueError(f"Unknown background {self.facecam_bg!r}. Use one of {', '.join(cutout.BACKGROUNDS)}.")
        if self.caption_style not in STYLES:
            raise ValueError(f"Unknown caption style {self.caption_style!r}. Use one of {', '.join(STYLES)}.")
        if self.edit not in editing.PRESET_NAMES:
            raise ValueError(f"Unknown editing style {self.edit!r}. Use one of {', '.join(editing.PRESET_NAMES)}.")

    def active_effects(self) -> Dict[str, bool]:
        return editing.effects_for(self.edit, self.effects) if self.edit != "off" or self.effects else editing.effects_for("off")


def clip_words(segments: Optional[List[Dict]], start: float, end: float) -> List[Tuple[float, float, str]]:
    """The spoken words of [start, end], times relative to ``start``."""
    out = []
    for seg in segments or []:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        for a, b, text in _segment_words(seg):
            if a >= start - 0.2 and a < end - 0.05:
                out.append((max(0.0, a - start), min(end - start, b - start), text))
    return sorted(out)


def clip_reactions(segments: Optional[List[Dict]], start: float, end: float) -> List[Tuple[float, str]]:
    """Laughter / applause notes of the captions inside [start, end], times relative to ``start``."""
    out: List[Tuple[float, str]] = []
    for seg in segments or []:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        out += [(t - start, kind) for t, kind in split_annotations(_raw_words(seg))[1] if start <= t < end]
    return out


def _segments_from_words(words, pause: float = 0.7) -> List[Dict]:
    """Caption segments from (start, end, text) words: a card never spans a pause."""
    groups: List[List] = []
    for word in words:
        if not groups or word[0] - editing.spoken_span(groups[-1][-1])[1] > pause:
            groups.append([])
        groups[-1].append(word)
    return [{"start": g[0][0], "end": g[-1][1], "text": " ".join(w[2] for w in g),
             "words": [{"start": a, "end": b, "text": t} for a, b, t in g]} for g in groups]


def _plan_framing(an: Dict, options: "RenderOptions", out_w: int, out_h: int, thumbs, path: str = "") -> Dict:
    ratio = _ratio(options.aspect_ratio)
    layout = options.layout
    if layout in ("auto", "cutout") and ratio < 1.0 and path:
        control.report("render", 0.0, "looking for a cut-out streamer")
        plan = cutout.plan(path, an, out_w, out_h, options.facecam_bg or config.facecam_background(), force=layout == "cutout")
        if plan:
            return plan
    layout = "auto" if layout == "cutout" else layout
    plan = _plan(an, layout, ratio, out_w, out_h)
    if layout == "auto" and plan["layout"] != "split" and ratio < 1.0:
        plan = framing.smart_plan(an, thumbs, ratio, out_h)      # shows, podcasts, vlogs: framed shot by shot
    return plan


def _reframe(in_path: str, out_path: str, options: RenderOptions, clip_start: float, clip_end: float,
             transcript_segments: Optional[List[Dict]], title: str, hints: Optional[Dict] = None) -> Dict:
    options.validate()
    out_w, out_h = _out_size(options.aspect_ratio)
    control.report("render", 0.0, "looking at the clip")
    thumbs_box: Dict = {}
    want_smart = options.layout in ("auto", "cutout") and _ratio(options.aspect_ratio) < 1.0
    side = None
    if want_smart:                                              # tiny pictures of every frame for cut detection, meanwhile
        side = threading.Thread(target=lambda: thumbs_box.update(t=framing.thumbnails(in_path)), daemon=True)
        side.start()
    an = _analyse(in_path)
    if side:
        side.join()
    plan = _plan_framing(an, options, out_w, out_h, thumbs_box.get("t"), in_path)

    effects = options.active_effects()
    edit = None
    if any(effects.values()):
        duration = min(clip_end - clip_start, an["n_frames"] / an["fps"])
        words = clip_words(transcript_segments, clip_start, clip_start + duration)
        out_fps = an["fps"] / max(1, int(round(an["fps"] / OUT_FPS_MAX)))
        control.report("render", 0.0, "planning the edit")
        db = editing.audio_levels(in_path) if _has_audio(in_path) else None
        edit = editing.build_plan(duration, out_fps, words, db, effects, hints, title,
                                  reactions=clip_reactions(transcript_segments, clip_start, clip_start + duration),
                                  library=memes.scan() if effects.get("memes") else ())

    ass_text = None
    wants_captions = bool(options.captions or (edit and edit.effects.get("captions")))
    hook = edit.hook if edit else ""
    cta = edit.cta if edit else ""
    wants_text = bool((wants_captions and transcript_segments) or options.title_card or options.watermark or hook or cta)
    if wants_text:
        if not _captions_supported():
            print("[clip/local]   text overlays skipped: this ffmpeg has no libass/'ass' filter", flush=True)
        else:
            if edit:                                             # captions follow the cut timeline
                segments = _segments_from_words(edit.words)
                start, end = 0.0, edit.duration
            else:
                segments, start, end = transcript_segments or [], clip_start, clip_end
            title_top = (plan["top_h"] + round(out_h * 0.02)) if plan["layout"] == "split" else None
            margin_bottom, cta_y = None, 0.16 if plan["layout"] in ("smart", "fit") else 0.40     # not over the faces of a stage shot
            if plan["layout"] == "cutout":                       # headline above the streamer, words just under the divider
                title_top, cta_y = round(out_h * (0.08 if options.watermark else 0.025)), 0.66      # under the channel badge
                margin_bottom = max(0.1, 1.0 - plan["top_h"] / out_h - 0.1)
            if plan["layout"] == "smart" and plan["shots"][0]["kind"] == "stack":
                title_top = round(out_h * 0.47)                  # on the dividing line, not on the upper face
            ass_text = build_ass(
                segments, start, end, out_w, out_h,
                style=options.caption_style, captions=bool(wants_captions and segments),
                title=title if options.title_card and not hook else "",
                title_top=title_top, watermark=options.watermark,
                keywords=edit.keywords if edit else (), hook=hook, cta=cta,
                cta_y=cta_y, margin_bottom=margin_bottom,
            )
            if ass_text is None and wants_captions:
                print("[clip/local]   no speech in this clip: nothing to caption", flush=True)
    extras = [x for x, on in (("captions", wants_captions and ass_text), ("title", options.title_card and ass_text),
                              ("watermark", options.watermark and ass_text)) if on]
    print(f"[clip/local]   layout={plan['layout']} ({plan['why']}) -> {out_w}x{out_h}"
          f"{' + ' + ', '.join(extras) if extras else ''}", flush=True)
    info = {"layout": plan["layout"], "why": plan["why"], "size": (out_w, out_h), "has_audio": _has_audio(in_path)}
    if edit:
        summary = edit.summary()
        info.update(edit=summary, duration=edit.duration)
        print(f"[clip/local]   edit: {summary['cuts']} cut(s) removing {summary['removed_seconds']}s, {summary['zooms']} zoom(s), "
              f"{summary['popups']} popup(s), {summary['memes']} meme(s), {summary['sounds']} sound(s)", flush=True)
        for cue in edit.memes:
            print(f"[clip/local]   meme at {cue.t:.1f}s for {cue.dur:.1f}s ({cue.meme.how}): {cue.meme.name}", flush=True)
    else:
        info["duration"] = min(clip_end - clip_start, an["n_frames"] / an["fps"])
    info["encoder"] = _render(in_path, out_path, an, plan, out_w, out_h, ass_text, options.normalize_audio, edit)
    return info


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def render_clip(
    media_path: str,
    out_path: str,
    clip_start: float,
    clip_end: float,
    options: Optional[RenderOptions] = None,
    *,
    transcript_segments: Optional[List[Dict]] = None,
    title: str = "",
    precut: bool = False,
    hints: Optional[Dict] = None,
    info: Optional[Dict] = None,
) -> str:
    """Render one clip to ``out_path`` and return it.

    ``clip_start``/``clip_end`` are positions in the original video (they also place the captions).
    ``precut=True``: ``media_path`` is a section that already starts at ``clip_start`` (what the
    streaming fetch produces). Otherwise it is the whole local file and the clip is cut out of it.
    ``hints``: the AI's edit suggestions ({hook, popups, keywords}); ``info``: filled with what was done.
    """
    options = options or RenderOptions()
    options.validate()
    cut_path = None
    work = media_path
    try:
        if not precut:
            cut_path = out_path + ".cut.mp4"
            _cut_subclip(media_path, clip_start, clip_end, cut_path)
            work = cut_path
        result = _reframe(work, out_path, options, clip_start, clip_end, transcript_segments, title, hints)
        if info is not None:
            info.update(result)
    finally:
        if cut_path:
            _remove_quietly(cut_path)
    return out_path


def slug(title: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(title).lower()).strip("-")[:limit].strip("-")


def clip_filename(label: int, title: str) -> str:
    """``short_07_the-plane-crash.mp4``: the same candidate always maps to the same file name."""
    s = slug(title)
    return f"short_{label:02d}{'_' + s if s else ''}.mp4"
