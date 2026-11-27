"""How a wide picture becomes a vertical one: crop geometry, and "smart" framing that decides **per camera shot**.

A show, a podcast or a panel changes the camera every few seconds. One crop that follows "a face" for the whole clip then
slides across the stage after every cut, or cuts people off. Here every shot gets its own, steady framing:

  crop       one speaker (or a tight group): a full-height 9:16 window centred on them
  group      several people spread across the stage: just the part of the picture with the performers, as wide as the
             screen, over a blurred copy of the picture (everyone stays in, and bigger than a plain letterbox)
  stack      two people far apart, both close enough to the camera: one above the other
  fit        nobody to follow: the whole picture over a blurred background

"Performers" are told apart from bystanders (guards, crew standing at the edges) by position and by how much they move.
"""
import subprocess
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

THUMB_W, THUMB_H = 96, 54
CUT_MIN = 18.0                  # mean absolute change (0..255) of the tiny picture that can be a camera cut ...
CUT_RATIO = 4.0                 # ... if it is also this many times the usual change (busy footage moves a lot by itself)
MIN_SHOT_SECONDS = 0.8          # shorter "shots" are flashes; they stay with the one before
GROUP_TOP = 0.40                # where a group block sits: 0 = top of the screen, 1 = bottom
GROUP_ZOOM = 0.35               # group shots take only this share of a punch-in: people at the edges must not be cut off


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------------------------- #
# Pixels
# --------------------------------------------------------------------------------------------- #
def zoom_rect(x0: float, y0: float, w: float, h: float, z: float, fx: Optional[float] = None, fy: Optional[float] = None):
    """The part of (x0, y0, w, h) seen at zoom ``z`` (>1), centred as near to (fx, fy) as the rectangle allows."""
    if z <= 1.001:
        return x0, y0, w, h
    nw, nh = w / z, h / z
    fx = x0 + w / 2 if fx is None else fx
    fy = y0 + h / 2 if fy is None else fy
    return clamp(fx - nw / 2, x0, x0 + w - nw), clamp(fy - nh / 2, y0, y0 + h - nh), nw, nh


def warp(frame: "np.ndarray", x0: float, y0: float, cw: float, ch: float, out_w: int, out_h: int) -> "np.ndarray":
    """Crop (x0, y0, cw, ch) and scale to out_w x out_h with sub-pixel precision.

    Same scale factor on both axes whenever the crop has the output's aspect ratio, so nothing is stretched. Sub-pixel
    translation keeps slow pans smooth instead of stepping a whole source pixel at a time. Linear sampling: cubic cost
    twice as long per frame and the difference does not survive the video encoder.
    """
    sx, sy = out_w / cw, out_h / ch
    if sx < 1.0 or sy < 1.0:       # down-scaling: area filter
        xi, yi = int(round(x0)), int(round(y0))
        sub = frame[yi:yi + int(round(ch)), xi:xi + int(round(cw))]
        return cv2.resize(sub, (out_w, out_h), interpolation=cv2.INTER_AREA)
    m = np.float32([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]])
    return cv2.warpAffine(frame, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def blurred_background(frame: "np.ndarray", roi, out_w: int, out_h: int, dim: float = 0.5) -> "np.ndarray":
    """A soft, dark copy of the picture that fills the screen. Blurred at 1/12 size: cheap."""
    rx, ry, rw, rh = roi
    want = out_w / out_h
    cw, ch = (rh * want, rh) if rw / rh > want else (rw, rw / want)
    x0, y0 = int(rx + (rw - cw) / 2), int(ry + (rh - ch) / 2)
    src = frame[y0:y0 + int(ch), x0:x0 + int(cw)]
    small = cv2.resize(src, (max(2, out_w // 12), max(2, out_h // 12)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    small = cv2.convertScaleAbs(small, alpha=dim)
    return cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def compose_block(frame: "np.ndarray", region, roi, out_w: int, out_h: int, top_frac: float = GROUP_TOP) -> "np.ndarray":
    """``region`` of the picture, as large as it fits on the screen, undistorted, over a blurred background."""
    x0, y0, w, h = region
    canvas = blurred_background(frame, roi, out_w, out_h)
    k = min(out_w / w, out_h / h)
    fw, fh = int(round(w * k)), int(round(h * k))
    fg = warp(frame, x0, y0, w, h, fw, fh)
    top, left = int(round((out_h - fh) * top_frac)), (out_w - fw) // 2
    canvas[top:top + fh, left:left + fw] = fg
    return canvas


# --------------------------------------------------------------------------------------------- #
# Camera cuts
# --------------------------------------------------------------------------------------------- #
def thumbnails(path: str) -> "np.ndarray":
    """Every frame as a tiny grey picture (ffmpeg decodes and shrinks them far faster than a Python loop)."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-vf", f"scale={THUMB_W}:{THUMB_H},format=gray",
                           "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True)
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = data.size // (THUMB_W * THUMB_H)
    return data[: n * THUMB_W * THUMB_H].reshape(n, THUMB_H, THUMB_W)


def find_cuts(thumbs: "np.ndarray", fps: float) -> List[int]:
    """Frame numbers where a new camera shot starts (never closer together than MIN_SHOT_SECONDS)."""
    if len(thumbs) < 3:
        return []
    t = thumbs.astype(np.int16)
    diff = np.concatenate([[0.0], np.abs(t[1:] - t[:-1]).mean(axis=(1, 2))])
    usual = float(np.median(diff[1:]))
    limit = max(CUT_MIN, CUT_RATIO * usual)
    gap = int(MIN_SHOT_SECONDS * fps)
    cuts: List[int] = []
    for i in range(1, len(diff)):
        if diff[i] > limit and diff[i] >= diff[max(1, i - 2):i + 3].max():
            if i >= gap and (not cuts or i - cuts[-1] >= gap):
                cuts.append(i)
    return cuts


def shot_ranges(cuts: List[int], n_frames: int, fps: float) -> List[Tuple[int, int]]:
    bounds = [0] + [c for c in cuts if 0 < c < n_frames] + [n_frames]
    shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    if len(shots) > 1 and shots[-1][1] - shots[-1][0] < MIN_SHOT_SECONDS * fps:       # a stub at the end
        shots[-2:] = [(shots[-2][0], shots[-1][1])]
    return shots


# --------------------------------------------------------------------------------------------- #
# Who is in a shot
# --------------------------------------------------------------------------------------------- #
def tracks_in(sidx: List[int], boxes: List[list], start: int, end: int) -> List[Dict]:
    """Faces of one shot grouped by position. Each track: presence (share of samples), median box, hits {sample frame: box}."""
    keys = [(i, b) for i, b in zip(sidx, boxes) if start <= i < end]
    if not keys:
        return []
    tracks: List[Dict] = []
    for frame_no, found in keys:
        taken: set = set()
        for (x, y, w, h) in sorted(found, key=lambda b: -b[2] * b[3]):
            cx, cy = x + w / 2.0, y + h / 2.0
            best, best_d = None, 1e18
            for k, t in enumerate(tracks):
                d = ((cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2) ** 0.5
                if k not in taken and d < 1.4 * max(w, t["w"]) and d < best_d:
                    best, best_d = k, d
            if best is None:
                tracks.append({"cx": cx, "cy": cy, "w": float(w), "hits": {frame_no: (x, y, w, h)}})
                taken.add(len(tracks) - 1)
            else:
                t = tracks[best]
                t["hits"][frame_no] = (x, y, w, h)
                t.update(cx=cx, cy=cy, w=float(w))
                taken.add(best)
    for t in tracks:
        arr = np.array(list(t["hits"].values()), dtype=float)
        t["box"] = tuple(float(v) for v in np.median(arr, axis=0))
        t["presence"] = len(t["hits"]) / len(keys)
        t["cx"], t["cy"] = t["box"][0] + t["box"][2] / 2, t["box"][1] + t["box"][3] / 2
    return tracks


def body_motion(thumbs: Optional["np.ndarray"], box, size, start: int, end: int) -> float:
    """How much the area of the body under a face changes: performers gesture, guards stand."""
    if thumbs is None or len(thumbs) < 2:
        return 0.0
    w_img, h_img = size
    x, y, w, h = box
    x0, x1 = int(max(0, (x - w) * THUMB_W / w_img)), int(min(THUMB_W, (x + 2 * w) * THUMB_W / w_img) + 1)
    y0, y1 = int(max(0, (y + h) * THUMB_H / h_img)), int(min(THUMB_H, (y + 5 * h) * THUMB_H / h_img) + 1)
    part = thumbs[start:min(end, len(thumbs)):3, y0:y1, x0:x1].astype(np.int16)
    if len(part) < 2 or part.shape[1] == 0 or part.shape[2] == 0:
        return 0.0
    return float(np.abs(part[1:] - part[:-1]).mean())


def performers(tracks: List[Dict], roi) -> List[Dict]:
    """The people the shot is about: seen often enough, near enough, and (in a crowd) moving and not at the very edge."""
    seen = [t for t in tracks if t["presence"] >= 0.4]
    if not seen:
        return []
    tallest = max(t["box"][3] for t in seen)
    seen = [t for t in seen if t["box"][3] >= 0.55 * tallest]
    rx, _, rw, _ = roi
    if len(seen) > 2:                                         # trim bystanders, but never below two people
        edge = [t for t in seen if not (rx + 0.12 * rw <= t["cx"] <= rx + 0.88 * rw)]
        if len(seen) - len(edge) >= 2:
            seen = [t for t in seen if t not in edge]
    if len(seen) > 2:
        motions = sorted(t.get("motion", 0.0) for t in seen)
        typical = motions[len(motions) // 2]
        moving = [t for t in seen if t.get("motion", 0.0) >= 0.3 * typical]
        if len(moving) >= 2:
            seen = moving
    return sorted(seen, key=lambda t: t["cx"])


# --------------------------------------------------------------------------------------------- #
# What to do with a shot
# --------------------------------------------------------------------------------------------- #
def _crop_size(roi, ratio: float) -> Tuple[float, float]:
    _, _, rw, rh = roi
    ch = float(rh)
    cw = ch * ratio
    if cw > rw:
        cw, ch = float(rw), rw / ratio
    return cw, ch


def _crop_around(center_x: float, roi, ratio: float) -> Tuple[float, float, float, float]:
    rx, ry, rw, rh = roi
    cw, ch = _crop_size(roi, ratio)
    return clamp(center_x - cw / 2, rx, rx + rw - cw), ry + (rh - ch) / 2, cw, ch


def plan_shot(people: List[Dict], roi, ratio: float, out_h: int) -> Dict:
    """The framing of one shot, given its performers (left to right). Geometry is in source pixels."""
    rx, ry, rw, rh = roi
    if not people:
        return {"kind": "fit"}
    cw, ch = _crop_size(roi, ratio)
    if len(people) == 1:
        p = people[0]
        return {"kind": "crop", "rect": _crop_around(p["cx"], roi, ratio), "focus": (p["cx"], p["cy"]), "people": 1}

    left = min(p["box"][0] - 0.6 * p["box"][2] for p in people)
    right = max(p["box"][0] + 1.6 * p["box"][2] for p in people)
    biggest = max(people, key=lambda p: p["box"][3])
    if right - left <= cw:
        return {"kind": "crop", "rect": _crop_around((left + right) / 2, roi, ratio), "focus": (biggest["cx"], biggest["cy"]), "people": len(people)}

    if len(people) == 2 and min(p["box"][3] for p in people) >= 0.15 * rh and ratio < 0.8:
        panels = []
        panel_ratio = ratio * 2                                # each panel is half the screen high
        for p in people:
            ph = clamp(p["box"][3] * 3.6, 0.4 * rh, rh)
            pw = ph * panel_ratio
            if pw > rw:
                pw, ph = float(rw), rw / panel_ratio
            panels.append((clamp(p["cx"] - pw / 2, rx, rx + rw - pw), clamp(p["cy"] - ph * 0.4, ry, ry + rh - ph), pw, ph))
        return {"kind": "stack", "panels": panels, "focus": (biggest["cx"], biggest["cy"]), "people": 2}

    # group: only the stretch of the stage with the performers, with room for their bodies and gestures
    left = min(p["box"][0] - 2.0 * p["box"][2] for p in people)
    right = max(p["box"][0] + 3.0 * p["box"][2] for p in people)
    gw = clamp(right - left, min(rw, rh * 0.9), rw)             # never narrower than almost square: it would just be a crop
    gx = clamp((left + right) / 2 - gw / 2, rx, rx + rw - gw)
    return {"kind": "group", "rect": (gx, ry, gw, float(rh)), "focus": (biggest["cx"], biggest["cy"]), "people": len(people)}


def smart_plan(an: Dict, thumbs: Optional["np.ndarray"], ratio: float, out_h: int) -> Dict:
    """Per-shot framing for the whole clip: {"layout": "smart", "shots": [{start, end, kind, ...}], "why"}."""
    roi, size, fps = an["roi"], an["size"], an["fps"]
    n = int(an["n_frames"])
    cuts = find_cuts(thumbs, fps) if thumbs is not None else []
    shots = []
    for start, end in shot_ranges(cuts, n, fps):
        tracks = tracks_in(an["sidx"], an["boxes"], start, end)
        for t in tracks:
            t["motion"] = body_motion(thumbs, t["box"], size, start, end)
        shot = plan_shot(performers(tracks, roi), roi, ratio, out_h)
        shot.update(start=start, end=end)
        # a talking head who walks: follow slowly instead of one fixed window (only inside this shot, never across a cut)
        if shot["kind"] == "crop" and shot.get("people") == 1:
            follow = _follow(an, shot, ratio)
            if follow is not None:
                shot["x0s"] = follow
        shots.append(shot)
    if shots and all(s["kind"] == "fit" for s in shots):
        return {"layout": "fit", "why": "nobody to follow"}
    kinds = [s["kind"] for s in shots]
    summary = ", ".join(f"{kinds.count(k)} {k}" for k in ("crop", "group", "stack", "fit") if k in kinds)
    return {"layout": "smart", "shots": shots, "why": f"{len(shots)} camera shot(s): {summary}"}


def _follow(an: Dict, shot: Dict, ratio: float) -> Optional["np.ndarray"]:
    """Left edge per source frame of a window following one person, or None when they barely move."""
    start, end = shot["start"], shot["end"]
    rx, ry, rw, rh = an["roi"]
    cw = shot["rect"][2]
    fx = shot["focus"][0]
    keys, centers = [], []
    for i, found in zip(an["sidx"], an["boxes"]):
        if start <= i < end and found:
            x, y, w, h = min(found, key=lambda b: abs(b[0] + b[2] / 2 - (centers[-1] if centers else fx)))
            keys.append(i)
            centers.append(x + w / 2)
    if len(centers) < 4 or float(np.ptp(np.array(centers))) < 0.18 * cw:
        return None
    arr = np.array(centers, dtype=float)
    pad = np.pad(arr, 2, mode="edge")
    arr = np.array([np.median(pad[k:k + 5]) for k in range(len(arr))])
    step_max = 0.3 * cw * (keys[1] - keys[0]) / max(an["fps"], 1.0)
    cur, path = arr[0], []
    for c in arr:
        err = c - cur
        if abs(err) > 0.08 * cw:
            cur += float(np.clip(err - np.sign(err) * 0.08 * cw, -step_max, step_max))
        path.append(cur)
    per_frame = np.interp(np.arange(start, end), keys, path)
    return np.clip(per_frame - cw / 2, rx, rx + rw - cw)


def shot_for(plan: Dict, frame_no: int) -> Dict:
    shots = plan["shots"]
    for s in shots:
        if s["start"] <= frame_no < s["end"]:
            return s
    return shots[-1]


def compose_smart(frame: "np.ndarray", plan: Dict, frame_no: int, roi, out_w: int, out_h: int, zoom: float = 1.0) -> "np.ndarray":
    shot = shot_for(plan, frame_no)
    kind = shot["kind"]
    fx, fy = shot.get("focus", (None, None))
    if kind == "crop":
        x0, y0, w, h = shot["rect"]
        if "x0s" in shot:
            x0 = float(shot["x0s"][min(max(frame_no - shot["start"], 0), len(shot["x0s"]) - 1)])
        return warp(frame, *zoom_rect(x0, y0, w, h, zoom, fx, fy), out_w, out_h)
    if kind == "group":
        return compose_block(frame, zoom_rect(*shot["rect"], 1.0 + (zoom - 1.0) * GROUP_ZOOM), roi, out_w, out_h)
    if kind == "stack":
        half = out_h // 2
        out = np.empty((out_h, out_w, 3), dtype=np.uint8)
        top, bottom = shot["panels"]
        out[:half] = warp(frame, *zoom_rect(*top, zoom), out_w, half)
        out[half:] = warp(frame, *zoom_rect(*bottom, zoom), out_w, out_h - half)
        out[half - 3:half + 3] = 0
        return out
    return compose_block(frame, zoom_rect(*roi, 1.0 + (zoom - 1.0) * GROUP_ZOOM), roi, out_w, out_h, top_frac=0.5)
