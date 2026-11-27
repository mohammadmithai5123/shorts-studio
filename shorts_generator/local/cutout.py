"""A streamer cut out over the game (green screen: no box around them).

Such a stream is 16:9 gameplay with the person (and often their chair) laid over it near the bottom. A vertical crop on
their face throws most of the game away. This layout instead puts

  * the streamer, big, on a plain colour at the top (the person only: a half-cut chair looks cheap),
  * the game, wide, below, with the small copy of the streamer smoothed away.

Telling who is who:
  * the person: OpenCV's PP-HumanSeg model (assets/models, Apache-2.0);
  * the chair and anything else laid over the game: pixels near the person that never change while the game around does.
    That is the evidence that this is a cut-out, and it is part of what gets smoothed away below.
"""
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from . import framing

MODEL = Path(__file__).resolve().parents[2] / "assets" / "models" / "human_segmentation_pphumanseg_2023mar.onnx"
BACKGROUNDS = {"white": (245, 245, 245), "green": (64, 177, 0), "black": (18, 18, 18), "blue": (190, 100, 25),
               "purple": (150, 40, 100), "pink": (160, 90, 240)}                       # BGR
TOP_SHARE = 0.43              # of the screen height: the streamer's panel
PERSON_SHARE = 0.84           # of that panel: how tall the cut-out is drawn (the rest is room for the headline)
GAME_ASPECT = 4 / 3           # the game keeps three quarters of its width
STILL_STD = 9.0               # a pixel that varies less than this over the clip is "laid over" the picture
MIN_PRESENCE = 0.6            # share of the clip in which the streamer must be found
SAMPLES = 24                  # frames per position used to learn the overlay

_local = threading.local()


def available() -> bool:
    return MODEL.is_file() and hasattr(cv2, "dnn")


def person_mask(image: "np.ndarray", full: bool = True) -> "np.ndarray":
    """Per pixel, how sure the model is that it shows a person (0..1). One network per thread: OpenCV's is not shareable.

    ``full=False`` returns the model's own 192x192 answer, for callers that scale it themselves.
    """
    net = getattr(_local, "net", None)
    if net is None:
        net = _local.net = cv2.dnn.readNet(str(MODEL))
    small = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
    blob = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5) / 0.5
    net.setInput(cv2.dnn.blobFromImage(blob))
    out = net.forward()[0]
    if out.min() < 0 or out.max() > 1.001:                    # raw scores: turn them into probabilities
        e = np.exp(out - out.max(axis=0, keepdims=True))
        out = e / e.sum(axis=0, keepdims=True)
    return cv2.resize(out[1], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR) if full else out[1]


# --------------------------------------------------------------------------------------------- #
# Where is the streamer, and when
# --------------------------------------------------------------------------------------------- #
def _low_face(boxes: Sequence, roi) -> Optional[Tuple[int, int, int, int]]:
    """The face that could be a cut-out streamer's: smallish, in the lower part of the picture."""
    rx, ry, rw, rh = roi
    low = [b for b in boxes if b[1] + b[3] / 2 > ry + 0.62 * rh and 0.06 * rh <= b[3] <= 0.26 * rh]
    return max(low, key=lambda b: b[2] * b[3]) if low else None


def positions(an: Dict) -> List[Dict]:
    """The clip as stretches [{start, end, face}] in source frames; ``face`` is None while the streamer is not there.

    Streamers move their cut-out between scenes (left to right of the screen), so each place gets its own stretch.
    Gaps and blips shorter than two seconds are absorbed by their neighbours.
    """
    roi, fps, n = an["roi"], max(an["fps"], 1.0), int(an["n_frames"])
    runs: List[Dict] = []
    for frame_no, boxes in zip(an["sidx"], an["boxes"]):
        face = _low_face(boxes, roi)
        cx = None if face is None else face[0] + face[2] / 2
        last = runs[-1] if runs else None
        same = last is not None and ((cx is None and last["cx"] is None)
                                     or (cx is not None and last["cx"] is not None and abs(cx - last["cx"]) < 0.12 * roi[2]))
        if same:
            if face is not None:
                last["faces"].append(face)
                last["cx"] = 0.8 * last["cx"] + 0.2 * cx
        else:
            runs.append({"start": frame_no, "cx": cx, "faces": [face] if face is not None else []})
    for run, following in zip(runs, runs[1:]):
        run["end"] = following["start"]
    if runs:
        runs[0]["start"], runs[-1]["end"] = 0, n

    long_enough = 2.0 * fps
    keep = [r for r in runs if r["end"] - r["start"] >= long_enough] or runs
    merged: List[Dict] = []
    for run in keep:
        last = merged[-1] if merged else None
        if last and (last["cx"] is None) == (run["cx"] is None) and (run["cx"] is None or abs(run["cx"] - last["cx"]) < 0.12 * roi[2]):
            last["end"] = run["end"]
            last["faces"] = last["faces"] + run["faces"]
        else:
            if last:
                last["end"] = run["start"]
            merged.append(dict(run))
    if merged:
        merged[0]["start"], merged[-1]["end"] = 0, n
    return [{"start": r["start"], "end": r["end"],
             "face": tuple(float(v) for v in np.median(np.array(r["faces"], dtype=float), axis=0)) if r["faces"] else None}
            for r in merged]


def box_around(face, roi) -> Tuple[int, int, int, int]:
    """The part of the picture that holds the whole overlay: shoulders, chair, down to the bottom edge."""
    rx, ry, rw, rh = roi
    x, y, w, h = face
    cx = x + w / 2
    x0, x1 = int(max(rx, cx - 2.5 * w)), int(min(rx + rw, cx + 2.5 * w))
    y0, y1 = int(max(ry, y - 0.8 * h)), int(ry + rh)
    return x0, y0, x1, y1


def _read_frames(path: str, numbers: Sequence[int]) -> Dict[int, "np.ndarray"]:
    wanted, last = set(numbers), max(numbers, default=-1)
    out: Dict[int, "np.ndarray"] = {}
    cap = cv2.VideoCapture(path)
    try:
        for i in range(last + 1):
            if not cap.grab():
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    out[i] = frame
    finally:
        cap.release()
    return out


# --------------------------------------------------------------------------------------------- #
# What is laid over the game around one position
# --------------------------------------------------------------------------------------------- #
def learn_overlay(crops: List["np.ndarray"], face_w: float) -> Dict:
    """From several moments of the same crop: where to erase (person and chair) and evidence that it is a cut-out."""
    std = np.stack(crops).astype(np.float32).std(axis=0).mean(axis=2)
    seen = np.max(np.stack([person_mask(c) for c in crops[::4]]), axis=0)
    person = (seen > 0.5).astype(np.uint8)
    k = max(3, int(face_w * 0.9)) | 1
    near = cv2.dilate(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    still = ((std < STILL_STD) & (near > 0)).astype(np.uint8)
    still = cv2.morphologyEx(still, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels = cv2.connectedComponents(still)
    touching = [i for i in range(1, count) if person[labels == i].any()]
    still = np.isin(labels, touching).astype(np.uint8)

    shape = np.maximum(person, still)
    info = {"bottom": bool(person[-10:].any()), "boxiness": 1.0, "changing": 0.0}
    erase = np.zeros_like(shape)
    if shape.any():
        x, y, w, h = cv2.boundingRect(shape)
        info["boxiness"] = float(shape.sum()) / max(1, w * h)          # a webcam box is a filled rectangle, a cut-out is not
        wide = max(3, int(face_w)) | 1
        ring = cv2.dilate(shape, np.ones((wide, wide), np.uint8)) - cv2.dilate(shape, np.ones((9, 9), np.uint8))
        if ring.any():
            info["changing"] = float((std[ring > 0] > STILL_STD * 1.3).mean())     # does the game move right up to them?
        points = cv2.findNonZero(shape)
        cv2.fillConvexPoly(erase, cv2.convexHull(points), 1)
        grow = max(3, int(face_w * 0.22)) | 1
        erase = cv2.dilate(erase, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return {"erase": cv2.GaussianBlur(erase.astype(np.float32), (0, 0), max(1.0, face_w * 0.09)), "info": info}


def is_cutout(info: Dict) -> bool:
    return info["bottom"] and info["changing"] >= 0.3 and info["boxiness"] <= 0.86


# --------------------------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------------------------- #
def plan(path: str, an: Dict, out_w: int, out_h: int, background: str = "white", force: bool = False) -> Optional[Dict]:
    """{"layout": "cutout", ...} when this clip is a cut-out streamer over moving footage, else None.

    ``force``: the person asked for this layout, so only a streamer to follow is needed, not the evidence.
    """
    if not available():
        return None
    runs = positions(an)
    total = max(1, int(an["n_frames"]))
    present = sum(r["end"] - r["start"] for r in runs if r["face"]) / total
    if present < (0.2 if force else MIN_PRESENCE):
        return None

    wanted: Dict[int, List[int]] = {}
    for i, run in enumerate(runs):
        if run["face"]:
            span = run["end"] - run["start"]
            wanted[i] = sorted({run["start"] + int(span * (k + 0.5) / SAMPLES) for k in range(SAMPLES)})
    frames = _read_frames(path, [n for numbers in wanted.values() for n in numbers])

    proof = 0
    for i, numbers in wanted.items():
        run = runs[i]
        x0, y0, x1, y1 = box_around(run["face"], an["roi"])
        crops = [frames[n][y0:y1, x0:x1] for n in numbers if n in frames]
        if len(crops) < 4:
            run["face"] = None
            continue
        run["box"] = (x0, y0, x1, y1)
        run.update(learn_overlay(crops, run["face"][2]))
        if is_cutout(run["info"]):
            proof += run["end"] - run["start"]
    if not force and proof < 0.25 * present * total:
        return None
    if not any(r.get("box") for r in runs):
        return None

    rx, ry, rw, rh = an["roi"]
    top_h = int(round(out_h * TOP_SHARE / 2) * 2)
    gw = min(float(rw), rh * GAME_ASPECT)
    gh = gw / GAME_ASPECT
    game_h = int(round(out_w / GAME_ASPECT / 2) * 2)
    if game_h > out_h - top_h:                                   # squarer outputs: the game gets what is left
        game_h = out_h - top_h
        gh = float(rh)
        gw = min(float(rw), gh * out_w / game_h)
        gh = gw * game_h / out_w
    game = (int(rx + (rw - gw) / 2), int(ry + (rh - gh) / 2), int(gw), int(gh))
    for run in runs:
        if run.get("box"):
            run["patch"] = _patch(run, game)
    places = len({r["box"] for r in runs if r.get("box")})
    return {"layout": "cutout", "runs": runs, "top_h": top_h, "game": game, "game_h": game_h,
            "background": BACKGROUNDS.get(background, BACKGROUNDS["white"]),
            "why": f"streamer cut out over the game ({places} position{'s' if places != 1 else ''}, found in {present:.0%} of the clip)"}


# --------------------------------------------------------------------------------------------- #
# One frame
# --------------------------------------------------------------------------------------------- #
def _run_for(plan_: Dict, frame_no: int) -> Dict:
    for run in plan_["runs"]:
        if run["start"] <= frame_no < run["end"]:
            return run
    return plan_["runs"][-1]


def _patch(run: Dict, game: Tuple[int, int, int, int]) -> Optional[Dict]:
    """Everything about erasing one position that does not change from frame to frame (coords inside the game crop)."""
    x0, y0, x1, y1 = run["box"]
    gx, gy, gw, gh = game
    mask = np.zeros((gh, gw), dtype=np.float32)
    left, top = max(0, x0 - gx), max(0, y0 - gy)
    right, bottom = min(gw, x1 - gx), min(gh, y1 - gy)
    if right <= left or bottom <= top:
        return None
    mask[top:bottom, left:right] = run["erase"][top + gy - y0:bottom + gy - y0, left + gx - x0:right + gx - x0]
    ys, xs = np.nonzero(mask > 0.01)
    if not len(xs):
        return None
    pad = int((x1 - x0) * 0.3)
    a, b = max(0, int(ys.min()) - pad), min(gh, int(ys.max()) + pad)
    c, d = max(0, int(xs.min()) - pad), min(gw, int(xs.max()) + pad)
    soft = mask[a:b, c:d]
    small = (max(4, (d - c) // 12), max(4, (b - a) // 12))
    keep = (cv2.resize(soft, small, interpolation=cv2.INTER_AREA) < 0.05).astype(np.float32)
    sigma = max(1.5, small[0] / 9.0)
    return {"slice": (a, b, c, d), "alpha": soft, "rest": 1.0 - soft, "small": small, "keep": keep[..., None], "sigma": sigma,
            "weight": cv2.GaussianBlur(keep, (0, 0), sigma)[..., None]}


def _erase(game: "np.ndarray", patch: Optional[Dict]) -> None:
    """Smooth the small copy of the streamer out of the game picture (in place): fill the spot from its surroundings."""
    if patch is None:
        return
    a, b, c, d = patch["slice"]
    part = game[a:b, c:d]
    tiny = cv2.resize(part, patch["small"], interpolation=cv2.INTER_AREA).astype(np.float32)
    total = cv2.GaussianBlur(tiny * patch["keep"], (0, 0), patch["sigma"])
    weight = patch["weight"]
    known = weight > 1e-3
    fill = np.where(known, total / np.maximum(weight, 1e-3), 0.0)
    if not known.all():                                          # the middle of a big spot: the average of what is around
        fill = np.where(known, fill, total.reshape(-1, 3).sum(axis=0) / max(float(weight.sum()), 1e-3))
    fill = cv2.resize(fill.astype(np.uint8), (d - c, b - a), interpolation=cv2.INTER_LINEAR)
    game[a:b, c:d] = cv2.blendLinear(part, fill, patch["rest"], patch["alpha"])


def _one_person(sure: "np.ndarray") -> "np.ndarray":
    """Only the largest connected body: bits of the chair or the game that look a little like a person are dropped."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((sure > 0.65).astype(np.uint8))
    if count <= 1:
        return sure
    body = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return sure * cv2.dilate(body, np.ones((5, 5), np.uint8))


def compose(frame: "np.ndarray", plan_: Dict, frame_no: int, roi, out_w: int, out_h: int, zoom: float = 1.0) -> "np.ndarray":
    run = _run_for(plan_, frame_no)
    if not run.get("box"):                                       # the streamer is not on screen: just show the picture
        return framing.compose_block(frame, framing.zoom_rect(*roi, 1.0 + (zoom - 1.0) * framing.GROUP_ZOOM), roi, out_w, out_h, top_frac=0.5)
    top_h, game_h = plan_["top_h"], plan_["game_h"]
    if top_h + game_h < out_h:
        canvas = framing.blurred_background(frame, roi, out_w, out_h, dim=0.45)
    else:
        canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)

    gx, gy, gw, gh = plan_["game"]
    game = frame[gy:gy + gh, gx:gx + gw].copy()
    _erase(game, run["patch"])
    canvas[top_h:top_h + game_h] = framing.warp(game, *framing.zoom_rect(0, 0, gw, gh, 1.0 + (zoom - 1.0) * 0.3), out_w, game_h)

    # the streamer: sitting on the dividing line, centred
    canvas[:top_h] = plan_["background"]
    x0, y0, x1, y1 = run["box"]
    crop = frame[y0:y1, x0:x1]
    scale = min(top_h * PERSON_SHARE / crop.shape[0], out_w / (crop.shape[1] * 0.72)) * min(zoom, 1.12)
    w, h = int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))
    big = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    sure = cv2.resize(_one_person(person_mask(crop, full=False)), (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip((sure - 0.5) * (1.0 / 0.35), 0.0, 1.0)          # only what the model is sure of: no haze, no half chair
    y, x = top_h - h, (out_w - w) // 2
    sy, sx = max(0, -y), max(0, -x)
    dy, dx = max(0, y), max(0, x)
    hh, ww = min(h - sy, top_h - dy), min(w - sx, out_w - dx)
    a = alpha[sy:sy + hh, sx:sx + ww]
    area = canvas[dy:dy + hh, dx:dx + ww]
    canvas[dy:dy + hh, dx:dx + ww] = cv2.blendLinear(area, big[sy:sy + hh, sx:sx + ww], 1.0 - a, a)
    canvas[top_h - 4:top_h + 4] = (30, 190, 255)                # a thin amber divider
    return canvas
