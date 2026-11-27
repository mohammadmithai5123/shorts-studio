"""The upload queue: which rendered clips go to YouTube, and what happened to each one.

``<project folder>/uploads.json`` maps a clip file name to a record::

    {"state": "queued" | "uploading" | "uploaded" | "failed", "privacy", "publish_at" (UTC text or None), "category",
     "queued" (epoch), "attempts", "error", "retry_after" (epoch), "title",
     "video_id", "url", "studio_url", "actual_privacy", "notice", "uploaded" (local time text)}

It is a file of its own (not part of project.json) so that re-rendering a clip or searching the video again can never
forget that a clip is already on the channel, and so that nothing is ever uploaded twice by accident.

Two ways in: the person ticks clips and presses Upload (:func:`enqueue`, then a job calls :func:`drain`), or autopilot does
the same after rendering. A queued clip that has to wait (daily cap, YouTube's quota, no internet) stays queued; the app's
watcher starts another drain when it is worth trying again.
"""
import calendar
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .. import config
from ..logsetup import get as _get_logger
from . import control, romanize, youtube
from . import project as P

log = _get_logger("uploads")
PRIVACIES = ("private", "unlisted", "public")
MIN_LEAD_SECONDS = 600                     # a scheduled video must go live at least this far ahead (YouTube wants "in the future")


def _path(project_id: str) -> Path:
    return P.project_dir(project_id) / "uploads.json"


def load(project_id: str) -> Dict[str, Dict]:
    import json

    try:
        data = json.loads(_path(project_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _update(project_id: str, mutate: Callable[[Dict[str, Dict]], None]) -> Dict[str, Dict]:
    with P._lock_for(project_id):                      # the job worker and the web requests both write this file
        data = load(project_id)
        mutate(data)
        P.write_json_atomic(_path(project_id), data)
        return data


def short_warnings(clip: Dict) -> List[str]:
    """YouTube files a video under Shorts when it is square or vertical and at most 3 minutes. Say so when a clip is not."""
    out = []
    w, _, h = str(clip.get("aspect_ratio") or "9:16").partition(":")
    try:
        if float(w) > float(h):
            out.append("This clip is wide (" + str(clip.get("aspect_ratio")) + "): YouTube will publish it as a normal video, not a Short.")
    except ValueError:
        pass
    length = float(clip.get("end") or 0) - float(clip.get("start") or 0)
    if length > 180:
        out.append(f"This clip is {length:.0f} s long: YouTube Shorts can be at most 3 minutes.")
    return out


def _parse_utc(text: str) -> float:
    return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))


def clean_override(raw: object) -> Dict:
    """The person's own title / description / hashtags for one clip, trimmed to what YouTube accepts (empty fields are dropped)."""
    if not isinstance(raw, dict):
        return {}
    out: Dict = {}
    title = " ".join(str(raw.get("title") or "").split())[:100]
    if title:
        out["title"] = title
    if "description" in raw and raw["description"] is not None:
        out["description"] = str(raw["description"]).strip()[:4000]
    tags = raw.get("hashtags")
    if isinstance(tags, str):
        tags = tags.replace(",", " ").split()
    if isinstance(tags, list):
        out["hashtags"] = [("#" + str(t).strip().lstrip("#"))[:40] for t in tags if str(t).strip().lstrip("#")][:15]
    return out


def enqueue(project: Dict, files: List[str], *, privacy: Optional[str] = None, schedule_start: Optional[float] = None,
            every_hours: float = 24.0, category: Optional[str] = None, force: bool = False,
            overrides: Optional[Dict[str, Dict]] = None) -> Dict:
    """Put rendered clips in the queue. Returns {"queued": [...], "skipped": [{"file", "reason"}], "warnings": {file: [...]}}.

    ``schedule_start`` (epoch seconds) makes the first clip go live then and each next one ``every_hours`` later; scheduled
    videos are uploaded private and YouTube publishes them by itself. Clips already on YouTube are skipped unless ``force``.
    """
    privacy = privacy or config.youtube_privacy()
    if privacy not in PRIVACIES:
        raise ValueError("Visibility must be private, unlisted or public.")
    now = time.time()
    if schedule_start is not None:
        if schedule_start < now + MIN_LEAD_SECONDS:
            raise ValueError("Pick a publishing time at least 10 minutes from now.")
        if not (0.25 <= float(every_hours) <= 24 * 30):
            raise ValueError("The gap between clips must be between 15 minutes and 30 days.")
    pid = project["id"]
    folder = P.project_dir(pid)
    by_file = {c["file"]: c for c in project.get("clips", []) if c.get("file")}
    queued: List[str] = []
    skipped: List[Dict] = []
    warnings: Dict[str, List[str]] = {}

    def put(data: Dict[str, Dict]) -> None:
        for file in dict.fromkeys(files):                       # in the order given, once each
            clip = by_file.get(file)
            if clip is None or not (folder / file).is_file():
                skipped.append({"file": file, "reason": "That clip is not rendered (any more)."})
                continue
            old = data.get(file) or {}
            if old.get("state") in ("queued", "uploading"):
                skipped.append({"file": file, "reason": "Already waiting to be uploaded."})
                continue
            if old.get("state") == "uploaded" and not force:
                skipped.append({"file": file, "reason": "Already on YouTube."})
                continue
            n = len(queued)
            publish_at = youtube.iso_utc(schedule_start + n * float(every_hours) * 3600) if schedule_start is not None else None
            rec = {"state": "queued", "privacy": privacy, "publish_at": publish_at, "category": category or config.youtube_category(),
                   "queued": now + n * 0.001, "attempts": 0, "error": None, "retry_after": 0, "title": clip.get("title")}
            override = clean_override((overrides or {}).get(file))
            if override:
                rec["override"] = override
            if old.get("video_id"):                               # a re-upload: remember the first one, so cancelling restores it
                rec["previous"] = {k: v for k, v in old.items() if k != "previous"}
            data[file] = rec
            queued.append(file)
            if short_warnings(clip):
                warnings[file] = short_warnings(clip)

    _update(pid, put)
    return {"queued": queued, "skipped": skipped, "warnings": warnings}


def dequeue(project_id: str, files: Optional[List[str]] = None) -> int:
    """Take clips out of the queue (queued or failed ones; never one that is uploading or already on YouTube)."""
    removed: List[str] = []

    def drop(data: Dict[str, Dict]) -> None:
        for file in list(data):
            if (files is None or file in files) and data[file].get("state") in ("queued", "failed"):
                if data[file].get("previous"):                  # a re-upload that was cancelled: back to "uploaded"
                    data[file] = data[file]["previous"]
                else:
                    del data[file]
                removed.append(file)

    _update(project_id, drop)
    return len(removed)


def pending_projects(now: Optional[float] = None) -> List[str]:
    """Projects with clips that are queued and due (not waiting for a retry time)."""
    import json

    now = time.time() if now is None else now
    out: List[str] = []
    for f in sorted(Path(config.LOCAL_OUTPUT_DIR).glob("clips_*/uploads.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if any(isinstance(r, dict) and r.get("state") == "queued" and float(r.get("retry_after") or 0) <= now for r in data.values()):
            out.append(f.parent.name[len("clips_"):])
    return out


def all_records() -> List[Dict]:
    """Every upload record of every project (for the YouTube page): newest first."""
    import json

    titles = {p["id"]: p.get("title") or p["id"] for p in P.list_projects()}
    out: List[Dict] = []
    for f in Path(config.LOCAL_OUTPUT_DIR).glob("clips_*/uploads.json"):
        pid = f.parent.name[len("clips_"):]
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for file, rec in (data if isinstance(data, dict) else {}).items():
            if isinstance(rec, dict):
                out.append({"project_id": pid, "project_title": titles.get(pid, pid), "file": file,
                            **{k: rec.get(k) for k in ("state", "title", "privacy", "publish_at", "queued", "uploaded", "error", "retry_after",
                                                      "url", "studio_url", "actual_privacy", "notice", "video_id")}})
    out.sort(key=lambda r: (r.get("uploaded") or "", r.get("queued") or 0), reverse=True)
    return out


def dequeue_everywhere() -> int:
    """Take every waiting or failed clip of every project out of the queue (the "stop everything" button)."""
    import json

    total = 0
    for f in sorted(Path(config.LOCAL_OUTPUT_DIR).glob("clips_*/uploads.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if any(isinstance(r, dict) and r.get("state") in ("queued", "failed") for r in data.values()):
            total += dequeue(f.parent.name[len("clips_"):])
    return total


def claimed_ranges(project: Dict) -> List[tuple]:
    """(start, end) of the moments of this video that are already on the channel or waiting to go: Autopilot must not repeat them."""
    recs = load(project["id"])
    out = []
    for c in project.get("clips", []):
        if c.get("file") and (recs.get(c["file"]) or {}).get("state") in ("uploaded", "queued", "uploading"):
            out.append((float(c.get("start") or 0), float(c.get("end") or 0)))
    return out


def overlaps(start: float, end: float, ranges: List[tuple], fraction: float = 0.5) -> bool:
    length = max(end - start, 0.001)
    return any(min(end, b) - max(start, a) > fraction * min(length, max(b - a, 0.001)) for a, b in ranges)


def queue_size() -> int:
    """How many clips are waiting (or uploading) right now, across all videos (the badge on the YouTube tab)."""
    import json

    n = 0
    for f in Path(config.LOCAL_OUTPUT_DIR).glob("clips_*/uploads.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        n += sum(1 for r in data.values() if isinstance(r, dict) and r.get("state") in ("queued", "uploading"))
    return n


def counts(project_id: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in load(project_id).values():
        out[rec.get("state", "?")] = out.get(rec.get("state", "?"), 0) + 1
    return out


def _metadata_for(project: Dict, clip: Dict, rec: Dict) -> Dict:
    cand = None
    cid = clip.get("candidate")
    if isinstance(cid, int) and 0 <= cid < len(project.get("candidates", [])):
        cand = project["candidates"][cid]
    kit = dict((cand or {}).get("kit") or {})
    title = kit.get("title") or clip.get("title") or (cand or {}).get("title") or "Short"
    if config.caption_script() == "roman":                     # the AI's text in the same letters as the subtitles
        title = romanize.text(title)
        kit["description"] = romanize.text(kit.get("description", ""))
    override = rec.get("override") or {}
    kit.update(override)                                       # what the person typed in the review step wins over the AI's text
    title = override.get("title") or title
    publish_at = rec.get("publish_at")
    if publish_at and _parse_utc(publish_at) < time.time() + MIN_LEAD_SECONDS / 2:      # the queue waited: never ask for a time in the past
        publish_at = youtube.iso_utc(time.time() + MIN_LEAD_SECONDS)
        print("[upload] the scheduled time had passed while the clip waited: it goes live in about 10 minutes instead", flush=True)
    return youtube.build_metadata(title, kit.get("description", ""), kit.get("hashtags") or [], privacy=rec.get("privacy") or "private",
                                  publish_at=publish_at, category=rec.get("category") or "22")


def drain(project_id: str, ensure_kit: Optional[Callable[[Dict, List[int]], object]] = None) -> Dict:
    """Upload this project's queued clips, one after the other, until the queue is empty or YouTube says "not now".

    Returns {"uploaded": n, "failed": n, "waiting": n, "until": epoch or None, "reason": text}. Raises NotConnected /
    ReconnectNeeded (nothing is lost: the clips stay queued) and Cancelled.
    """
    youtube.require_ready()
    result: Dict = {"uploaded": 0, "failed": 0, "waiting": 0, "until": None, "reason": ""}
    if youtube.is_paused():                                     # the big red switch: leave everything queued
        result["waiting"] = len([1 for r in load(project_id).values() if r.get("state") == "queued"])
        result["reason"] = "Uploads are paused (YouTube page: Resume)."
        return result

    def revive(data: Dict[str, Dict]) -> None:                  # "uploading" can only be left over from a crash: nothing else uploads
        for rec in data.values():
            if rec.get("state") == "uploading":
                rec["state"] = "queued"

    _update(project_id, revive)
    project = P.load_project(project_id)

    def due() -> List[str]:
        now = time.time()
        recs = load(project_id)
        return sorted((f for f, r in recs.items() if r.get("state") == "queued" and float(r.get("retry_after") or 0) <= now),
                      key=lambda f: (recs[f].get("queued") or 0, f))

    if ensure_kit is not None:                                   # one AI call for all clips that have no title / hashtags yet
        by_file = {c["file"]: c for c in project.get("clips", []) if c.get("file")}
        need = sorted({by_file[f]["candidate"] for f in due() if f in by_file and isinstance(by_file[f].get("candidate"), int)
                       and not (project["candidates"][by_file[f]["candidate"]].get("kit"))})
        if need:
            try:
                ensure_kit(project, need)
                project = P.load_project(project_id)
            except Exception as e:  # noqa: BLE001 - upload with the working titles rather than not at all
                print(f"[upload] could not write titles and hashtags ({e}); using the clips' working titles", flush=True)

    total = len(due())
    done_count = 0
    while True:
        control.check()
        queue = due()
        if not queue:
            break
        pk = youtube.parked()
        if pk or youtube.uploads_left() <= 0:
            result["until"] = pk["until"] if pk else youtube.next_pacific_midnight(time.time())
            result["reason"] = pk["reason"] if pk else f"Your limit of {config.youtube_daily_uploads()} uploads per day is reached."
            break
        file = queue[0]
        project = P.load_project(project_id)
        clip = next((c for c in project.get("clips", []) if c.get("file") == file), None)
        path = P.project_dir(project_id) / file
        if clip is None or not path.is_file():
            _update(project_id, lambda d, f=file: d.__setitem__(f, {**d[f], "state": "failed", "error": "The clip file is gone."}))
            result["failed"] += 1
            continue
        rec = load(project_id)[file]
        meta = _metadata_for(project, clip, rec)
        _update(project_id, lambda d, f=file: d[f].update(state="uploading", attempts=int(d[f].get("attempts") or 0) + 1, error=None))
        label = meta["snippet"]["title"]
        print(f"[upload] {done_count + 1}/{max(total, done_count + 1)}: {label}", flush=True)

        def progress(sent: int, size: int, n: int = done_count) -> None:
            control.report("upload", sent / size if size else None, f"{n + 1}/{max(total, n + 1)}: {label[:50]} ({sent * 100 // max(size, 1)}%)")

        try:
            with control.scale(done_count / max(total, 1), 1 / max(total, 1)):
                resource = youtube.upload_video(str(path), meta, progress)
        except control.Cancelled:
            _update(project_id, lambda d, f=file: d[f].update(state="queued"))
            raise
        except youtube.QuotaExceeded as e:
            _update(project_id, lambda d, f=file: d[f].update(state="queued", error=None))
            youtube.park(e.until, str(e))
            result.update(until=e.until, reason=str(e))
            break
        except (youtube.ReconnectNeeded, youtube.NotConnected):
            _update(project_id, lambda d, f=file: d[f].update(state="queued"))
            raise
        except youtube.NetworkError as e:
            _update(project_id, lambda d, f=file, msg=str(e): d[f].update(state="queued", error=msg, retry_after=time.time() + 300))
            result.update(until=time.time() + 300, reason=str(e))
            break
        except youtube.UploadRejected as e:
            _update(project_id, lambda d, f=file, msg=str(e): d[f].update(state="failed", error=msg))
            print(f"[upload] refused: {e}", flush=True)
            result["failed"] += 1
            done_count += 1
            continue
        except Exception as e:  # noqa: BLE001 - a bug or an odd answer must not stop the other clips
            log.exception("upload of %s failed", file)
            _update(project_id, lambda d, f=file, msg=(str(e) or e.__class__.__name__): d[f].update(state="failed", error=msg))
            result["failed"] += 1
            done_count += 1
            continue

        vid = str(resource.get("id") or "")
        actual = ((resource.get("status") or {}).get("privacyStatus")) or ""
        notice = ""
        if not rec.get("publish_at") and actual and actual != rec.get("privacy"):
            notice = (f"YouTube made this video {actual} although {rec.get('privacy')} was requested. YouTube does that for Google "
                      "projects that have not passed its API audit. Open YouTube Studio to change the visibility, or request the audit.")
        _update(project_id, lambda d, f=file, v=vid, a=actual, nt=notice: d[f].update(
            state="uploaded", video_id=v, url=f"https://www.youtube.com/shorts/{v}", studio_url=f"https://studio.youtube.com/video/{v}/edit",
            actual_privacy=a, notice=nt, error=None, uploaded=time.strftime("%Y-%m-%dT%H:%M:%S"), title=label))
        youtube.count_upload()
        result["uploaded"] += 1
        done_count += 1
        print(f"[upload] done: https://www.youtube.com/shorts/{vid}" + (f"  (!) {notice}" if notice else ""), flush=True)

    result["waiting"] = len([1 for r in load(project_id).values() if r.get("state") == "queued"])
    return result
