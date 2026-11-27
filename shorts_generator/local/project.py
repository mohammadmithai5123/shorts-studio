"""A *project* is one analysed video: its ranked candidate clips and what has been rendered so far.

Layout on disk (under LOCAL_OUTPUT_DIR, default ``output/``):

    output/clips_<id>/project.json      the project (small)
    output/clips_<id>/short_NN_<title>.mp4   rendered clips
    output/cache/<id>/transcript.json   the transcript (re-used when the same video is analysed again)
    output/cache/<id>/energy.json       per-second voice loudness
    output/.llm_cache/                  cached LLM answers

project.json (schema 2):
{
  "schema": 2, "id": "169ZzvPxvvM", "title": "...", "created": "...", "updated": "...", "duration": 7439.0,
  "source": {kind, id, url, title, duration, language, embeddable, heatmap, chapters, local_path},
  "options": {focus, min_seconds, max_seconds, candidates, language, use_audio},
  "transcript_info": {"source": "youtube-captions:auto:hi-orig", "language": "hi", "segments": 2485},
  "candidates": [{"id": 0, "start_time", "end_time", "score", "title", "hook_sentence", "virality_reason",
                  "llm_score"?, "signals"?: {"energy": 93, "replay": 80}, "kit"?: {title, description, hashtags}}],
  "clips": [{"candidate": 0, "file": "short_01_x.mp4", "layout", "captions", "caption_style", "aspect_ratio",
             "start", "end", "quality", "error", "created"}]
}
Schema-1 projects (transcript embedded, ``source_path``/``source_url``) are upgraded on load.
"""
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List

from ..config import LOCAL_OUTPUT_DIR

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
SCHEMA = 2


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def valid_id(project_id: str) -> bool:
    return bool(_ID_RE.match(project_id or ""))


def project_dir(project_id: str) -> Path:
    return Path(LOCAL_OUTPUT_DIR) / f"clips_{project_id}"


def cache_dir(video_id: str) -> Path:
    return Path(LOCAL_OUTPUT_DIR) / "cache" / video_id


def new_project(source_meta: Dict, transcript: Dict, candidates: List[Dict], options: Dict) -> Dict:
    ranked = [{"id": i, **{k: v for k, v in h.items() if k != "id"}} for i, h in enumerate(candidates)]
    return {
        "schema": SCHEMA,
        "id": source_meta["id"],
        "title": source_meta.get("title") or source_meta["id"],
        "created": _now(),
        "updated": _now(),
        "duration": float(source_meta.get("duration") or transcript.get("duration") or 0.0),
        "source": source_meta,
        "options": options,
        "transcript_info": {
            "source": transcript.get("source", ""), "language": transcript.get("language", ""),
            "segments": len(transcript.get("segments", [])),
        },
        "candidates": ranked,
        "clips": [],
    }


def _upgrade(p: Dict) -> Dict:
    """Schema 1 -> 2 (in memory; written back on the next save)."""
    if p.get("schema") == SCHEMA and "source" in p:
        return p
    url = p.get("source_url") or ""
    path = p.get("source_path") or ""
    local = path if path and os.path.isfile(path) else None
    kind = "youtube" if re.search(r"youtu\.?be", url) else ("web" if url.startswith("http") else "file")
    p["source"] = {
        "kind": kind, "id": p["id"], "url": url or path, "title": p.get("title"), "duration": p.get("duration"),
        "language": None, "embeddable": kind == "youtube", "heatmap": None, "chapters": None, "local_path": local,
    }
    p.setdefault("options", {})
    t = p.get("transcript") or {}
    p.setdefault("transcript_info", {"source": t.get("source") or "whisper", "language": t.get("language", ""),
                                      "segments": len(t.get("segments", []))})
    p["schema"] = SCHEMA
    return p


_LOCKS: Dict[str, "threading.RLock"] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(project_id: str) -> "threading.RLock":
    """One lock per project: the job worker and the web requests both write project.json."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(project_id, threading.RLock())


def write_json_atomic(path: Path, data) -> None:
    """Temp file + replace (a reader never sees half a file); a replace Windows refuses for a moment is retried."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def _read_text(path: Path) -> str:
    """Read a file another thread may be replacing right now: Windows refuses that for a moment."""
    for attempt in range(5):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    return path.read_text(encoding="utf-8")


def save_project(project: Dict) -> Path:
    """Atomic write (temp file + replace), so a UI poll never reads a half-written file.

    Writers are serialised per project, and a replace that Windows refuses for a moment (a reader, a virus scanner)
    is retried. To change a project that may have been edited meanwhile, use :func:`update_project` instead.
    """
    folder = project_dir(project["id"])
    with _lock_for(project["id"]):
        folder.mkdir(parents=True, exist_ok=True)
        project["updated"] = _now()
        path = folder / "project.json"
        tmp = folder / "project.json.tmp"
        tmp.write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")
        if path.exists():
            shutil.copy2(path, folder / "project.json.bak")    # one step of undo for a bad write or a bad edit
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
    return path


def update_project(project: Dict, mutate: Callable[[Dict], None]) -> Dict:
    """Read-modify-write under the project lock.

    A render job holds its copy of the project for minutes; meanwhile the person may add their own clip or delete a
    rendered one. Saving the old copy would silently undo that. Instead the saved file is re-read, ``mutate(fresh)``
    changes only what this caller owns, and the result is saved. ``project`` (the caller's dict) is refreshed in place.
    """
    with _lock_for(project["id"]):
        try:
            fresh = load_project(project["id"])
        except (KeyError, ValueError):
            fresh = project                                    # deleted or unreadable: fall back to what we hold
        mutate(fresh)
        save_project(fresh)
        if fresh is not project:
            project.clear()
            project.update(fresh)
    return project


def load_project(project_id: str) -> Dict:
    if not valid_id(project_id):
        raise KeyError(project_id)
    path = project_dir(project_id) / "project.json"
    if not path.exists():
        raise KeyError(project_id)
    try:
        return _upgrade(json.loads(_read_text(path)))
    except ValueError:                                         # half-written / damaged file: fall back to the last good copy
        bak = path.with_name("project.json.bak")
        if bak.exists():
            return _upgrade(json.loads(bak.read_text(encoding="utf-8")))
        raise


def load_transcript(project: Dict) -> Dict:
    """The project's transcript: from the cache (schema 2) or embedded (schema 1)."""
    if project.get("transcript", {}).get("segments"):
        return project["transcript"]
    path = cache_dir(project["id"]) / "transcript.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise KeyError("transcript")


def list_projects() -> List[Dict]:
    """Small summaries (no transcript), newest first."""
    out: List[Dict] = []
    base = Path(LOCAL_OUTPUT_DIR)
    if not base.exists():
        return out
    for p in base.glob("clips_*/project.json"):
        try:
            d = _upgrade(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            continue
        src = d.get("source", {})
        out.append({
            "id": d.get("id"), "title": d.get("title"), "kind": src.get("kind"), "source_url": src.get("url"),
            "created": d.get("created"), "updated": d.get("updated"), "duration": d.get("duration"),
            "n_candidates": len(d.get("candidates", [])),
            "n_clips": len([c for c in d.get("clips", []) if c.get("file")]),
            "thumbnail": f"https://i.ytimg.com/vi/{src.get('id')}/mqdefault.jpg" if src.get("kind") == "youtube" else None,
        })
    out.sort(key=lambda d: d.get("updated") or "", reverse=True)
    return out


def add_manual_candidate(project: Dict, start: float, end: float, title: str = "") -> Dict:
    """A clip the person chose themselves (appended, so existing candidate ids never change)."""
    cand = {
        "id": len(project["candidates"]), "start_time": float(start), "end_time": float(end),
        "score": None, "title": title.strip() or "My clip", "hook_sentence": "", "virality_reason": "Added by you",
        "manual": True,
    }
    project["candidates"].append(cand)
    return cand


def delete_project(project_id: str, delete_cache: bool = False) -> int:
    """Remove the project folder (and optionally its cache). Returns bytes freed."""
    if not valid_id(project_id):
        raise KeyError(project_id)
    freed = 0
    for folder in [project_dir(project_id)] + ([cache_dir(project_id)] if delete_cache else []):
        if folder.exists():
            freed += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            shutil.rmtree(folder, ignore_errors=True)
    return freed
