"""Keeping the computer light, by itself. Runs when the app starts, after every job and every few hours.

What goes:
  * a rendered clip once it is on YouTube (its history stays, so it is never made or uploaded twice);
  * saved captions / loudness / AI answers not used for a few days (all can be fetched or asked again);
  * leftovers of interrupted work (half downloads, temporary render folders).
What stays: clips that are not posted yet, project lists, upload history, sign-ins, settings, your memes.
"""
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from .. import config
from ..logsetup import get as _get_logger
from . import project as P

log = _get_logger("housekeeping")

HOUR, DAY = 3600.0, 86400.0
TEMP_PREFIXES = ("shorts_clip_", "shorts_caps_")


def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _newest(path: Path) -> float:
    try:
        return max([path.stat().st_mtime] + [f.stat().st_mtime for f in path.rglob("*")])
    except OSError:
        return time.time()


def _remove(path: Path) -> int:
    """Delete a file or folder; returns the bytes freed (0 if Windows would not let go of it: next time)."""
    size = _size(path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        return 0
    return size


def _uploaded_at(record: Dict) -> Optional[float]:
    try:
        return time.mktime(time.strptime(str(record.get("uploaded")), "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError, OverflowError):
        return None


def remove_posted_clips(base: Path) -> int:
    """Delete clip files that are on YouTube. A file rendered again after its upload is a new clip and stays."""
    freed = 0
    for queue in base.glob("clips_*/uploads.json"):
        try:
            records = json.loads(queue.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        gone: Dict[str, Dict] = {}
        for name, record in records.items() if isinstance(records, dict) else []:
            path = queue.parent / name
            when = _uploaded_at(record)
            if record.get("state") != "uploaded" or when is None or Path(name).name != name or not path.is_file():
                continue
            if path.stat().st_mtime > when + 2:
                continue
            size = _remove(path)
            if size:
                freed += size
                gone[name] = record
        if gone:
            _note_posted(queue.parent.name[len("clips_"):], gone)
    return freed


def _note_posted(project_id: str, gone: Dict[str, Dict]) -> None:
    def mark(fresh: Dict) -> None:
        for clip in fresh.get("clips", []):
            record = gone.get(clip.get("file") or "")
            if record:
                clip["removed_after_upload"] = True
                clip["youtube_url"] = record.get("url")

    try:
        P.update_project(P.load_project(project_id), mark)
    except (KeyError, ValueError, OSError):
        log.warning("could not note the posted clips of %s", project_id)


def clear_stale(base: Path, days: float, now: float) -> int:
    freed = 0
    cache = base / "cache"
    if cache.is_dir():
        for folder in cache.iterdir():
            if not folder.is_dir():
                continue
            for part in (folder / "sections").glob("*"):                 # clips' source pieces: deleted after use anyway
                if now - _newest(part) > 6 * HOUR:
                    freed += _remove(part)
            if now - _newest(folder) > days * DAY:
                freed += _remove(folder)
    for answer in (base / ".llm_cache").glob("*.json"):
        if now - _newest(answer) > days * DAY:
            freed += _remove(answer)
    for pattern in ("clips_*/*.cut.mp4", "clips_*/*.part", "clips_*/*.tmp"):
        for leftover in base.glob(pattern):
            if now - _newest(leftover) > 6 * HOUR:
                freed += _remove(leftover)
    temp = Path(tempfile.gettempdir())
    for prefix in TEMP_PREFIXES:
        for folder in temp.glob(prefix + "*"):
            if folder.is_dir() and now - _newest(folder) > DAY:
                freed += _remove(folder)
    return freed


def sweep(now: Optional[float] = None) -> Dict[str, int]:
    """One round of cleaning; returns bytes freed. Never raises: tidying up must not break anything."""
    base = Path(config.LOCAL_OUTPUT_DIR)
    freed = {"posted_clips": 0, "old_data": 0}
    if not base.is_dir():
        return freed
    try:
        if config.delete_posted_clips():
            freed["posted_clips"] = remove_posted_clips(base)
        freed["old_data"] = clear_stale(base, config.cache_days(), now if now is not None else time.time())
    except Exception:  # noqa: BLE001
        log.exception("clean-up failed")
    if freed["posted_clips"] or freed["old_data"]:
        log.info("cleaned up %.1f MB (posted clips %.1f MB, old data %.1f MB)", sum(freed.values()) / 1e6,
                 freed["posted_clips"] / 1e6, freed["old_data"] / 1e6)
    return freed
