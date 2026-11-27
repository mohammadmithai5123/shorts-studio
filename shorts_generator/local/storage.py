"""How much disk the tool uses, and what can safely be cleaned.

Streaming means a normal run leaves only the rendered clips and a few MB of cache. Older versions
downloaded the whole video (``output/source_<id>.mp4``, often 1-3 GB); those files are not needed any
more and can be removed here.
"""
import shutil
from pathlib import Path
from typing import Dict, List

from ..config import LOCAL_OUTPUT_DIR


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def _legacy_files(base: Path) -> List[Path]:
    """Whole-video downloads and Whisper caches left behind by the download-everything versions."""
    return [p for p in base.glob("source_*") if p.is_file()]


def usage() -> Dict[str, int]:
    base = Path(LOCAL_OUTPUT_DIR)
    clips = sum(_size(p) for p in base.glob("clips_*"))
    legacy = sum(_size(p) for p in _legacy_files(base)) if base.exists() else 0
    cache = _size(base / "cache")
    llm = _size(base / ".llm_cache")
    return {"clips": clips, "legacy_downloads": legacy, "cache": cache, "llm_cache": llm,
            "total": clips + legacy + cache + llm}


def clean(what: List[str]) -> Dict[str, int]:
    """Delete the named groups ('legacy_downloads', 'cache', 'llm_cache'); returns bytes freed per group.

    Rendered clips are never touched here: delete a project to remove those.
    """
    base = Path(LOCAL_OUTPUT_DIR)
    freed: Dict[str, int] = {}
    if "legacy_downloads" in what and base.exists():
        n = 0
        for p in _legacy_files(base):
            n += _size(p)
            p.unlink(missing_ok=True)
        freed["legacy_downloads"] = n
    for name, folder in (("cache", base / "cache"), ("llm_cache", base / ".llm_cache")):
        if name in what and folder.exists():
            freed[name] = _size(folder)
            shutil.rmtree(folder, ignore_errors=True)
    return freed
