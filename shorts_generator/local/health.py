"""Is this machine ready? One place that knows how to check, shared by the CLI (``--doctor``), the web UI
(Settings -> Health) and the diagnostics bundle you attach to a bug report.

Every check returns ``{id, label, status: ok|warn|fail, detail, fix}``. Only ``fail`` means "cannot work";
``warn`` means "works, but you should know". Nothing here changes anything, except ``update_ytdlp``.
"""
import importlib.util
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .. import config
from ..logsetup import redact

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_FREE_MB = 400                  # below this we refuse to start work that writes video
_latest_cache: Dict = {"at": 0.0, "version": None}


class LowDiskSpace(RuntimeError):
    """Not enough free disk to do the job. The text is user-facing."""


def _check(cid: str, label: str, status: str, detail: str = "", fix: str = "") -> Dict:
    return {"id": cid, "label": label, "status": status, "detail": detail, "fix": fix}


# --------------------------------------------------------------------------------------------- #
# Small probes
# --------------------------------------------------------------------------------------------- #
def ffmpeg_version() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        first = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=15).stdout.splitlines()[0]
        m = re.search(r"version\s+(\S+)", first)
        return m.group(1) if m else first[:60]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def ytdlp_version() -> Optional[str]:
    try:
        from yt_dlp.version import __version__

        return __version__
    except Exception:
        return None


def _version_tuple(v: str) -> Tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:4])


def latest_ytdlp(timeout: float = 4.0) -> Optional[str]:
    """Newest yt-dlp on PyPI (cached for an hour); None when offline."""
    if _latest_cache["version"] and time.time() - _latest_cache["at"] < 3600:
        return _latest_cache["version"]
    try:
        r = requests.get("https://pypi.org/pypi/yt-dlp/json", timeout=timeout)
        r.raise_for_status()
        _latest_cache.update(at=time.time(), version=r.json()["info"]["version"])
        return _latest_cache["version"]
    except (requests.RequestException, ValueError, KeyError):
        return None


def free_mb(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(path).free // (1024 * 1024))


def require_free_space(path: Path, need_mb: int = MIN_FREE_MB) -> None:
    """Raise LowDiskSpace (with a message a person can act on) when the drive is nearly full."""
    free = free_mb(path)
    if free < need_mb:
        drive = path.resolve().drive or str(path.resolve())
        raise LowDiskSpace(
            f"Only {free} MB are free on {drive}; at least {need_mb} MB are needed to render clips. "
            "Free some space (Settings -> Disk used shows what the tool itself is using)."
        )


# --------------------------------------------------------------------------------------------- #
# The doctor
# --------------------------------------------------------------------------------------------- #
def youtube_check() -> Dict:
    """The "YouTube uploads" line of the health report. Deliberately without the channel name: diagnostics zips contain it."""
    from . import youtube

    yt = youtube.status()
    if not yt["configured"]:
        return _check("youtube_channel", "YouTube uploads (optional)", OK, "not set up", "Settings -> YouTube lets the app upload your shorts for you.")
    if yt["needs_reconnect"]:
        return _check("youtube_channel", "YouTube uploads", WARN, "the sign-in expired", "Settings -> YouTube -> Reconnect. Queued clips wait until then.")
    if yt["connected"]:
        detail = f"connected; {yt['uploads_today']} of {yt['daily_limit']} uploads used today"
        return _check("youtube_channel", "YouTube uploads", WARN if yt["parked_until"] else OK,
                      detail + ("; waiting: " + yt["parked_reason"] if yt["parked_until"] else ""),
                      "Uploads continue by themselves when YouTube allows it." if yt["parked_until"] else "")
    return _check("youtube_channel", "YouTube uploads", WARN, "client saved, channel not connected", "Settings -> YouTube -> Connect.")


def doctor(deep: bool = False) -> Dict:
    """Run every check. ``deep=True`` also talks to the network (PyPI, YouTube, the AI provider)."""
    checks: List[Dict] = []
    add = checks.append

    v = sys.version_info
    add(_check("python", "Python", OK if v >= (3, 10) else FAIL, f"{v.major}.{v.minor}.{v.micro}",
               "" if v >= (3, 10) else "Python 3.10 or newer is required."))

    ff = ffmpeg_version()
    add(_check("ffmpeg", "ffmpeg", OK if ff else FAIL, ff or "not found on PATH",
               "" if ff else "Install ffmpeg and add it to PATH (Windows: winget install Gyan.FFmpeg), then restart."))
    add(_check("ffprobe", "ffprobe", OK if shutil.which("ffprobe") else WARN, "found" if shutil.which("ffprobe") else "not found",
               "" if shutil.which("ffprobe") else "It ships with ffmpeg; without it local files cannot be measured."))
    if ff:
        try:
            from .clipper import _captions_supported

            ok = _captions_supported()
        except Exception:  # noqa: BLE001
            ok = False
        add(_check("libass", "Captions and title cards (libass)", OK if ok else WARN, "supported" if ok else "this ffmpeg cannot draw text",
                   "" if ok else "Install a 'full' ffmpeg build (it includes libass); clips still render, just without text."))

    try:
        import cv2

        haar = not cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").empty()
        add(_check("opencv", "OpenCV (face detection)", OK if haar else FAIL, f"{cv2.__version__}",
                   "" if haar else "pip install \"opencv-python-headless>=4.8,<5\""))
    except Exception as e:  # noqa: BLE001
        add(_check("opencv", "OpenCV (face detection)", FAIL, str(e)[:120], "pip install -r requirements-local.txt"))

    yv = ytdlp_version()
    if yv is None:
        add(_check("ytdlp", "yt-dlp", FAIL, "not installed", "pip install -r requirements-local.txt"))
    else:
        detail, status, fix = yv, OK, ""
        latest = latest_ytdlp() if deep else (_latest_cache["version"] if _latest_cache["version"] else None)
        try:
            age = datetime.now() - datetime.strptime(".".join(f"{int(p):d}" for p in yv.split(".")[:3]), "%Y.%m.%d")
        except ValueError:
            age = timedelta(0)
        if latest and _version_tuple(latest) > _version_tuple(yv):
            status, detail, fix = WARN, f"{yv} (newer: {latest})", "YouTube changes often; update yt-dlp (Settings -> Health -> Update yt-dlp)."
        elif not latest and age > timedelta(days=60):
            status, detail, fix = WARN, f"{yv} ({age.days} days old)", "Old versions stop working when YouTube changes. Update it (Settings -> Health)."
        add(_check("ytdlp", "yt-dlp (reads YouTube)", status, detail, fix))

    provider = config.llm_provider()
    key = config.gemini_key() if provider == "gemini" else config.openai_key()
    add(_check("llm_key", f"AI key ({provider})", OK if key else FAIL, "saved" if key else "missing",
               "" if key else "Add a key in Settings (a free Gemini key: https://aistudio.google.com/apikey)."))

    add(youtube_check())
    if ff:
        from . import encoder
        from .faces import MODEL

        hw = encoder.choose() != "cpu"
        add(_check("encoder", "Video encoder", OK, encoder.describe() + ("" if hw else " (no graphics-chip encoder found: renders are slower)")))
        add(_check("faces", "Face finding for framing", OK if MODEL.is_file() else WARN, "YuNet" if MODEL.is_file() else "basic fallback only",
                   "" if MODEL.is_file() else "assets/models/face_detection_yunet_2023mar.onnx is missing: framing of shows is much weaker."))
        from .cutout import available as cutout_available

        add(_check("cutout", "Green-screen streamer layout", OK if cutout_available() else WARN, "person model found" if cutout_available() else "off",
                   "" if cutout_available() else "assets/models/human_segmentation_pphumanseg_2023mar.onnx is missing."))

    out = Path(config.LOCAL_OUTPUT_DIR)
    try:
        mb = free_mb(out)
        status = OK if mb >= 2048 else WARN if mb >= MIN_FREE_MB else FAIL
        add(_check("disk", "Free disk space", status, f"{mb / 1024:.1f} GB free", "" if status == OK else "Free up some space before rendering."))
        probe = out / ".write_test"
        probe.write_text("x")
        probe.unlink()
        add(_check("writable", "Output folder", OK, str(out.resolve())))
    except OSError as e:
        add(_check("writable", "Output folder", FAIL, str(e)[:120], "Choose a writable LOCAL_OUTPUT_DIR."))

    whisper = importlib.util.find_spec("faster_whisper") is not None
    add(_check("whisper", "Speech recognition (optional)", OK if whisper else WARN,
               "installed" if whisper else "not installed (only needed for videos without captions)",
               "" if whisper else "pip install -r requirements-whisper.txt"))

    extras = []
    if config.env("YTDLP_COOKIES_FILE"):
        extras.append("cookies file")
    elif config.env("YTDLP_COOKIES_BROWSER"):
        extras.append("cookies from " + config.env("YTDLP_COOKIES_BROWSER"))
    if config.env("YTDLP_PROXY"):
        extras.append("proxy")
    add(_check("network_cfg", "Login / proxy", OK, ", ".join(extras) or "none configured"))

    if deep:
        try:
            r = requests.get("https://www.youtube.com/generate_204", timeout=6)
            add(_check("youtube", "Reach YouTube", OK if r.status_code in (200, 204) else WARN, f"HTTP {r.status_code}"))
        except requests.RequestException as e:
            add(_check("youtube", "Reach YouTube", FAIL, str(e)[:120], "Check the internet connection / proxy."))
        if key:
            from .llm import test_connection

            t = test_connection()
            add(_check("llm_live", "AI answers", OK if t.get("ok") else FAIL,
                       f"{t.get('provider')} answered in {t.get('seconds')}s" if t.get("ok") else t.get("error", "failed"),
                       "" if t.get("ok") else "Open Settings and check the key / model."))

    return {"ok": not any(c["status"] == FAIL for c in checks), "checks": checks,
            "version": config.version(), "python": platform.python_version(), "platform": platform.platform()}


def format_report(report: Dict) -> str:
    icon = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}
    lines = [f"Shorts Studio {report['version']}  |  Python {report['python']}  |  {report['platform']}", ""]
    for c in report["checks"]:
        lines.append(f"  [{icon[c['status']]}] {c['label']}: {c['detail']}")
        if c["fix"] and c["status"] != OK:
            lines.append(f"         -> {c['fix']}")
    lines += ["", "Ready." if report["ok"] else "Not ready: fix the FAIL lines above."]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- #
# yt-dlp update
# --------------------------------------------------------------------------------------------- #
def _unpatch_urllib3() -> None:
    """yt-dlp replaces urllib3's percent-encoding regex with a wrapper object. Importing yt-dlp a second time (after an
    update) would wrap the old wrapper, and every later ``requests`` call in the process fails with
    "'Urllib3PercentREOverride' object has no attribute 'sub'". Put the original back first.
    """
    try:
        import urllib3.util.url as u3

        original = getattr(getattr(u3, "_PERCENT_RE", None), "re", None)
        if original is not None and hasattr(original, "sub"):
            u3._PERCENT_RE = original
    except Exception:  # noqa: BLE001 — best effort
        pass


def update_ytdlp() -> Dict:
    """``pip install -U yt-dlp`` into this environment, then forget the old module so the next use loads the new one."""
    before = ytdlp_version()
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--disable-pip-version-check", "yt-dlp"],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError("pip could not update yt-dlp: " + (proc.stderr or proc.stdout).strip().splitlines()[-1][:300])
    _unpatch_urllib3()
    for name in [m for m in sys.modules if m == "yt_dlp" or m.startswith("yt_dlp.")]:
        del sys.modules[name]                       # pure Python: a fresh import picks up the new files
    after = ytdlp_version()
    _latest_cache.update(at=0.0, version=None)
    return {"before": before, "after": after, "changed": before != after}


# --------------------------------------------------------------------------------------------- #
# Diagnostics bundle
# --------------------------------------------------------------------------------------------- #
def diagnostics_zip(log_dir: Path, jobs: Optional[List[Dict]] = None) -> bytes:
    """A zip you can attach to a bug report: logs, health report, versions, recent jobs. Never contains secrets."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(log_dir.glob("studio.log*")) if log_dir.exists() else []:
            z.writestr(f"logs/{f.name}", redact(f.read_text(encoding="utf-8", errors="replace")))
        z.writestr("health.json", json.dumps(doctor(deep=False), indent=1))
        z.writestr("settings.json", json.dumps({
            "llm_provider": config.llm_provider(), "gemini_model": config.gemini_model(), "openai_model": config.openai_model(),
            "gemini_key_set": bool(config.gemini_key()), "openai_key_set": bool(config.openai_key()),
            "cookies_browser": config.env("YTDLP_COOKIES_BROWSER"), "cookies_file_set": bool(config.env("YTDLP_COOKIES_FILE")),
            "proxy_set": bool(config.env("YTDLP_PROXY")), "whisper_model": config.whisper_model(),
        }, indent=1))
        try:
            freeze = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"],
                                    capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            freeze = ""
        z.writestr("versions.txt", f"Shorts Studio {config.version()}\nPython {platform.python_version()}\n{platform.platform()}\n\n{freeze}")
        z.writestr("jobs.json", redact(json.dumps(jobs or [], indent=1, default=str)))
    return buf.getvalue()
