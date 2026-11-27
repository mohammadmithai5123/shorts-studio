"""Shorts Studio — local web UI.

    python app.py            # then open http://127.0.0.1:8765   (or double-click run_ui.bat)

Paste video links, let the tool rank the best moments of the WHOLE video (nothing is downloaded as a
whole), preview the candidates, trim them, tick the ones you like and render them. Everything runs on
this machine; the server only listens on 127.0.0.1.

This file is only the web layer: validation, a one-at-a-time job queue with live progress and cancel,
and JSON endpoints over shorts_generator.pipeline. The real work lives in shorts_generator/.
"""
import argparse
import contextlib
import csv
import html
import importlib.util
import io
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

from flask import Flask, Response, abort, jsonify, request, send_file

from shorts_generator import config, logsetup, pipeline
from shorts_generator.highlights import _excerpt
from shorts_generator.local import control, desktop, editing, encoder, health, housekeeping, llm, memes, storage, uploads, youtube
from shorts_generator.local import project as P
from shorts_generator.local.clipper import LAYOUTS, RenderOptions
from shorts_generator.local.captions import STYLES
from shorts_generator.local.signals import downsample, heatmap_series
from shorts_generator.local.source import SourceError

ROOT = Path(__file__).resolve().parent
WEBUI = ROOT / "webui"
SAFE_CLIP = re.compile(r"^[A-Za-z0-9._-]+\.mp4$")
STATIC_FILES = {"app.css": "text/css", "app.js": "text/javascript"}

app = Flask(__name__, static_folder=None)
app.json.ensure_ascii = False
log = logsetup.get("web")
MAX_WAITING_JOBS = 25
JOB_HISTORY = 40
# A busy or unreachable AI service is the most common reason a job fails (Gemini 503s, timeouts, a dropped connection). The
# work done so far is cached, so such a job puts itself back in the queue after a pause instead of making the person press
# "Try again" four times. Only while the work is still cheap to redo: after rendering began, a retry would re-render.
RETRY_DELAYS = (60, 180, 420, 900)                  # seconds before the 2nd, 3rd, 4th, 5th attempt
RETRY_STAGES = ("", "source", "transcript", "audio", "signals", "rank")

# What the page may load. 'unsafe-inline' for styles only: the page sets a few style attributes from script.
# The YouTube hosts are for the preview player; thumbnails come from ytimg.
CSP = ("default-src 'self'; script-src 'self' https://www.youtube.com https://s.ytimg.com; "
       "style-src 'self' 'unsafe-inline'; img-src 'self' data: https://*.ytimg.com https://*.ggpht.com https://*.googleusercontent.com; "
       "frame-src https://www.youtube.com https://www.youtube-nocookie.com; media-src 'self'; connect-src 'self'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


# --------------------------------------------------------------------------- #
# Guard rails: this is a local tool, but a web page open in another tab could still try to
# talk to it. Only answer our own host name (blocks DNS rebinding) and require a custom
# header on anything that changes state (a cross-site form or fetch cannot send one).
# --------------------------------------------------------------------------- #
@app.before_request
def _guard():
    host = (request.host or "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        abort(403)
    if request.method in ("POST", "PUT", "DELETE", "PATCH") and request.headers.get("X-Studio") != "1":
        abort(403)


@app.after_request
def _headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    if request.path == "/" and "Content-Security-Policy" not in resp.headers:
        resp.headers["Content-Security-Policy"] = CSP
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# --------------------------------------------------------------------------- #
# Background jobs: one worker, so two heavy jobs never fight for the CPU. While a job runs its
# print() output is captured and shown live in the page, and its progress is structured.
# The last JOB_HISTORY jobs are saved to <output>/state/jobs.json, so after a restart you still see what
# happened; a job that was running when the app closed shows up as "interrupted" with a Try again button
# (everything it had finished is cached, so a retry continues where it stopped).
# --------------------------------------------------------------------------- #
class Job:
    def __init__(self, kind: str, label: str, project_id: Optional[str] = None, request: Optional[Dict] = None):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.label, self.project_id = kind, label, project_id
        self.request = request                    # {"path": ..., "body": ...}: exactly what "Try again" re-sends
        self.status = "queued"                    # queued | running | done | error | cancelled | interrupted
        self.log: deque = deque(maxlen=600)
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
        self.stage = ""
        self.stages: List[str] = []               # stages seen, in order
        self.fraction: Optional[float] = None
        self.note = ""
        self.attempt = 0                          # automatic re-tries so far (see RETRY_DELAYS)
        self.retry_at: Optional[float] = None     # when the next automatic attempt starts (status stays "queued" until then)
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.cancel = threading.Event()

    def progress(self, stage: str, fraction: Optional[float], note: str) -> None:
        if stage != self.stage:
            self.stage = stage
            if stage not in self.stages:
                self.stages.append(stage)
        self.fraction = fraction
        self.note = note

    def to_dict(self, tail: int = 200) -> Dict:
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "project_id": self.project_id, "status": self.status,
            "request": self.request, "stage": self.stage, "stages": list(self.stages), "fraction": self.fraction,
            "note": self.note, "log": list(self.log)[-tail:] if tail else [], "result": self.result, "error": self.error,
            "created": self.created, "started": self.started, "finished": self.finished,
            "attempt": self.attempt, "max_attempts": len(RETRY_DELAYS) + 1, "retry_at": self.retry_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Job":
        job = cls(d.get("kind", "?"), d.get("label", ""), d.get("project_id"), d.get("request"))
        job.id = str(d.get("id") or job.id)
        job.status, job.result, job.error = d.get("status", "error"), d.get("result"), d.get("error")
        job.stage, job.stages, job.fraction, job.note = d.get("stage", ""), list(d.get("stages") or []), d.get("fraction"), d.get("note", "")
        job.created, job.started, job.finished = d.get("created") or job.created, d.get("started"), d.get("finished")
        job.attempt = int(d.get("attempt") or 0)
        job.log.extend(d.get("log") or [])
        return job


class _JobLog(io.TextIOBase):
    """File-like that turns everything printed during a job into log lines (and echoes to the console and the log file)."""

    def __init__(self, job: Job):
        self.job, self._buf = job, ""
        self._file_log = logsetup.get("job")

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, s: str) -> int:
        with contextlib.suppress(Exception):
            if sys.__stdout__:
                sys.__stdout__.write(s)
        self._buf += s
        while chr(10) in self._buf:
            line, self._buf = self._buf.split(chr(10), 1)
            line = line.rstrip(chr(13))
            if line.strip():
                self.job.log.append(line)
                self._file_log.info("[%s] %s", self.job.id, line)
        return len(s)

    def flush(self) -> None:
        pass


JOBS: Dict[str, Job] = {}
_QUEUE: "queue.Queue" = queue.Queue()
_LOCK = threading.Lock()


def _jobs_file() -> Path:
    return Path(config.LOCAL_OUTPUT_DIR) / "state" / "jobs.json"


_SAVE_LOCK = threading.Lock()      # the worker and the request threads both save: one writer at a time (Windows refuses a rename onto a busy file)


def _save_jobs() -> None:
    """Best effort: a failure to save history must never break a job."""
    try:
        with _SAVE_LOCK:
            with _LOCK:                    # snapshot inside the save lock, so the newest state is always the one written last
                jobs = sorted(JOBS.values(), key=lambda j: j.created)[-JOB_HISTORY:]
                data = [j.to_dict(tail=120) for j in jobs]
            path = _jobs_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            for attempt in range(5):       # an antivirus scan or a file indexer can hold the file for a moment
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
    except Exception:  # noqa: BLE001
        log.exception("could not save the job history")


def _load_jobs() -> None:
    """Called once at start-up. Anything that was queued or running when the app closed is 'interrupted'."""
    try:
        data = json.loads(_jobs_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for d in data if isinstance(data, list) else []:
        try:
            job = Job.from_dict(d)
        except Exception:  # noqa: BLE001
            continue
        if job.status in ("queued", "running"):
            job.status = "interrupted"
            job.error = "The app was closed while this was running. Everything it had finished is saved: press Try again to continue."
            job.finished = job.finished or time.time()
        JOBS[job.id] = job


def _explain(e: BaseException) -> str:
    if isinstance(e, FileNotFoundError) and "ffmpeg" in str(e).lower():
        return "ffmpeg was not found. Install ffmpeg and make sure it is on your PATH."
    return str(e) or e.__class__.__name__


def _retry_delay(job: Job, e: BaseException) -> Optional[float]:
    """Seconds to wait before an automatic new attempt, or None when this failure should be shown to the person now."""
    if not isinstance(e, llm.LLMError) or e.kind not in ("overloaded", "network") or job.attempt >= len(RETRY_DELAYS):
        return None
    if job.kind != "kit" and not (job.kind == "analyze" and job.stage in RETRY_STAGES):
        return None
    return float(RETRY_DELAYS[job.attempt])


def _worker() -> None:
    while True:
        job, fn = _QUEUE.get()
        if job.status == "cancelled":                  # cancelled while it waited
            job.finished = time.time()
            _save_jobs()
            continue
        job.status, job.started, job.retry_at = "running", time.time(), None
        _save_jobs()
        control.bind(job.cancel, job.progress)
        requeued = False
        try:
            with contextlib.redirect_stdout(_JobLog(job)):
                job.result = fn()
            job.status = "done"
        except control.Cancelled:
            job.status = "cancelled"
            job.log.append("Cancelled.")
        except Exception as e:  # noqa: BLE001 — shown to the user in the page
            delay = _retry_delay(job, e)
            if delay is not None:
                job.attempt += 1
                job.retry_at = time.time() + delay
                job.status, requeued = "queued", True
                job.note = f"The AI service is busy or unreachable. Trying again automatically in {max(1, round(delay / 60))} min (attempt {job.attempt + 1} of {len(RETRY_DELAYS) + 1})."
                job.log.append(job.note)
                log.warning("job %s (%s): %s -> automatic retry in %.0fs", job.id, job.kind, str(e)[:160], delay)
                timer = threading.Timer(delay, _QUEUE.put, args=((job, fn),))
                timer.daemon = True
                timer.start()
            else:
                job.error = _explain(e)
                job.log.append("ERROR: " + job.error)
                expected = isinstance(e, (SourceError, ValueError, llm.LLMError, health.LowDiskSpace, youtube.YouTubeError))
                log.error("job %s (%s) failed: %s", job.id, job.kind, job.error, exc_info=not expected)
                if not expected:
                    traceback.print_exc()
                job.status = "error"
        finally:
            control.unbind()
            if not requeued:
                job.finished = time.time()
                log.info("job %s (%s) -> %s in %.0fs", job.id, job.kind, job.status, job.finished - (job.started or job.finished))
            _save_jobs()
            housekeeping.sweep()                          # posted clips and leftovers go as soon as the work is done


_worker_thread: Optional[threading.Thread] = None


def start_worker() -> None:
    """Start the single job worker (idempotent: the server and the tests may both ask)."""
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker, daemon=True, name="job-worker")
        _worker_thread.start()


def _submit(kind: str, label: str, fn: Callable[[], Dict], project_id: Optional[str] = None,
            request_info: Optional[Dict] = None) -> Job:
    with _LOCK:
        waiting = [j for j in JOBS.values() if j.status in ("queued", "running")]
        if request_info is not None:                   # a double click must not start the same work twice
            for j in waiting:
                if j.kind == kind and j.project_id == project_id and j.request == request_info:
                    return j
        if len(waiting) >= MAX_WAITING_JOBS:
            abort(429, "Too many jobs are waiting. Let some finish, or cancel a few.")
        job = Job(kind, label, project_id, request_info)
        JOBS[job.id] = job
        for old in sorted(JOBS.values(), key=lambda j: j.created)[:-JOB_HISTORY]:    # keep memory bounded
            if old.status not in ("queued", "running"):
                JOBS.pop(old.id, None)
    _QUEUE.put((job, fn))
    _save_jobs()
    return job


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _body() -> Dict:
    data = request.get_json(silent=True, force=True)
    if not isinstance(data, dict):
        abort(400, "JSON object expected")
    return data


def _project_or_404(pid: str) -> Dict:
    try:
        return P.load_project(pid)
    except KeyError:
        abort(404, "unknown project")


def _int(value, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _analyze_options(d: Dict) -> pipeline.AnalyzeOptions:
    language = str(d.get("language", "") or "").strip().lower() or None
    if language and not re.match(r"^[a-z]{2,3}(-[a-z]{2,4})?$", language):
        abort(400, "bad language code")
    lo, hi = _int(d.get("min_seconds"), 20, 5, 170), _int(d.get("max_seconds"), 60, 10, 180)
    if lo >= hi:
        abort(400, "the shortest clip must be shorter than the longest")
    return pipeline.AnalyzeOptions(
        candidates=_int(d.get("candidates"), 10, 3, pipeline.MAX_CANDIDATES), language=language,
        focus=" ".join(str(d.get("focus", "") or "").split())[:300], min_seconds=lo, max_seconds=hi,
        use_audio=bool(d.get("use_audio", True)), use_replay=bool(d.get("use_replay", True)),
        refresh=bool(d.get("refresh", False)),
    )


def _render_options(d: Dict) -> RenderOptions:
    layout = str(d.get("layout", "auto"))
    style = str(d.get("caption_style", "pop"))
    edit = str(d.get("edit", "off"))
    if edit not in editing.PRESET_NAMES:
        abort(400, "bad editing style")
    effects = d.get("effects")
    if effects is not None and (not isinstance(effects, dict) or any(k not in editing.EFFECTS or not isinstance(v, bool) for k, v in effects.items())):
        abort(400, "bad effects")
    aspect = str(d.get("aspect_ratio", "9:16"))
    if layout not in LAYOUTS:
        abort(400, "bad layout")
    if style not in STYLES:
        abort(400, "bad caption style")
    if not re.match(r"^\d{1,2}:\d{1,2}$", aspect) or aspect.split(":")[0] == "0" or aspect.split(":")[1] == "0":
        abort(400, "bad aspect ratio")
    return RenderOptions(
        aspect_ratio=aspect, layout=layout, captions=bool(d.get("captions", False)), caption_style=style,
        title_card=bool(d.get("title_card", False)), watermark=str(d.get("watermark", "") or "")[:40],
        normalize_audio=bool(d.get("normalize_audio", True)), edit=edit, effects=effects or None,
    )


def _quality(d: Dict) -> int:
    q = _int(d.get("quality"), 1080, 360, 2160)
    return q


def _upload_options(d: Dict) -> Dict:
    """Validated upload choices. ``start_in_hours`` is resolved when the clips are queued (see _resolve_schedule), so an
    autopilot run that analyses and renders for twenty minutes still starts its schedule from the moment of the upload."""
    privacy = str(d.get("privacy") or config.youtube_privacy())
    if privacy not in uploads.PRIVACIES:
        abort(400, "visibility must be private, unlisted or public")
    out: Dict = {"privacy": privacy, "force": bool(d.get("force")), "every_hours": 24.0, "schedule_start": None, "start_in_hours": None}
    try:
        if d.get("every_hours") not in (None, ""):
            out["every_hours"] = float(d["every_hours"])
        if d.get("start_in_hours") not in (None, ""):
            out["start_in_hours"] = float(d["start_in_hours"])
        if d.get("schedule_start") not in (None, ""):
            ts = float(d["schedule_start"])
            out["schedule_start"] = ts / 1000 if ts > 1e11 else ts            # the browser sends milliseconds
    except (TypeError, ValueError):
        abort(400, "bad schedule values")
    if not (0.25 <= out["every_hours"] <= 24 * 30):
        abort(400, "the gap between clips must be between 15 minutes and 30 days")
    if out["start_in_hours"] is not None and not (0 <= out["start_in_hours"] <= 24 * 30):
        abort(400, "the first clip can go live between now and 30 days from now")
    return out


def _resolve_schedule(opts: Dict) -> Dict:
    """kwargs for uploads.enqueue: ``start_in_hours`` becomes an absolute time now."""
    o = dict(opts)
    hours = o.pop("start_in_hours", None)
    if hours is not None and o.get("schedule_start") is None:
        o["schedule_start"] = time.time() + max(hours, 0.25) * 3600
    if o.get("schedule_start") is None:
        o["every_hours"] = 24.0
    return o


def _require_youtube() -> None:
    st = youtube.status()
    if st["needs_reconnect"]:
        abort(400, "The YouTube sign-in expired. Open Settings → YouTube and press Reconnect first.")
    if not st["connected"]:
        abort(400, "Connect your YouTube channel first (Settings → YouTube).")


def _public_project(p: Dict) -> Dict:
    """Project for the browser: no raw transcript, but a text excerpt per candidate and media URLs."""
    pid = p["id"]
    transcript = {}
    with contextlib.suppress(KeyError, OSError, ValueError):
        transcript = P.load_transcript(p)
    src = dict(p.get("source", {}))
    heat = src.pop("heatmap", None)
    local_path = src.pop("local_path", None)
    src["has_heatmap"] = bool(heat)
    src["local_available"] = bool(local_path and os.path.isfile(local_path))
    src["thumbnail"] = f"https://i.ytimg.com/vi/{src.get('id')}/mqdefault.jpg" if src.get("kind") == "youtube" else None
    out = {k: v for k, v in p.items() if k not in ("transcript", "source", "source_path")}
    out["source"] = src
    out["source_media"] = f"/media/{pid}/source" if src["local_available"] else None
    out["candidates"] = [
        {**c, "excerpt": _excerpt(transcript, float(c["start_time"]), float(c["end_time"]), limit=260) if transcript else ""}
        for c in p.get("candidates", [])
    ]
    folder = P.project_dir(pid)
    sent = uploads.load(pid)
    out["clips"] = [
        {**c, "url": f"/media/{pid}/clip/{c['file']}" if c.get("file") and (folder / c["file"]).is_file() else None,
         "youtube": sent.get(c.get("file") or "")}
        for c in p.get("clips", [])
    ]
    return out


def _yt_status() -> Dict:
    """youtube.status() plus how many clips are waiting: everything the page shows about the channel."""
    return {**youtube.status(), "queued": uploads.queue_size()}


def _settings_view() -> Dict:
    return {
        "llm_provider": config.llm_provider(),
        "gemini_model": config.gemini_model(),
        "openai_model": config.openai_model(),
        "gemini_key_set": bool(config.gemini_key()),
        "openai_key_set": bool(config.openai_key()),
        "ytdlp_cookies_browser": config.env("YTDLP_COOKIES_BROWSER"),
        "ytdlp_cookies_file": config.env("YTDLP_COOKIES_FILE"),
        "ytdlp_proxy_set": bool(config.env("YTDLP_PROXY")),            # may hold credentials: never sent back
        "whisper_model": config.whisper_model(),
        "encoder": (config.env("LOCAL_ENCODER") or "auto").lower(),
        "caption_script": config.caption_script(),
        "facecam_bg": config.facecam_background(),
        "delete_uploaded": config.delete_posted_clips(), "cache_days": config.cache_days(),
        "memes": memes.summary(),
        "youtube_client_id": config.youtube_client_id(),
        "youtube_client_secret_set": bool(config.youtube_client_secret()),             # a secret: never sent back
        "youtube_privacy": config.youtube_privacy(), "youtube_category": config.youtube_category(),
        "youtube_daily_uploads": config.youtube_daily_uploads(), "youtube_auto_resume": config.youtube_auto_resume(),
        "youtube_notify_subscribers": config.youtube_notify_subscribers(),
    }


def _explorer_open(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))                                    # noqa: S606 — a folder we created
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# --------------------------------------------------------------------------- #
# Routes: pages, settings
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    if request.args.get("state") and ("code" in request.args or "error" in request.args):
        return oauth_youtube_callback()                      # Google sends the browser back to "/?code=...&state=...": the YouTube sign-in
    return send_file(WEBUI / "index.html")


@app.get("/static/<name>")
def static_file(name: str):
    mime = STATIC_FILES.get(name)
    if mime is None:
        abort(404)
    resp = send_file(WEBUI / name, mimetype=mime)
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/api/config")
def api_config():
    from shorts_generator.local.clipper import _captions_supported

    return jsonify({
        **_settings_view(), "version": config.version(),
        "output_dir": str(Path(config.LOCAL_OUTPUT_DIR).resolve()),
        "layouts": list(LAYOUTS), "caption_styles": list(STYLES), "edit_presets": list(editing.PRESET_NAMES),
        "ffmpeg": bool(shutil.which("ffmpeg")), "captions_supported": _captions_supported(),
        "whisper_installed": importlib.util.find_spec("faster_whisper") is not None,
        "youtube": _yt_status(),
    })


@app.get("/api/settings")
def api_settings():
    return jsonify(_settings_view())


@app.post("/api/settings")
def api_settings_save():
    d = _body()
    mapping = {
        "llm_provider": "LLM_PROVIDER", "gemini_model": "GEMINI_MODEL", "openai_model": "OPENAI_MODEL",
        "gemini_api_key": "GEMINI_API_KEY", "openai_api_key": "OPENAI_API_KEY",
        "ytdlp_cookies_browser": "YTDLP_COOKIES_BROWSER", "ytdlp_cookies_file": "YTDLP_COOKIES_FILE",
        "ytdlp_proxy": "YTDLP_PROXY", "whisper_model": "LOCAL_WHISPER_MODEL", "encoder": "LOCAL_ENCODER",
        "caption_script": "LOCAL_CAPTION_SCRIPT", "facecam_bg": "LOCAL_FACECAM_BG",
        "delete_uploaded": "LOCAL_DELETE_UPLOADED", "cache_days": "LOCAL_CACHE_DAYS",
        "youtube_client_id": "YOUTUBE_CLIENT_ID", "youtube_client_secret": "YOUTUBE_CLIENT_SECRET", "youtube_privacy": "YOUTUBE_PRIVACY",
        "youtube_category": "YOUTUBE_CATEGORY", "youtube_daily_uploads": "YOUTUBE_DAILY_UPLOADS",
        "youtube_auto_resume": "YOUTUBE_AUTO_RESUME", "youtube_notify_subscribers": "YOUTUBE_NOTIFY_SUBSCRIBERS",
    }
    clearable = ("ytdlp_cookies_browser", "ytdlp_cookies_file", "ytdlp_proxy", "youtube_client_id")       # an empty value means "turn it off"
    values = {env_name: str(d[k]) for k, env_name in mapping.items() if k in d and (d[k] != "" or k in clearable)}
    try:
        config.save_settings(values)
    except ValueError as e:
        abort(400, str(e))
    if "LOCAL_ENCODER" in values:
        encoder.reset()                                          # try the new choice on the next render
    return jsonify(_settings_view())


@app.post("/api/settings/test")
def api_settings_test():
    return jsonify(llm.test_connection())


@app.get("/api/models")
def api_models():
    if config.llm_provider() != "gemini" or not config.gemini_key():
        return jsonify({"models": []})
    try:
        return jsonify({"models": llm.gemini_models(refresh=request.args.get("refresh") == "1")})
    except llm.LLMError as e:                                  # a rejected key or no internet: the dialog just shows "Auto"
        return jsonify({"models": [], "error": str(e)})


@app.get("/api/storage")
def api_storage():
    return jsonify(storage.usage())


@app.post("/api/storage/clean")
def api_storage_clean():
    what = [w for w in _body().get("what", []) if w in ("legacy_downloads", "cache", "llm_cache")]
    return jsonify({"freed": storage.clean(what), "usage": storage.usage()})


# --------------------------------------------------------------------------- #
# Routes: projects
# --------------------------------------------------------------------------- #
@app.get("/api/projects")
def api_projects():
    out = P.list_projects()
    for p in out:                                              # what happened to this video's clips on YouTube
        c = uploads.counts(p["id"])
        p["n_uploaded"], p["n_queued"] = c.get("uploaded", 0), c.get("queued", 0) + c.get("uploading", 0)
    return jsonify(out)


@app.get("/api/projects/<pid>")
def api_project(pid: str):
    return jsonify(_public_project(_project_or_404(pid)))


@app.delete("/api/projects/<pid>")
def api_project_delete(pid: str):
    _project_or_404(pid)
    return jsonify({"freed": P.delete_project(pid, delete_cache=request.args.get("cache") == "1")})


@app.post("/api/projects/<pid>/open-folder")
def api_open_folder(pid: str):
    _project_or_404(pid)
    folder = P.project_dir(pid)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        _explorer_open(folder.resolve())
    except OSError as e:
        abort(400, f"Could not open the folder: {e}")
    return jsonify({"opened": str(folder.resolve())})


@app.get("/api/projects/<pid>/signals")
def api_signals(pid: str):
    """Series for the timeline: voice loudness, 'most replayed', and where the candidates sit."""
    p = _project_or_404(pid)
    energy = []
    path = P.cache_dir(pid) / "energy.json"
    if path.exists():
        with contextlib.suppress(OSError, ValueError, KeyError):
            import json as _json
            energy = downsample(_json.loads(path.read_text(encoding="utf-8"))["db"], 400)
    replay = downsample(heatmap_series(p.get("source", {}).get("heatmap"), float(p.get("duration") or 0)), 400)
    return jsonify({
        "duration": p.get("duration"), "energy": energy, "replay": replay,
        "candidates": [{"id": c["id"], "start": c["start_time"], "end": c["end_time"], "score": c.get("score")}
                       for c in p.get("candidates", [])],
    })


@app.post("/api/projects/<pid>/candidates")
def api_add_candidate(pid: str):
    p = _project_or_404(pid)
    d = _body()
    try:
        start, end = float(d.get("start")), float(d.get("end"))
    except (TypeError, ValueError):
        abort(400, "start and end are required (seconds)")
    duration = float(p.get("duration") or 0)
    if not (0 <= start < end) or (duration and end > duration + 1) or not (3 <= end - start <= 180):
        abort(400, "use a start before the end, inside the video, 3 to 180 seconds long")
    made: Dict = {}
    P.update_project(p, lambda fresh: made.update(P.add_manual_candidate(fresh, start, end, str(d.get("title", ""))[:100])))
    return jsonify(made), 201


@app.post("/api/projects/<pid>/reanalyze")
def api_reanalyze(pid: str):
    p = _project_or_404(pid)
    body = _body()
    opts = _analyze_options(body)

    def run() -> Dict:
        fresh = P.load_project(pid)
        pipeline.reanalyze_local(fresh, opts)
        return {"project_id": pid}

    info = {"path": f"/api/projects/{pid}/reanalyze", "body": body}
    return jsonify(_submit("analyze", p.get("title") or pid, run, pid, info).to_dict()), 202


@app.post("/api/projects/<pid>/render")
def api_render(pid: str):
    p = _project_or_404(pid)
    d = _body()
    n = len(p["candidates"])
    try:
        ids = sorted({int(i) for i in d.get("ids", [])})
    except (TypeError, ValueError):
        abort(400, "ids must be integers")
    if not ids or len(ids) > 12 or any(i < 0 or i >= n for i in ids):
        abort(400, "pick between 1 and 12 existing candidates")
    options = _render_options(d)
    overrides: Dict[int, tuple] = {}
    for key, val in (d.get("overrides") or {}).items():
        try:
            cid = int(key)
            overrides[cid] = (float(val["start"]), float(val["end"]))
        except (TypeError, ValueError, KeyError):
            abort(400, "bad trim values")
        if cid not in ids:
            abort(400, "trim given for a clip that is not selected")
    quality, kit = _quality(d), bool(d.get("kit", False))

    def run() -> Dict:
        fresh = P.load_project(pid)        # re-read: an earlier queued job may have updated it
        pipeline.render_local(fresh, ids, options, quality=quality, overrides=overrides, kit=kit)
        return {"project_id": pid}

    info = {"path": f"/api/projects/{pid}/render", "body": d}
    return jsonify(_submit("render", f"{p.get('title') or pid}: {len(ids)} clip(s)", run, pid, info).to_dict()), 202


@app.post("/api/projects/<pid>/kit")
def api_kit(pid: str):
    p = _project_or_404(pid)
    d = _body()
    try:
        ids = sorted({int(i) for i in d.get("ids", [])})
    except (TypeError, ValueError):
        abort(400, "ids must be integers")
    if not ids or len(ids) > 20 or any(i < 0 or i >= len(p["candidates"]) for i in ids):
        abort(400, "pick between 1 and 20 existing candidates")

    def run() -> Dict:
        pipeline.publish_kit_local(P.load_project(pid), ids)
        return {"project_id": pid}

    info = {"path": f"/api/projects/{pid}/kit", "body": d}
    return jsonify(_submit("kit", f"{p.get('title') or pid}: titles and hashtags", run, pid, info).to_dict()), 202


@app.post("/api/analyze")
def api_analyze():
    d = _body()
    urls = d.get("urls") if isinstance(d.get("urls"), list) else [d.get("url", "")]
    urls = [str(u).strip() for u in urls if str(u).strip()]
    if not urls or len(urls) > 10 or any(len(u) > 2000 for u in urls):
        abort(400, "paste between 1 and 10 video links or file paths")
    opts = _analyze_options(d)
    auto = _int(d.get("auto_render"), 0, 0, 12)
    render_opts = _render_options(d.get("render") or {}) if auto else None
    quality = _quality(d.get("render") or {}) if auto else 1080
    min_score = _int(d.get("auto_min_score"), 0, 0, 100)
    # Autopilot: after rendering, the clips go to the connected YouTube channel without anyone clicking again.
    upload_opts = _upload_options(d["upload"]) if isinstance(d.get("upload"), dict) else None
    if upload_opts is not None:
        if not auto:
            abort(400, "Autopilot needs the number of clips to render.")
        _require_youtube()
    with_kit = bool((d.get("render") or {}).get("kit")) or upload_opts is not None     # uploads need a title and hashtags

    jobs = []
    for url in urls:
        def run(url: str = url) -> Dict:
            if not auto:
                project = pipeline.analyze_local(url, opts)
                return {"project_id": project["id"]}
            with control.scale(0.0, 0.4 if upload_opts else 0.5):
                project = pipeline.analyze_local(url, opts)
            taken = uploads.claimed_ranges(P.load_project(project["id"])) if upload_opts else []
            scored = [c for c in project["candidates"] if c.get("score") is None or c["score"] >= min_score]
            ranked = [c for c in scored if not uploads.overlaps(float(c["start_time"]), float(c["end_time"]), taken)]
            ids = [c["id"] for c in ranked[:auto]]
            if not ids and scored:
                raise ValueError("Every good moment of this video is already on your channel (or waiting to go). "
                                 "Lower the minimum score, or search again with a different focus.")
            if not ids:
                raise ValueError(f"None of the moments scored {min_score} or more. Lower the minimum score, or choose the clips yourself.")
            with control.scale(0.4 if upload_opts else 0.5, 0.45 if upload_opts else 0.5):
                records = pipeline.render_local(project, ids, render_opts, quality=quality, kit=with_kit)
            out: Dict = {"project_id": project["id"]}
            if upload_opts:
                files = [r["file"] for r in records if r.get("file")]
                if not files:
                    raise ValueError("No clip could be rendered, so there is nothing to upload.")
                queued = uploads.enqueue(P.load_project(project["id"]), files, **_resolve_schedule(upload_opts))
                with control.scale(0.85, 0.15):
                    out.update(pipeline.upload_local(project["id"]))
                out["queued"] = len(queued["queued"])
                out["skipped"] = queued["skipped"]
            return out

        one = {**d, "urls": [url]}
        one.pop("url", None)
        jobs.append(_submit("analyze", url, run, None, {"path": "/api/analyze", "body": one}).to_dict())
    return jsonify({"jobs": jobs}), 202


# --------------------------------------------------------------------------- #
# Routes: jobs, media
# --------------------------------------------------------------------------- #
@app.delete("/api/projects/<pid>/clips/<name>")
def api_clip_delete(pid: str, name: str):
    p = _project_or_404(pid)
    if not SAFE_CLIP.match(name):
        abort(404)
    def drop(fresh: Dict) -> None:
        fresh["clips"] = [c for c in fresh.get("clips", []) if c.get("file") != name]

    with contextlib.suppress(FileNotFoundError):
        (P.project_dir(pid) / name).unlink()
    P.update_project(p, drop)
    return jsonify({"deleted": name})


def _posts_text(p: Dict, clips: List[Dict]) -> str:
    """The text to paste next to each clip (or just its title when no publish kit was written)."""
    lines = [f"{p.get('title', '')}", "=" * 60, ""]
    for c in clips:
        cand = p["candidates"][c["candidate"]] if c.get("candidate") is not None and c["candidate"] < len(p["candidates"]) else {}
        kit = cand.get("kit") or {}
        lines += [f"{c['file']}   ({c.get('start', 0):.0f}s-{c.get('end', 0):.0f}s)", f"Title: {kit.get('title') or c.get('title', '')}"]
        if kit:
            lines += [f"Description: {kit.get('description', '')}", "Hashtags: " + " ".join(kit.get("hashtags", []))]
        lines.append("")
    return "\n".join(lines)


def _clips_csv(p: Dict, clips: List[Dict]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["file", "title", "start_s", "end_s", "score", "post_title", "post_description", "hashtags"])
    for c in clips:
        cand = p["candidates"][c["candidate"]] if c.get("candidate") is not None and c["candidate"] < len(p["candidates"]) else {}
        kit = cand.get("kit") or {}
        w.writerow([c["file"], c.get("title", ""), round(c.get("start", 0), 1), round(c.get("end", 0), 1), cand.get("score", ""),
                    kit.get("title", ""), kit.get("description", ""), " ".join(kit.get("hashtags", []))])
    return chr(0xFEFF) + out.getvalue()                       # a BOM, so Excel reads the UTF-8 titles correctly


@app.get("/api/projects/<pid>/export.zip")
def api_export(pid: str):
    """Every rendered clip + posts.txt + clips.csv in one download (mp4 is already compressed: stored, not deflated)."""
    p = _project_or_404(pid)
    folder = P.project_dir(pid)
    clips = [c for c in p.get("clips", []) if c.get("file") and (folder / c["file"]).is_file()]
    if not clips:
        abort(404, "nothing has been rendered yet")
    tmp = tempfile.NamedTemporaryFile(prefix="export_", suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as z:
        for c in clips:
            z.write(folder / c["file"], c["file"])
        z.writestr("posts.txt", _posts_text(p, clips))
        z.writestr("clips.csv", _clips_csv(p, clips))
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", p.get("title") or pid)[:60].strip("_") or pid
    resp = send_file(tmp.name, mimetype="application/zip", as_attachment=True, download_name=f"{name}.zip")
    resp.call_on_close(lambda: Path(tmp.name).unlink(missing_ok=True))
    return resp


# --------------------------------------------------------------------------- #
# Routes: your YouTube channel (sign-in, uploads)
# --------------------------------------------------------------------------- #
@app.get("/api/youtube")
def api_youtube():
    return jsonify(_yt_status())


@app.post("/api/youtube/credentials")
def api_youtube_credentials():
    """The Google OAuth client: either the downloaded client_secret JSON (as text) or the two values."""
    d = _body()
    try:
        cid, secret = youtube.parse_client_json(d["json"]) if d.get("json") else (str(d.get("client_id", "")).strip(), str(d.get("client_secret", "")).strip())
        if not cid or not secret:
            raise ValueError("Both the client id and the client secret are needed.")
        config.save_settings({"YOUTUBE_CLIENT_ID": cid, "YOUTUBE_CLIENT_SECRET": secret})
    except ValueError as e:
        abort(400, str(e))
    return jsonify(_yt_status())


@app.post("/api/youtube/connect")
def api_youtube_connect():
    """Returns the Google page to open. Google sends the browser back to / (or the alias /oauth/youtube/callback) on this same server."""
    if not youtube.configured():
        abort(400, "Add your Google client id and secret first.")
    port = request.environ.get("SERVER_PORT") or "8765"
    return jsonify({"url": youtube.auth_url(f"http://127.0.0.1:{port}{youtube.REDIRECT_PATH}")})


@app.get(youtube.CALLBACK_PATH)
def oauth_youtube_callback():
    try:
        channel = youtube.finish_auth(request.args.get("state", ""), request.args.get("code"), request.args.get("error"))
        title, body, ok = "YouTube connected", ("Connected to " + html.escape(channel["title"]) + "." if channel.get("title") else "Connected."), True
    except youtube.YouTubeError as e:
        title, body, ok = "Could not connect", html.escape(str(e)), False
    page = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{title}</title>"
            "<body style=\"font:16px system-ui,sans-serif;max-width:520px;margin:12vh auto;padding:0 20px;line-height:1.5\">"
            f"<h2>{'✓ ' if ok else '✗ '}{title}</h2><p>{body}</p>"
            "<p>" + ("You can close this tab and go back to Shorts Studio." if ok else "Go back to Shorts Studio → Settings → YouTube and try again.") +
            "</p><p><a href='/'>Open Shorts Studio</a></p>")
    resp = Response(page, mimetype="text/html")
    resp.headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
    resp.headers["Cache-Control"] = "no-store"
    return resp, 200 if ok else 400


@app.post("/api/youtube/check")
def api_youtube_check():
    """Does Google accept the saved client id + secret? (Run after saving them, before the sign-in.)"""
    return jsonify(youtube.check_client())


@app.post("/api/youtube/pause")
def api_youtube_pause():
    youtube.set_paused(bool(_body().get("paused")))
    return jsonify(_yt_status())


@app.get("/api/uploads")
def api_uploads():
    """Everything uploaded, uploading or waiting, across all videos (the YouTube page)."""
    return jsonify({"items": uploads.all_records(), "status": _yt_status()})


@app.post("/api/youtube/resume")
def api_youtube_resume():
    """"Try now": stop waiting after YouTube said "not now" and send the queue again. If YouTube still says no, it waits again."""
    _require_youtube()
    if youtube.status()["paused"]:
        abort(400, "Uploads are paused: press Resume first.")
    if youtube.uploads_left() <= 0:
        abort(400, "Your own limit of uploads per day is reached: raise it in the upload settings, or wait for tomorrow.")
    youtube.unpark()
    started = _upload_tick(asked=True)
    jobs = [j.to_dict() for j in list(JOBS.values()) if j.kind == "upload" and j.status in ("queued", "running")]
    return jsonify({"started": started, "jobs": jobs, **_yt_status()})


@app.post("/api/youtube/queue/clear")
def api_youtube_queue_clear():
    return jsonify({"removed": uploads.dequeue_everywhere()})


@app.post("/api/youtube/disconnect")
def api_youtube_disconnect():
    youtube.disconnect(revoke=True)
    return jsonify(_yt_status())


@app.post("/api/projects/<pid>/upload")
def api_upload(pid: str):
    """Queue the chosen rendered clips and start uploading them (in the job queue, so progress and Cancel work)."""
    p = _project_or_404(pid)
    d = _body()
    files = d.get("files")
    if not isinstance(files, list) or not files or len(files) > 60 or not all(isinstance(f, str) and SAFE_CLIP.match(f) for f in files):
        abort(400, "pick between 1 and 60 rendered clips")
    opts = _upload_options(d)
    _require_youtube()
    meta = d.get("meta")
    if meta is not None and (not isinstance(meta, dict) or any(not isinstance(v, dict) for v in meta.values())):
        abort(400, "meta must map clip files to {title, description, hashtags}")
    try:
        queued = uploads.enqueue(p, files, overrides=meta, **_resolve_schedule(opts))
    except ValueError as e:
        abort(400, str(e))
    job = _upload_job(pid, p.get("title") or pid, len(queued["queued"])) if queued["queued"] else None
    return jsonify({"job": job.to_dict() if job else None, **queued}), 202 if job else 200


def _upload_job(pid: str, title: str, n: int = 0) -> Job:
    def run() -> Dict:
        return {"project_id": pid, **pipeline.upload_local(pid)}

    label = f"{title}: {n} clip(s) to YouTube" if n else f"{title}: queued clips to YouTube"
    return _submit("upload", label, run, pid, {"path": f"/api/projects/{pid}/upload/run", "body": {}})


@app.post("/api/projects/<pid>/upload/run")
def api_upload_run(pid: str):
    """Start uploading whatever is queued for this project (also what "Try again" and the watcher send)."""
    p = _project_or_404(pid)
    _require_youtube()
    if not uploads.counts(pid).get("queued"):
        abort(400, "Nothing is waiting to be uploaded.")
    return jsonify(_upload_job(pid, p.get("title") or pid).to_dict()), 202


@app.post("/api/projects/<pid>/upload/dequeue")
def api_upload_dequeue(pid: str):
    """Take clips out of the queue before they are uploaded. {"files": [...]} or {"all": true}."""
    _project_or_404(pid)
    d = _body()
    files = d.get("files")
    if not d.get("all") and (not isinstance(files, list) or not all(isinstance(f, str) and SAFE_CLIP.match(f) for f in files)):
        abort(400, "give the clip files, or all=true")
    return jsonify({"removed": uploads.dequeue(pid, None if d.get("all") else files)})


# The watcher: queued uploads that had to wait (daily cap, YouTube's quota, no internet) continue by themselves.
def _upload_tick(asked: bool = False) -> int:
    """Start a drain job for every project with due queued clips, if YouTube can take them now. Returns how many were started.

    ``asked``: the person pressed "Try now" (their press counts even when continuing by itself is switched off).
    """
    if not (asked or config.youtube_auto_resume()):
        return 0
    st = youtube.status()
    if not st["connected"] or st["paused"] or st["parked_until"] or st["uploads_left"] <= 0:
        return 0
    started = 0
    for pid in uploads.pending_projects():
        try:
            known = set(JOBS)
            job = _upload_job(pid, P.load_project(pid).get("title") or pid)      # the duplicate guard returns the job already waiting
            started += job.id not in known
        except Exception:  # noqa: BLE001 - a full queue (HTTP 429) or a deleted project: try again next minute
            log.debug("upload watcher skipped %s", pid, exc_info=True)
    return started


def _upload_watcher(period: float = 60.0) -> None:
    rounds = 0
    while True:
        if rounds % 180 == 0:                             # at start, then every three hours
            housekeeping.sweep()
        rounds += 1
        time.sleep(period)
        try:
            _upload_tick()
        except Exception:  # noqa: BLE001
            log.exception("upload watcher failed")


_watcher_thread: Optional[threading.Thread] = None


def start_upload_watcher() -> None:
    global _watcher_thread
    if _watcher_thread is None or not _watcher_thread.is_alive():
        _watcher_thread = threading.Thread(target=_upload_watcher, daemon=True, name="upload-watcher")
        _watcher_thread.start()


# --------------------------------------------------------------------------- #
# Routes: the app itself (start with Windows, quit), the memes folder
# --------------------------------------------------------------------------- #
@app.get("/api/app")
def api_app():
    return jsonify({"supported": desktop.supported(), "autostart": desktop.autostart_enabled()})


@app.post("/api/app/autostart")
def api_app_autostart():
    if not desktop.supported():
        abort(400, "Starting with the computer is only set up for Windows.")
    desktop.set_autostart(bool(_body().get("on")))
    return jsonify({"autostart": desktop.autostart_enabled()})


@app.post("/api/quit")
def api_quit():
    if any(j.status == "running" for j in JOBS.values()):
        abort(409, "Something is still being made or uploaded. Wait for it, or cancel it first.")
    flag = ROOT / "output" / "state" / "quit.flag"            # tells the keeper (launcher.pyw) not to start the app again
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("quit", encoding="utf-8")
    log.info("Quit asked from the page")
    timer = threading.Timer(0.8, _exit_process)
    timer.daemon = True
    timer.start()
    return jsonify({"ok": True})


def _exit_process() -> None:
    _save_jobs()
    os._exit(0)


@app.post("/api/memes/open")
def api_memes_open():
    _explorer_open(memes.ensure_folder())
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# Routes: health, updates, diagnostics
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def api_health():
    return jsonify(health.doctor(deep=request.args.get("deep") == "1"))


@app.post("/api/ytdlp/update")
def api_ytdlp_update():
    def run() -> Dict:
        print(f"[update] yt-dlp {health.ytdlp_version()}: asking pip for the newest version")
        result = health.update_ytdlp()
        print(f"[update] yt-dlp {result['before']} -> {result['after']}")
        return result

    return jsonify(_submit("update", "Update yt-dlp", run, None, {"path": "/api/ytdlp/update", "body": {}}).to_dict()), 202


@app.get("/api/diagnostics.zip")
def api_diagnostics():
    """Logs + health + versions + recent jobs, never any key: attach it to a bug report."""
    jobs = [j.to_dict(tail=60) for j in sorted(JOBS.values(), key=lambda j: j.created)[-20:]]
    data = health.diagnostics_zip(logsetup.log_dir(), jobs)
    return send_file(io.BytesIO(data), mimetype="application/zip", as_attachment=True,
                     download_name=f"shorts-studio-diagnostics-{time.strftime('%Y%m%d-%H%M')}.zip")


@app.get("/api/jobs")
def api_jobs():
    jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)
    return jsonify([j.to_dict(tail=0) for j in jobs])


@app.get("/api/jobs/<jid>")
def api_job(jid: str):
    job = JOBS.get(jid)
    if job is None:
        abort(404)
    return jsonify(job.to_dict())


@app.post("/api/jobs/<jid>/cancel")
def api_job_cancel(jid: str):
    job = JOBS.get(jid)
    if job is None:
        abort(404)
    if job.status == "queued":
        job.status = "cancelled"
        _save_jobs()
    elif job.status == "running":
        job.cancel.set()
        job.note = "Cancelling…"
    return jsonify(job.to_dict())


@app.get("/media/<pid>/source")
def media_source(pid: str):
    """Only for projects made from a local file (streamed videos are previewed on their own site)."""
    project = _project_or_404(pid)
    path = project.get("source", {}).get("local_path")
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(Path(path).resolve(), conditional=True)         # conditional = HTTP Range, so seeking works


@app.get("/media/<pid>/clip/<name>")
def media_clip(pid: str, name: str):
    _project_or_404(pid)
    if not SAFE_CLIP.match(name):
        abort(404)
    path = P.project_dir(pid) / name
    if not path.is_file():
        abort(404)
    return send_file(path.resolve(), conditional=True)


@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(409)
@app.errorhandler(429)
def _json_error(e):
    return jsonify({"error": getattr(e, "description", str(e))}), e.code


def main() -> None:
    ap = argparse.ArgumentParser(description="Shorts Studio — local web UI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    args = ap.parse_args()

    os.chdir(ROOT)      # the default output folder ("output") is relative: always resolve it inside the repo
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    logsetup.setup()
    _load_jobs()
    try:
        memes.ensure_folder()
    except OSError:
        log.warning("could not create the memes folder")
    start_worker()
    start_upload_watcher()
    url = f"http://127.0.0.1:{args.port}"
    log.info("Shorts Studio %s starting on %s (output: %s)", config.version(), url, Path(config.LOCAL_OUTPUT_DIR).resolve())
    print(f"Shorts Studio {config.version()} running at {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
