"""Settings. Everything comes from environment variables / the ``.env`` file in the repo root.

Values that people change while the app is running (LLM provider, model, API key) are read through
the small accessor functions below on every use, so the web UI's Settings panel takes effect at once,
without a restart. The module-level constants are the import-time snapshot kept for the MuAPI (api
mode) code and for settings that never change at runtime.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)

# Every Short stays under a minute (the person's wish; YouTube also blocks a longer one everywhere when it has a
# Content ID claim). A little under 60 s so that rounding to frames can never cross it.
SHORT_SECONDS = 59.0


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def version() -> str:
    from ._version import __version__

    return __version__


# --- MuAPI (api mode) ---------------------------------------------------------------------------
MUAPI_API_KEY = env("MUAPI_API_KEY")
MUAPI_BASE_URL = os.getenv("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")
POLL_INTERVAL_SECONDS = float(os.getenv("MUAPI_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("MUAPI_POLL_TIMEOUT", "600"))

# --- local mode: fixed at import time ------------------------------------------------------------
LOCAL_WHISPER_DEVICE = env("LOCAL_WHISPER_DEVICE", "auto")      # auto / cpu / cuda
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "output")

# Whisper is only the fallback for videos without captions. Beam 5 is the original behaviour; beam 1
# (greedy) was only ~25% faster in a noisy benchmark and can change the transcript, so it is opt-in.
LOCAL_WHISPER_BEAM_SIZE = env("LOCAL_WHISPER_BEAM_SIZE", "5").lower()
LOCAL_WHISPER_THREADS = int(os.getenv("LOCAL_WHISPER_THREADS", "0") or 0)   # 0 = CTranslate2 default (4); keep <= physical cores

# Burned-in captions: font family override (default: Arial, Nirmala UI for Devanagari).
LOCAL_CAPTION_FONT = env("LOCAL_CAPTION_FONT")

# VAD (Voice Activity Detection) settings for faster-whisper. Disabled by default: too aggressive on
# mixed speech/music content.
LOCAL_WHISPER_VAD_FILTER = env("LOCAL_WHISPER_VAD_FILTER", "false").lower() == "true"
_vad_params_env = os.getenv("LOCAL_WHISPER_VAD_PARAMETERS", "")
if _vad_params_env:
    LOCAL_WHISPER_VAD_PARAMETERS = json.loads(_vad_params_env)
else:
    LOCAL_WHISPER_VAD_PARAMETERS = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }


# --- local mode: read live ----------------------------------------------------------------------
def gemini_key() -> str:
    return env("GEMINI_API_KEY")


def openai_key() -> str:
    return env("OPENAI_API_KEY")


def llm_provider() -> str:
    """LLM_PROVIDER if set, otherwise whichever provider has a key."""
    v = env("LLM_PROVIDER").lower()
    if v in ("gemini", "openai"):
        return v
    if gemini_key() or not openai_key():
        return "gemini"
    return "openai"


def gemini_model() -> str:
    """'auto' (the default) picks the best available Gemini Flash model and falls back to the others."""
    return env("GEMINI_MODEL", "auto") or "auto"


def openai_model() -> str:
    return env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"


def llm_cache_enabled() -> bool:
    return env("LLM_CACHE", "1").lower() not in ("0", "false", "no", "off")


def whisper_model() -> str:
    return env("LOCAL_WHISPER_MODEL", "base") or "base"


def caption_script() -> str:
    """Letters of the burned-in text: "roman" (Hinglish / Roman Urdu, the default) or "original" (as spoken: Hindi / Urdu script)."""
    return "original" if env("LOCAL_CAPTION_SCRIPT", "roman").lower() == "original" else "roman"


def delete_posted_clips() -> bool:
    """A clip that is on YouTube is removed from this computer (its history stays)."""
    return env("LOCAL_DELETE_UPLOADED", "1").lower() not in ("0", "false", "no", "off")


def cache_days() -> int:
    """Saved captions, loudness and AI answers older than this are removed (they can be fetched again)."""
    try:
        return max(1, min(90, int(env("LOCAL_CACHE_DAYS", "3") or 3)))
    except ValueError:
        return 3


def facecam_background() -> str:
    """Colour behind a streamer cut out over a game (see local/cutout.py)."""
    v = env("LOCAL_FACECAM_BG", "white").lower()
    return v if v in ("white", "green", "black", "blue", "purple", "pink") else "white"


# --- YouTube channel (automatic uploads) ------------------------------------------------------------
def youtube_client_id() -> str:
    return env("YOUTUBE_CLIENT_ID")


def youtube_client_secret() -> str:
    return env("YOUTUBE_CLIENT_SECRET")


def youtube_privacy() -> str:
    """Default visibility of uploaded shorts. 'private' on purpose: the first uploads should be looked at before they go public."""
    v = env("YOUTUBE_PRIVACY", "private").lower()
    return v if v in ("private", "unlisted", "public") else "private"


def youtube_category() -> str:
    v = env("YOUTUBE_CATEGORY", "22")
    return v if v.isdigit() else "22"          # 22 = People & Blogs


def youtube_daily_uploads() -> int:
    """Our own cap per (Pacific) day, so one busy evening cannot burn the whole API quota or look like spam."""
    try:
        return max(1, min(50, int(env("YOUTUBE_DAILY_UPLOADS", "10"))))
    except ValueError:
        return 10


def youtube_auto_resume() -> bool:
    """Uploads that had to wait (daily cap, quota, no internet) continue by themselves while the app is open."""
    return env("YOUTUBE_AUTO_RESUME", "1").lower() not in ("0", "false", "no", "off")


def youtube_notify_subscribers() -> bool:
    return env("YOUTUBE_NOTIFY_SUBSCRIBERS", "0").lower() in ("1", "true", "yes", "on")


# --- settings written by the UI -------------------------------------------------------------------
_VALIDATORS = {
    "LLM_PROVIDER": re.compile(r"^(gemini|openai)$"),
    "GEMINI_MODEL": re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$"),
    "OPENAI_MODEL": re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$"),
    "GEMINI_API_KEY": re.compile(r"^[A-Za-z0-9_.-]{8,200}$"),
    "OPENAI_API_KEY": re.compile(r"^[A-Za-z0-9_.-]{8,300}$"),
    "YTDLP_COOKIES_BROWSER": re.compile(r"^(|chrome|chromium|edge|firefox|brave|opera|vivaldi|safari)$"),
    "YTDLP_COOKIES_FILE": re.compile(r'^[^\r\n"<>|*?]{0,400}$'),
    "YTDLP_PROXY": re.compile(r"^(|(https?|socks4a?|socks5h?)://[^\s\"'<>]{3,300})$"),
    "LOCAL_WHISPER_MODEL": re.compile(r"^(tiny|base|small|medium|large-v3|distil-large-v3)(\.en)?$"),
    "LOCAL_ENCODER": re.compile(r"^(auto|cpu|qsv|nvenc|amf)$"),
    "LOCAL_CAPTION_SCRIPT": re.compile(r"^(roman|original)$"),
    "LOCAL_DELETE_UPLOADED": re.compile(r"^[01]$"),
    "LOCAL_CACHE_DAYS": re.compile(r"^([1-9]|[1-8][0-9]|90)$"),
    "LOCAL_FACECAM_BG": re.compile(r"^(white|green|black|blue|purple|pink)$"),
    "YOUTUBE_CLIENT_ID": re.compile(r"^(|[A-Za-z0-9._-]{10,200})$"),
    "YOUTUBE_CLIENT_SECRET": re.compile(r"^[A-Za-z0-9_.-]{8,200}$"),
    "YOUTUBE_PRIVACY": re.compile(r"^(private|unlisted|public)$"),
    "YOUTUBE_CATEGORY": re.compile(r"^[0-9]{1,2}$"),
    "YOUTUBE_DAILY_UPLOADS": re.compile(r"^([1-9]|[1-4][0-9]|50)$"),
    "YOUTUBE_AUTO_RESUME": re.compile(r"^[01]$"),
    "YOUTUBE_NOTIFY_SUBSCRIBERS": re.compile(r"^[01]$"),
}
SECRET_SETTINGS = {"GEMINI_API_KEY", "OPENAI_API_KEY", "YTDLP_PROXY", "YOUTUBE_CLIENT_SECRET"}      # may hold credentials: write-only in the UI


def save_settings(values: Dict[str, str]) -> None:
    """Validate, apply to this process, and persist to .env. Raises ValueError on a bad value."""
    from dotenv import set_key

    clean: Dict[str, str] = {}
    for name, value in values.items():
        rx = _VALIDATORS.get(name)
        if rx is None:
            raise ValueError(f"Setting {name!r} cannot be changed here.")
        value = str(value).strip()
        if not rx.match(value):
            raise ValueError(f"Invalid value for {name}.")
        if name == "YTDLP_COOKIES_FILE" and value and not Path(value).is_file():
            raise ValueError("That cookies file does not exist.")
        clean[name] = value
    if not ENV_FILE.exists():
        ENV_FILE.write_text("", encoding="utf-8")
    for name, value in clean.items():
        set_key(str(ENV_FILE), name, value, quote_mode="never")
        os.environ[name] = value


# --- legacy helpers (api mode / older callers) ---------------------------------------------------
def require_api_key() -> str:
    if not MUAPI_API_KEY:
        raise RuntimeError(
            "MUAPI_API_KEY is not set. Add it to your .env file or export it as an env var."
        )
    return MUAPI_API_KEY


def require_openai_key() -> str:
    key = openai_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it in Settings (or to your .env), or pick the Gemini provider."
        )
    return key


def require_gemini_key() -> str:
    key = gemini_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it in Settings (or to your .env) — a free key is at "
            "https://aistudio.google.com/apikey"
        )
    return key
