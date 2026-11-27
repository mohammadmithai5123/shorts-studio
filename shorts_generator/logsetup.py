"""File logging for the web app and the CLI: ``<output>/logs/studio.log`` (1 MB x 4 rotating files).

The console and the live job log stay short and friendly; this file keeps the full story, including
tracebacks, so "it failed yesterday" can still be diagnosed. Idempotent: calling it twice adds one handler.
"""
import logging
import logging.handlers
import os
import re
from pathlib import Path

_NAME = "studio"

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),                 # Google API keys
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                  # OpenAI-style keys
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),          # user:password@ inside any URL (proxies)
    re.compile(r"ya29\.[0-9A-Za-z_\-.]{20,}"),               # Google OAuth access tokens
    re.compile(r"1//[0-9A-Za-z_\-]{20,}"),                   # Google OAuth refresh tokens
    re.compile(r"GOCSPX-[0-9A-Za-z_\-]{10,}"),               # Google OAuth client secrets
    re.compile(r"(?<=[?&]code=)[^&\s\"']+"),                 # the one-time code in an OAuth redirect
]


def redact(text: str) -> str:
    """Hide anything that looks like a key or a password, plus the exact values of the saved keys/proxy.

    Applied to the log file and to everything put in a diagnostics zip: a bug report must never leak a secret,
    even one that an exception message happened to carry.
    """
    if not text:
        return text
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "MUAPI_API_KEY", "YTDLP_PROXY", "YOUTUBE_CLIENT_SECRET"):
        value = os.environ.get(name, "")
        if len(value) >= 8:
            text = text.replace(value, "***")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("***", text)
    return text


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def log_dir() -> Path:
    from .config import LOCAL_OUTPUT_DIR

    return Path(LOCAL_OUTPUT_DIR) / "logs"


def setup() -> logging.Logger:
    logger = logging.getLogger(_NAME)
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(d / "studio.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())      # an unwritable folder must never stop the app
    return logger


def get(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{_NAME}.{name}" if name else _NAME)
