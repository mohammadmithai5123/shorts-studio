"""Your YouTube channel: sign in once, then upload shorts to it (YouTube Data API v3 over plain ``requests``).

No Google SDK is installed on purpose (keeps the install small). What lives here:

* OAuth 2.0 for a "Desktop app" client, with PKCE and a loopback redirect to this very web app:
  :func:`auth_url` -> the person approves in the browser -> :func:`finish_auth` stores a long-lived refresh token.
* :func:`access_token` refreshes the short-lived token when needed. If Google says ``invalid_grant`` (revoked, expired,
  or the consent screen is still in "Testing", which expires tokens after 7 days) it raises :class:`ReconnectNeeded`.
* :func:`upload_video`: the resumable upload protocol. Big chunks, retry with back-off on 5xx / dropped connections, ask
  Google how much it already has and continue from there, start a new session if the old one expired.
* Bookkeeping: uploads per Pacific day (YouTube's quota resets at midnight Pacific) and a "parked until" time that is set
  when Google says the quota or the channel's upload limit is used up.

Files: ``<output>/state/youtube.json`` holds the refresh token and the counters (never sent to the browser, never in logs,
never in the diagnostics zip). The client id / secret live in ``.env`` like the other keys.

Everything that talks to Google goes through :data:`URLS`, so the tests point it at a fake local server.
"""
import base64
import hashlib
import json
import os
import random
import secrets
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests

from .. import config
from ..logsetup import get as _get_logger
from . import control
from .project import write_json_atomic

log = _get_logger("youtube")

SCOPE_UPLOAD = "https://www.googleapis.com/auth/youtube.upload"
SCOPE_READ = "https://www.googleapis.com/auth/youtube.readonly"          # only to show WHICH channel is connected
SCOPES = [SCOPE_UPLOAD, SCOPE_READ]

URLS: Dict[str, str] = {
    "auth": "https://accounts.google.com/o/oauth2/v2/auth",
    "token": "https://oauth2.googleapis.com/token",
    "revoke": "https://oauth2.googleapis.com/revoke",
    "channels": "https://www.googleapis.com/youtube/v3/channels",
    "upload": "https://www.googleapis.com/upload/youtube/v3/videos",
}
CHUNK = 8 * 1024 * 1024                    # the protocol wants a multiple of 256 KiB
MAX_FAILURES = 8                           # consecutive failures of one upload before giving up (back-off makes this ~3 minutes)
REDIRECT_PATH = "/"                          # the form Google documents for Desktop clients (http://127.0.0.1:<port>/)
CALLBACK_PATH = "/oauth/youtube/callback"   # an alias that answers the same way
_RETRYABLE = (500, 502, 503, 504)


# --------------------------------------------------------------------------------------------- #
# Errors: each one tells the caller what to DO (stop the queue, ask for a new sign-in, mark one clip failed...)
# --------------------------------------------------------------------------------------------- #
class YouTubeError(Exception):
    kind = "error"


class NotConnected(YouTubeError):
    kind = "not_connected"


class ReconnectNeeded(YouTubeError):
    kind = "reconnect"


class QuotaExceeded(YouTubeError):
    """Google refuses more uploads for now; ``until`` (epoch seconds) is when it is worth trying again."""
    kind = "quota"

    def __init__(self, message: str, until: float):
        super().__init__(message)
        self.until = until


class UploadRejected(YouTubeError):
    """This one video was refused for good (bad metadata, no channel, ...). Other videos may still go through."""
    kind = "rejected"


class NetworkError(YouTubeError):
    kind = "network"


def _now() -> float:
    return time.time()


def _sleep(seconds: float) -> None:
    """Sleep in short steps so Cancel is noticed (tests replace this with a no-op)."""
    end = time.monotonic() + seconds
    while True:
        control.check()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.5, left))


# --------------------------------------------------------------------------------------------- #
# Saved state
# --------------------------------------------------------------------------------------------- #
_STATE_LOCK = threading.RLock()
_REFRESH_LOCK = threading.Lock()


def state_file() -> Path:
    return Path(config.LOCAL_OUTPUT_DIR) / "state" / "youtube.json"


def _load() -> Dict:
    try:
        data = json.loads(state_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _mutate(fn: Callable[[Dict], None]) -> Dict:
    with _STATE_LOCK:
        data = _load()
        fn(data)
        write_json_atomic(state_file(), data)
        return data


def configured() -> bool:
    return bool(config.youtube_client_id() and config.youtube_client_secret())


def parse_client_json(text: str) -> Tuple[str, str]:
    """(client id, client secret) from the ``client_secret_*.json`` Google lets you download."""
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError("That is not a JSON file.") from e
    if isinstance(data, dict):
        data = data.get("installed") or data.get("web") or data
    cid = str((data or {}).get("client_id") or "").strip() if isinstance(data, dict) else ""
    secret = str((data or {}).get("client_secret") or "").strip() if isinstance(data, dict) else ""
    if not cid or not secret:
        raise ValueError("No client_id / client_secret found. Download the JSON of an OAuth client of type \"Desktop app\".")
    return cid, secret


# --------------------------------------------------------------------------------------------- #
# Pacific time (YouTube's daily quota resets at midnight Pacific; the tzdata package is not always installed on Windows)
# --------------------------------------------------------------------------------------------- #
def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def pacific_offset(ts: float) -> timedelta:
    """UTC-8, or UTC-7 from the 2nd Sunday of March (02:00 PST = 10:00 UTC) to the 1st Sunday of November (02:00 PDT = 09:00 UTC)."""
    dt = datetime.fromtimestamp(ts, timezone.utc)
    start = datetime(dt.year, 3, _nth_sunday(dt.year, 3, 2).day, 10, tzinfo=timezone.utc)
    end = datetime(dt.year, 11, _nth_sunday(dt.year, 11, 1).day, 9, tzinfo=timezone.utc)
    return timedelta(hours=-7 if start <= dt < end else -8)


def pacific_day(ts: float) -> str:
    return (datetime.fromtimestamp(ts, timezone.utc) + pacific_offset(ts)).strftime("%Y-%m-%d")


def next_pacific_midnight(ts: float) -> float:
    local = datetime.fromtimestamp(ts, timezone.utc) + pacific_offset(ts)
    tomorrow = datetime(local.year, local.month, local.day, tzinfo=timezone.utc) + timedelta(days=1)      # "wall clock" of the next midnight
    guess = tomorrow.timestamp() - pacific_offset(ts).total_seconds()
    return tomorrow.timestamp() - pacific_offset(guess).total_seconds()


# --------------------------------------------------------------------------------------------- #
# Status, counters, parking
# --------------------------------------------------------------------------------------------- #
def uploads_today(state: Optional[Dict] = None) -> int:
    state = state if state is not None else _load()
    return int((state.get("uploads") or {}).get(pacific_day(_now()), 0))


def uploads_left() -> int:
    return max(0, config.youtube_daily_uploads() - uploads_today())


def count_upload() -> None:
    day = pacific_day(_now())

    def bump(d: Dict) -> None:
        counts = d.setdefault("uploads", {})
        counts[day] = int(counts.get(day, 0)) + 1
        for old in sorted(counts)[:-7]:                      # a week of history is plenty
            counts.pop(old, None)

    _mutate(bump)


def parked() -> Optional[Dict]:
    """{"until": epoch, "reason": text} while Google says "not now", else None."""
    d = _load()
    until = float(d.get("parked_until") or 0)
    if until > _now():
        return {"until": until, "reason": d.get("parked_reason") or ""}
    return None


def park(until: float, reason: str) -> None:
    def put(d: Dict) -> None:
        d["parked_until"], d["parked_reason"] = until, reason[:300]

    _mutate(put)
    log.warning("uploads parked until %s: %s", time.strftime("%Y-%m-%d %H:%M", time.localtime(until)), reason[:200])


def unpark() -> None:
    def clear(d: Dict) -> None:
        d.pop("parked_until", None)
        d.pop("parked_reason", None)

    _mutate(clear)


def status() -> Dict:
    """Everything the page may know. Never contains a token or the client secret."""
    d = _load()
    now = _now()
    ends = float(d.get("refresh_expires_at") or 0)        # only present while the Google consent screen is still in "Testing"
    lapsed = bool(d.get("refresh_token")) and ends and ends < now
    ready = bool(d.get("refresh_token")) and not d.get("needs_reconnect") and not lapsed
    client_changed = bool(d.get("client_id")) and d.get("client_id") != config.youtube_client_id()
    pk = parked()
    return {
        "configured": configured(),
        "client_id": config.youtube_client_id(),
        "client_secret_set": bool(config.youtube_client_secret()),
        "connected": ready and not client_changed,
        "needs_reconnect": bool(d.get("refresh_token")) and (bool(d.get("needs_reconnect")) or client_changed or bool(lapsed)),
        "reconnect_reason": d.get("reconnect_reason") or ("The Google client was changed: connect again." if client_changed else "") or (
            "Google signs you out every 7 days while the consent screen is in Testing." if lapsed else ""),
        "paused": bool(d.get("paused")),
        "testing_mode": bool(ends) and not lapsed,
        "token_days_left": round((ends - now) / 86400, 1) if ends and not lapsed else None,
        "channel": d.get("channel"),
        "scope_ok": SCOPE_UPLOAD in (d.get("scope") or "") if d.get("refresh_token") else None,
        "uploads_today": uploads_today(d),
        "daily_limit": config.youtube_daily_uploads(),
        "uploads_left": uploads_left(),
        "parked_until": pk["until"] if pk else None,
        "parked_reason": pk["reason"] if pk else "",
        "privacy": config.youtube_privacy(),
        "category": config.youtube_category(),
        "auto_resume": config.youtube_auto_resume(),
        "notify_subscribers": config.youtube_notify_subscribers(),
        "connected_at": d.get("connected_at"),
    }


def _reconnect_text(reason: str) -> str:
    return (reason or "The YouTube sign-in is no longer valid.").rstrip(". ") + ". Open Settings → YouTube and press Reconnect. Your clips stay queued."


def is_paused() -> bool:
    return bool(_load().get("paused"))


def set_paused(paused: bool) -> None:
    """The big red switch: while paused nothing is uploaded (queued clips stay queued), whoever asks."""
    _mutate(lambda d: d.update(paused=bool(paused)))
    log.info("uploads %s", "paused" if paused else "resumed")


def check_client() -> Dict:
    """Does Google accept this client id + secret? A deliberately bogus code is sent: ``invalid_grant`` means the client is
    fine, ``invalid_client`` means it is not. Catches a mistyped secret or a client of the wrong type before the sign-in."""
    if not configured():
        return {"ok": False, "error": "Add the client id and secret first."}
    try:
        r = requests.post(URLS["token"], data={
            "grant_type": "authorization_code", "code": "shorts-studio-check", "redirect_uri": "http://127.0.0.1/", "code_verifier": "x" * 64,
            "client_id": config.youtube_client_id(), "client_secret": config.youtube_client_secret()}, timeout=15)
    except requests.RequestException as e:
        return {"ok": False, "network": True, "error": f"Could not reach Google ({e.__class__.__name__}). Check the internet connection."}
    try:
        err = str((r.json() or {}).get("error") or "")
    except ValueError:
        err = ""
    if err in ("invalid_client", "unauthorized_client"):
        return {"ok": False, "error": "Google does not recognise this client id / secret. Copy them again from the Credentials page "
                                      "(the client type must be \"Desktop app\")."}
    if err == "redirect_uri_mismatch":
        return {"ok": False, "error": "This client is not of type \"Desktop app\". Create a new OAuth client with that type."}
    if r.status_code == 400:
        return {"ok": True}
    return {"ok": False, "error": f"Unexpected answer from Google (HTTP {r.status_code} {err})."}


def require_ready() -> None:
    st = status()
    if not st["configured"]:
        raise NotConnected("YouTube is not set up yet. Open Settings → YouTube and add your Google client.")
    if st["needs_reconnect"]:
        raise ReconnectNeeded(_reconnect_text(st["reconnect_reason"]))
    if not st["connected"]:
        raise NotConnected("Your YouTube channel is not connected. Open Settings → YouTube and press Connect.")


# --------------------------------------------------------------------------------------------- #
# Sign-in (OAuth 2.0 authorization code + PKCE, loopback redirect)
# --------------------------------------------------------------------------------------------- #
_PENDING: Dict[str, Dict] = {}
_PENDING_TTL = 600.0


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def auth_url(redirect_uri: str) -> str:
    if not configured():
        raise NotConnected("Add your Google client id and secret first.")
    now = _now()
    for k in [k for k, v in _PENDING.items() if now - v["at"] > _PENDING_TTL]:
        _PENDING.pop(k, None)
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(24)
    _PENDING[state] = {"verifier": verifier, "redirect": redirect_uri, "at": now}
    return URLS["auth"] + "?" + urlencode({
        "client_id": config.youtube_client_id(), "redirect_uri": redirect_uri, "response_type": "code",
        "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent", "state": state,
        "code_challenge": _b64url(hashlib.sha256(verifier.encode("ascii")).digest()), "code_challenge_method": "S256",
    })


def _token_request(data: Dict) -> Dict:
    try:
        r = requests.post(URLS["token"], data=data, timeout=20)
    except requests.RequestException as e:
        raise NetworkError(f"Could not reach Google ({e.__class__.__name__}). Check the internet connection.") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        err = str(body.get("error") or r.status_code)
        desc = str(body.get("error_description") or "")
        if err == "invalid_grant":
            raise ReconnectNeeded("Google no longer accepts the saved sign-in (" + (desc or "revoked or expired") + ").")
        if err in ("invalid_client", "unauthorized_client"):
            raise YouTubeError("Google does not accept this client id / secret. Check them in Settings → YouTube "
                               "(the client type must be \"Desktop app\").")
        raise YouTubeError(f"Google refused the sign-in: {err} {desc}".strip())
    return body


def finish_auth(state: str, code: Optional[str], error: Optional[str] = None) -> Dict:
    """The browser came back from Google. Exchanges the code for tokens and saves them. Returns the channel (or {})."""
    pending = _PENDING.pop(state or "", None)
    if pending is None or _now() - pending["at"] > _PENDING_TTL:
        raise YouTubeError("This sign-in link is not valid any more. Go back to Settings → YouTube and press Connect again.")
    if error == "access_denied":
        raise YouTubeError("Google did not give access. If you saw \"Access blocked\" or \"has not completed the verification\", add your own "
                           "Google account under Audience → Test users (or publish the app) in the Google Cloud console, then press Connect again.")
    if error or not code:
        raise YouTubeError("Google did not give access" + (f" ({error})" if error else "") + ". Press Connect and approve it.")
    body = _token_request({
        "grant_type": "authorization_code", "code": code, "redirect_uri": pending["redirect"], "code_verifier": pending["verifier"],
        "client_id": config.youtube_client_id(), "client_secret": config.youtube_client_secret(),
    })
    scope = str(body.get("scope") or "")
    if SCOPE_UPLOAD not in scope:
        raise YouTubeError("The permission to upload videos was not ticked. Press Connect again and keep every box ticked.")
    if not body.get("refresh_token"):
        raise YouTubeError("Google did not give a long-lasting sign-in. Remove \"Shorts Studio\" at "
                           "myaccount.google.com/permissions and press Connect again.")

    def store(d: Dict) -> None:
        d.update(refresh_token=body["refresh_token"], access_token=body.get("access_token"), scope=scope,
                 expires_at=_now() + float(body.get("expires_in") or 3600), client_id=config.youtube_client_id(),
                 connected_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        for k in ("needs_reconnect", "reconnect_reason", "channel", "refresh_expires_at"):
            d.pop(k, None)
        if body.get("refresh_token_expires_in"):              # Google adds this only for consent screens in "Testing"
            d["refresh_expires_at"] = _now() + float(body["refresh_token_expires_in"])

    _mutate(store)
    channel: Dict = {}
    if SCOPE_READ in scope:
        channel = fetch_channel() or {}
        if not channel:
            disconnect(revoke=True)
            raise YouTubeError("This Google account has no YouTube channel. Create one at youtube.com first "
                               "(and choose the right account when Google asks).")
        _mutate(lambda d: d.update(channel=channel))
    log.info("YouTube connected%s", f" as {channel.get('title')!r}" if channel else "")
    return channel


def disconnect(revoke: bool = True) -> None:
    """Forget the sign-in (and tell Google to forget it too). The per-day counters stay: the quota belongs to the project."""
    d = _load()
    token = d.get("refresh_token")
    if revoke and token:
        try:
            requests.post(URLS["revoke"], data={"token": token}, timeout=10)
        except requests.RequestException:
            pass                                             # best effort: it can always be removed at myaccount.google.com

    def wipe(state: Dict) -> None:
        keep = {k: state[k] for k in ("uploads", "parked_until", "parked_reason", "paused") if k in state}
        state.clear()
        state.update(keep)

    _mutate(wipe)


def access_token(force: bool = False) -> str:
    d = _load()
    if not d.get("refresh_token"):
        raise NotConnected("Your YouTube channel is not connected. Open Settings → YouTube and press Connect.")
    if d.get("needs_reconnect"):
        raise ReconnectNeeded(_reconnect_text(d.get("reconnect_reason")))
    if d.get("client_id") and d["client_id"] != config.youtube_client_id():
        raise ReconnectNeeded(_reconnect_text("The Google client was changed"))
    if not force and d.get("access_token") and float(d.get("expires_at") or 0) - 60 > _now():
        return d["access_token"]
    with _REFRESH_LOCK:
        d = _load()                                          # another thread may have refreshed while we waited
        if not force and d.get("access_token") and float(d.get("expires_at") or 0) - 60 > _now():
            return d["access_token"]
        try:
            body = _token_request({"grant_type": "refresh_token", "refresh_token": d["refresh_token"],
                                   "client_id": config.youtube_client_id(), "client_secret": config.youtube_client_secret()})
        except ReconnectNeeded as e:
            reason = str(e)
            _mutate(lambda s: s.update(needs_reconnect=True, reconnect_reason=reason))
            raise ReconnectNeeded(_reconnect_text(reason)) from e
        _mutate(lambda s: s.update(access_token=body["access_token"], expires_at=_now() + float(body.get("expires_in") or 3600)))
        return body["access_token"]


def _bearer() -> str:
    return "Bearer " + access_token()


def _authed(method: str, url: str, **kw) -> requests.Response:
    """A request with the bearer token: refreshes once on 401, retries 5xx / dropped connections with back-off."""
    headers = dict(kw.pop("headers", {}) or {})
    kw.setdefault("timeout", 30)
    refreshed = False
    last_error: Optional[Exception] = None
    r: Optional[requests.Response] = None
    for attempt in range(5):
        control.check()
        try:
            r = requests.request(method, url, headers={**headers, "Authorization": _bearer()}, **kw)
        except requests.RequestException as e:
            last_error, r = e, None
            _sleep(min(2 ** attempt, 20) + random.random())
            continue
        if r.status_code == 401 and not refreshed:
            refreshed = True
            access_token(force=True)                 # raises ReconnectNeeded when Google no longer accepts the sign-in
            continue
        if r.status_code in _RETRYABLE:
            _sleep(min(2 ** attempt, 20) + random.random())
            continue
        return r
    if r is None:
        raise NetworkError(f"Could not reach YouTube ({last_error.__class__.__name__}). Check the internet connection.")
    return r                                                  # still a 5xx: the caller classifies it


def fetch_channel() -> Optional[Dict]:
    r = _authed("GET", URLS["channels"], params={"part": "snippet", "mine": "true"})
    if r.status_code != 200:
        raise _classify(r)
    items = r.json().get("items") or []
    if not items:
        return None
    sn = items[0].get("snippet") or {}
    thumbs = sn.get("thumbnails") or {}
    return {"id": items[0].get("id"), "title": sn.get("title") or "", "thumbnail": (thumbs.get("default") or {}).get("url")}


# --------------------------------------------------------------------------------------------- #
# Turning Google's error answers into the right Python exception
# --------------------------------------------------------------------------------------------- #
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
CHANNEL_LIMIT_WAIT = 3 * 3600                   # the channel's limit covers a rolling 24 hours: it may open again any time
_BUSY_REASONS = {"rateLimitExceeded", "userRateLimitExceeded"}


def _error_info(r: requests.Response) -> Tuple[str, str]:
    try:
        data = r.json()
    except ValueError:
        return "", (r.text or "")[:200]
    err = (data.get("error") if isinstance(data, dict) else None) or {}
    if not isinstance(err, dict):
        return str(err), ""
    reasons = [str(e.get("reason")) for e in err.get("errors") or [] if isinstance(e, dict) and e.get("reason")]
    return (reasons[0] if reasons else str(err.get("status") or "")), str(err.get("message") or "")


def _classify(r: requests.Response) -> YouTubeError:
    reason, message = _error_info(r)
    now = _now()
    if reason in _QUOTA_REASONS:
        return QuotaExceeded("YouTube's daily upload quota of this Google project is used up. Uploads continue after midnight Pacific time.",
                             next_pacific_midnight(now))
    if reason == "uploadLimitExceeded":
        return QuotaExceeded("YouTube refused: this channel reached the number of uploads YouTube allows in 24 hours. The app tries again every "
                             "3 hours.", now + CHANNEL_LIMIT_WAIT)
    if reason in _BUSY_REASONS:
        return QuotaExceeded("YouTube asks to slow down for a moment.", now + 15 * 60)
    if r.status_code >= 500:
        return NetworkError(f"YouTube is having problems right now (HTTP {r.status_code}). It stays queued and is tried again later.")
    if reason in ("youtubeSignupRequired", "channelNotFound"):             # arrives as a 401, so it must be checked before it
        return UploadRejected("This Google account has no YouTube channel. Create one at youtube.com first.")
    if r.status_code == 401 or reason in ("authError", "insufficientPermissions"):
        return ReconnectNeeded(_reconnect_text("YouTube does not accept the saved sign-in or it lacks the upload permission."))
    if reason in ("accountClosed", "accountSuspended", "accountDisabled"):
        return UploadRejected("YouTube says this channel is closed or suspended: " + message)
    return UploadRejected(f"YouTube refused it ({r.status_code} {reason}): {message}".strip()[:400])


# --------------------------------------------------------------------------------------------- #
# What goes on YouTube next to the video
# --------------------------------------------------------------------------------------------- #
def _clean_text(text: str) -> str:
    return str(text or "").replace("<", "").replace(">", "").strip()          # the API rejects angle brackets in titles / descriptions


def build_metadata(title: str, description: str = "", hashtags: Optional[list] = None, *, privacy: str = "private",
                   publish_at: Optional[str] = None, category: str = "22") -> Dict:
    """The video resource for ``videos.insert``. A scheduled video must be private until its ``publishAt`` time."""
    title = _clean_text(title) or "Short"
    if len(title) > 100:
        title = title[:99].rsplit(" ", 1)[0].rstrip(" ,.-") or title[:99]
    tags = []
    for t in hashtags or []:
        t = "#" + str(t).lstrip("#").strip()
        if len(t) > 1 and " " not in t and t.lower() not in [x.lower() for x in tags]:
            tags.append(t)
    if "#shorts" not in [t.lower() for t in tags]:
        tags.append("#Shorts")
    body = _clean_text(description)
    description = (body + "\n\n" if body else "") + " ".join(tags[:15])
    keywords, used = [], 0
    for t in tags:
        word = t.lstrip("#")
        if used + len(word) + 1 > 400:
            break
        keywords.append(word)
        used += len(word) + 1
    status = {"privacyStatus": "private" if publish_at else privacy, "selfDeclaredMadeForKids": False, "embeddable": True}
    if publish_at:
        status["publishAt"] = publish_at
    return {"snippet": {"title": title, "description": description[:4900], "tags": keywords, "categoryId": str(category)}, "status": status}


def iso_utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


# --------------------------------------------------------------------------------------------- #
# The resumable upload
# --------------------------------------------------------------------------------------------- #
def _start_session(meta: Dict, size: int, mime: str) -> str:
    r = _authed("POST", URLS["upload"], params={"uploadType": "resumable", "part": "snippet,status",
                                                  "notifySubscribers": "true" if config.youtube_notify_subscribers() else "false"},
                headers={"Content-Type": "application/json; charset=UTF-8", "X-Upload-Content-Length": str(size),
                         "X-Upload-Content-Type": mime},
                data=json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    if r.status_code != 200 or not r.headers.get("Location"):
        raise _classify(r) if r.status_code != 200 else UploadRejected("YouTube did not return an upload address.")
    return r.headers["Location"]


def _stored_offset(session: str, size: int, mime: str) -> Tuple[int, Optional[Dict]]:
    """Ask Google how much of the file it has. (offset, finished video resource or None)."""
    r = requests.put(session, data=b"", headers={"Content-Length": "0", "Content-Range": f"bytes */{size}", "Content-Type": mime,
                                                         "Authorization": _bearer()},
                     timeout=(10, 60))
    if r.status_code in (200, 201):
        return size, r.json()
    if r.status_code == 308:
        return _range_end(r), None
    if r.status_code in (404, 410):
        raise LookupError("session expired")
    raise _classify(r) if r.status_code < 500 else requests.ConnectionError(f"HTTP {r.status_code}")


def _range_end(r: requests.Response) -> int:
    rng = r.headers.get("Range") or ""
    try:
        return int(rng.rsplit("-", 1)[1]) + 1
    except (IndexError, ValueError):
        return 0                                              # no Range header: nothing stored yet


def upload_video(path: str, meta: Dict, progress: Optional[Callable[[int, int], None]] = None, chunk_size: int = CHUNK,
                 mime: str = "video/mp4") -> Dict:
    """Upload one file; returns the video resource (``id``, ``status``...). Raises one of the YouTubeError subclasses."""
    size = os.path.getsize(path)
    if size <= 0:
        raise UploadRejected("The clip file is empty.")
    chunk_size = max(256 * 1024, chunk_size - chunk_size % (256 * 1024))
    session = _start_session(meta, size, mime)
    offset, failures, restarts, stalls = 0, 0, 0, 0
    with open(path, "rb") as f:
        while True:
            control.check()
            f.seek(offset)
            data = f.read(chunk_size)
            last = offset + len(data) - 1
            try:
                r = requests.put(session, data=data, headers={"Content-Length": str(len(data)), "Content-Type": mime, "Authorization": _bearer(),
                                                              "Content-Range": f"bytes {offset}-{last}/{size}"}, timeout=(10, 180))
            except requests.RequestException:
                r = None
            if r is not None and r.status_code in (200, 201):
                if progress:
                    progress(size, size)
                return r.json()
            if r is not None and r.status_code == 308:
                failures = 0
                stored = _range_end(r)
                stalls = stalls + 1 if stored <= offset else 0           # Google answers "continue" but keeps nothing: do not loop forever
                if stalls > 5:
                    raise NetworkError("YouTube keeps asking for the same part of the file. It stays queued and is tried again later.")
                offset = stored
                if progress:
                    progress(offset, size)
                continue
            if r is not None and r.status_code in (404, 410):          # the session expired: start over, once or twice
                restarts += 1
                if restarts > 2:
                    raise NetworkError("YouTube keeps forgetting the upload session. Try again later.")
                session, offset = _start_session(meta, size, mime), 0
                continue
            if r is not None and r.status_code == 401:
                access_token(force=True)                               # raises ReconnectNeeded when the sign-in is gone
                failures += 1
            elif r is None or r.status_code in _RETRYABLE:
                failures += 1
            else:
                raise _classify(r)
            if failures > MAX_FAILURES:
                raise NetworkError("The upload keeps failing (the connection or YouTube is unstable). It stays queued and is tried again later.")
            _sleep(min(2 ** failures, 30) + random.random())
            try:                                                       # how much did Google keep? continue from there
                offset, done = _stored_offset(session, size, mime)
                if done is not None:
                    return done
                if progress:
                    progress(offset, size)
            except LookupError:
                restarts += 1
                if restarts > 2:
                    raise NetworkError("YouTube keeps forgetting the upload session. Try again later.")
                session, offset = _start_session(meta, size, mime), 0
            except requests.RequestException:
                pass                                                   # could not even ask: the loop retries after the next back-off
