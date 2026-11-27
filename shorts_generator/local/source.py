"""Where a video comes from, and how clips are cut out of it WITHOUT downloading the whole video.

* A link (YouTube, or anything yt-dlp understands): only metadata is fetched up front (a few
  seconds). Captions are a tiny text file, the loudness signal is streamed straight through ffmpeg
  (nothing is saved), and a clip is fetched as just its own time range, in H.264 so it decodes fast.
* A local file: used in place.

Nothing here knows about LLMs, layouts or the UI: it answers "what is this video" and "give me the
pixels of seconds A..B".
"""
import copy
import glob
import hashlib
import re
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

from ..config import env
from . import control


class SourceError(RuntimeError):
    """A problem the user can act on (private video, bot check, bad path...). The text is user-facing."""


def _is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text or ""))


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)[:80] or "video"


def ydl_opts(extra: Optional[Dict] = None) -> Dict:
    """Shared yt-dlp options. Videos that need a login: YTDLP_COOKIES_BROWSER=chrome|edge|firefox or YTDLP_COOKIES_FILE=cookies.txt."""
    opts: Dict = {
        "quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
        "socket_timeout": 30, "retries": 8, "fragment_retries": 8,
    }
    cookies_file, browser, proxy = env("YTDLP_COOKIES_FILE"), env("YTDLP_COOKIES_BROWSER"), env("YTDLP_PROXY")
    if cookies_file:
        opts["cookiefile"] = cookies_file                  # an exported cookies.txt works when reading the browser does not
    elif browser:
        opts["cookiesfrombrowser"] = (browser,)
    if proxy:
        opts["proxy"] = proxy
    if extra:
        opts.update(extra)
    return opts


def friendly_error(exc: BaseException) -> str:
    """Turn a yt-dlp failure into something a person can act on."""
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).replace("ERROR: ", "").strip()
    low = msg.lower()
    if "not a bot" in low or "sign in to confirm" in low:
        return ("YouTube wants a sign-in before it serves this video (bot check). Put YTDLP_COOKIES_BROWSER=chrome "
                "(or edge / firefox) in your .env, make sure you are signed in to YouTube in that browser, and try again.")
    if "private video" in low:
        return "This video is private."
    if "members-only" in low or "join this channel" in low:
        return "This video is for channel members only (set YTDLP_COOKIES_BROWSER if you are a member)."
    if "confirm your age" in low or "age-restricted" in low or "inappropriate for some users" in low:
        return "This video is age-restricted. Set YTDLP_COOKIES_BROWSER in your .env (signed in to YouTube) to open it."
    if "video unavailable" in low or "this video is not available" in low or "has been removed" in low:
        return "This video is unavailable (removed, region-locked or private)."
    if "unsupported url" in low:
        return "That link is not a video page this tool can read."
    if "is live" in low or "live event" in low or "premiere" in low:
        return "Live streams and premieres cannot be analysed until they have finished."
    if "http error 429" in low or "too many requests" in low:
        return "YouTube is rate-limiting this connection (HTTP 429). Wait a few minutes and try again."
    if "unable to download" in low or "urlopen error" in low or "timed out" in low or "getaddrinfo" in low:
        return "Could not reach the video site. Check your internet connection and try again."
    return msg[:300] or "yt-dlp failed."


def _ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def probe_codec(path: str) -> str:
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        return ""


def _local_path(target: str) -> Optional[Path]:
    t = target.strip().strip('"').strip("'")
    if t.lower().startswith("file://"):
        parsed = urlparse(t)
        raw = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", raw):          # file:///C:/x -> C:/x
            raw = raw[1:]
        t = raw
    p = Path(t).expanduser()
    return p if p.is_file() else None


@dataclass
class Source:
    kind: str                           # "youtube" | "web" | "file"
    id: str
    url: str                            # the link, or the file path
    title: str
    duration: float
    language: Optional[str] = None
    embeddable: bool = True
    heatmap: Optional[List[Dict]] = None        # YouTube "most replayed": [{start_time, end_time, value}]
    chapters: Optional[List[Dict]] = None
    local_path: Optional[str] = None
    channel: str = ""                           # whose video it is (named in the titles)
    _info: Dict = field(default_factory=dict, repr=False)

    # ---- description -------------------------------------------------------------------------
    def meta(self) -> Dict:
        """JSON-safe description stored in project.json."""
        return {
            "kind": self.kind, "id": self.id, "url": self.url, "title": self.title, "duration": self.duration,
            "language": self.language, "embeddable": self.embeddable, "heatmap": self.heatmap,
            "chapters": self.chapters, "local_path": self.local_path, "channel": self.channel,
        }

    @property
    def is_stream(self) -> bool:
        return self.kind != "file"

    # ---- captions ----------------------------------------------------------------------------
    def caption_track(self, language: Optional[str] = None) -> Optional[Dict]:
        """Best caption track: {kind: manual|auto, key, ext, url}, or None.

        Hand-made subtitles beat automatic ones; of the automatic ones the original-language track
        (YouTube names it '<lang>-orig') beats machine translations.
        """
        info = self._info or {}
        manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
        auto = info.get("automatic_captions") or {}
        want = (language or "").strip().lower() or None
        order: List[Tuple[str, str, List[Dict]]] = []

        def add(pool: Dict, kind: str, key: str) -> None:
            if key in pool and not any(o[1] == key and o[0] == kind for o in order):
                order.append((kind, key, pool[key]))

        if want:
            add(manual, "manual", want)
            for k in manual:
                if k.lower().startswith(want + "-"):
                    add(manual, "manual", k)
            add(auto, "auto", f"{want}-orig")
            add(auto, "auto", want)
        else:
            lang = (info.get("language") or "").lower()
            if lang:
                # the video's own language first (hand-made, then automatic), and only then any other track:
                # an English translation of a Hindi video must not win, its timing does not match the speech
                add(manual, "manual", lang)
                for k in manual:
                    if k.lower().startswith(lang + "-"):
                        add(manual, "manual", k)
                for k in auto:
                    if k.endswith("-orig"):
                        add(auto, "auto", k)
                add(auto, "auto", lang)
                for k in manual:
                    add(manual, "manual", k)
            else:
                for k in manual:
                    add(manual, "manual", k)
                for k in auto:
                    if k.endswith("-orig"):
                        add(auto, "auto", k)

        for kind, key, formats in order:
            for ext in ("json3", "vtt", "srv1"):
                hit = next((f for f in formats if f.get("ext") == ext and f.get("url")), None)
                if hit:
                    return {"kind": kind, "key": key, "ext": ext, "url": hit["url"]}
        return None

    # ---- audio (for the loudness signal / Whisper) --------------------------------------------
    def audio_stream(self) -> Optional[Tuple[str, Dict[str, str]]]:
        """(url_or_path, http_headers) of the smallest audio-only stream, readable by ffmpeg."""
        if self.kind == "file":
            return (self.local_path, {}) if self.local_path else None
        cands = [
            f for f in (self._info or {}).get("formats", [])
            if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none") and f.get("url")
            and str(f.get("protocol", "")).startswith("http") and "drc" not in str(f.get("format_id", ""))
        ]
        if not cands:
            return None
        usable = [f for f in cands if (f.get("abr") or 0) >= 40] or cands
        best = min(usable, key=lambda f: f.get("abr") or 999)
        return best["url"], dict(best.get("http_headers") or {})

    def fetch_audio_file(self, out_dir: Path) -> Path:
        """Audio-only copy on disk (tens of MB, never the video) — only for the Whisper fallback."""
        if self.kind == "file" and self.local_path:
            return Path(self.local_path)
        import yt_dlp

        out_dir.mkdir(parents=True, exist_ok=True)
        for old in glob.glob(str(out_dir / "audio.*")):
            Path(old).unlink(missing_ok=True)

        def hook(d: Dict) -> None:
            control.check()
            if d.get("status") == "downloading" and d.get("total_bytes") or d.get("total_bytes_estimate"):
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                control.report("audio", d.get("downloaded_bytes", 0) / total,
                               f"audio {d.get('downloaded_bytes', 0) / 1e6:.0f} of {total / 1e6:.0f} MB")

        # A fresh extraction (not self._info): replaying an old info dict into the plain HTTP downloader gets HTTP 403.
        opts = ydl_opts({"format": "ba[abr<=70]/ba/b", "outtmpl": str(out_dir / "audio.%(ext)s"),
                         "progress_hooks": [hook]})
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([self.url])
        except Exception as e:  # noqa: BLE001
            raise SourceError(friendly_error(e)) from e
        finally:
            for part in glob.glob(str(out_dir / "audio.*.part")):
                Path(part).unlink(missing_ok=True)
        found = sorted(glob.glob(str(out_dir / "audio.*")))
        if not found:
            raise SourceError("The audio could not be fetched.")
        return Path(found[0])

    # ---- links -------------------------------------------------------------------------------
    def thumbnail(self) -> Optional[str]:
        if self.kind == "youtube":
            return f"https://i.ytimg.com/vi/{self.id}/mqdefault.jpg"
        return (self._info or {}).get("thumbnail")

    def watch_url(self, seconds: float = 0) -> str:
        if self.kind == "youtube":
            return f"https://youtu.be/{self.id}?t={int(seconds)}"
        return self.url


def resolve(target: str) -> Source:
    """Describe a link or file. Network cost: one metadata request (~5 s). Nothing is downloaded."""
    target = (target or "").strip()
    if not target:
        raise SourceError("Paste a video link or the path of a video file.")

    path = _local_path(target)
    if path is not None:
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:6]
        return Source(
            kind="file", id=f"{_safe_id(path.stem)[:60]}-{digest}", url=str(path.resolve()), title=path.stem,
            duration=_ffprobe_duration(str(path)), embeddable=False, local_path=str(path.resolve()),
        )
    if not _is_url(target):
        raise SourceError(f"That is neither a web link nor an existing file: {target[:120]}")

    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(ydl_opts({"skip_download": True})) as ydl:
            info = ydl.extract_info(target, download=False)
    except Exception as e:  # noqa: BLE001
        raise SourceError(friendly_error(e)) from e
    if not info:
        raise SourceError("Nothing could be read from that link.")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        raise SourceError(
            "That link is a playlist. Paste the link of one video." if len(entries) != 1 else "Open the video itself, not its playlist."
        )
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise SourceError("Live streams and premieres cannot be analysed until they have finished.")

    extractor = str(info.get("extractor_key") or info.get("extractor") or "web")
    is_youtube = extractor.lower().startswith("youtube")
    vid = str(info.get("id") or "video")
    return Source(
        kind="youtube" if is_youtube else "web",
        id=_safe_id(vid if is_youtube else f"{extractor.lower()}_{vid}"),
        url=target, title=str(info.get("title") or vid), duration=float(info.get("duration") or 0.0),
        language=(info.get("language") or None), embeddable=bool(info.get("playable_in_embed", True)),
        heatmap=info.get("heatmap") or None,
        chapters=[{"start_time": c.get("start_time"), "end_time": c.get("end_time"), "title": c.get("title")}
                  for c in (info.get("chapters") or [])] or None,
        channel=str(info.get("channel") or info.get("uploader") or "")[:80],
        _info=info,
    )


# --------------------------------------------------------------------------------------------- #
# Clips: fetch only the requested time ranges
# --------------------------------------------------------------------------------------------- #
def _fast_decode(path: Path) -> Path:
    """OpenCV decodes H.264 at ~90 fps but AV1 1080p at ~10 fps, so anything else is converted once."""
    if probe_codec(str(path)) in ("h264", ""):
        return path
    fixed = path.with_name(path.stem + ".h264.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", "fps=30", "-c:v", "libx264",
         "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(fixed)],
        check=True,
    )
    path.unlink(missing_ok=True)
    return fixed


def _fetch_range(source: Source, index: int, start: float, end: float, out_dir: Path, max_height: int) -> Path:
    import yt_dlp
    from yt_dlp.utils import download_range_func

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sec{index:02d}_{int(start)}_{int(end)}"
    for old in glob.glob(str(out_dir / (stem + ".*"))):
        Path(old).unlink(missing_ok=True)
    h = int(max_height)
    opts = ydl_opts({
        # H.264 first (fast to decode); height is a ceiling, not a requirement
        "format": f"bv*[height<={h}]+ba/b[height<={h}]/bv*+ba/b",
        "format_sort": [f"res:{h}", "vcodec:h264"],
        "download_ranges": download_range_func(None, [(max(0.0, start), end)]),
        "outtmpl": str(out_dir / (stem + ".%(ext)s")), "merge_output_format": "mp4",
        "concurrent_fragment_downloads": 4,
    })
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(copy.deepcopy(source._info), download=True)
    except Exception as e:  # noqa: BLE001
        raise SourceError(friendly_error(e)) from e
    found = sorted(p for p in glob.glob(str(out_dir / (stem + ".*"))) if not p.endswith((".part", ".ytdl")))
    if not found:
        raise SourceError("The clip could not be fetched from the video.")
    return _fast_decode(Path(found[0]))


class SectionFetcher:
    """Fetches the time ranges of a stream in the background, two at a time.

    The pipeline renders clip N (CPU) while clip N+1 is still arriving (network), so the wait for the
    network mostly disappears. ``result(i)`` blocks until range i is on disk.
    """

    def __init__(self, source: Source, ranges: Sequence[Tuple[float, float]], out_dir: Path,
                 max_height: int = 1080, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="fetch")
        self._futures: List[Future] = [
            self._pool.submit(_fetch_range, source, i, s, e, out_dir, max_height) for i, (s, e) in enumerate(ranges)
        ]

    def result(self, i: int) -> Path:
        while True:
            control.check()
            try:
                return self._futures[i].result(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                raise

    def close(self) -> None:
        for f in self._futures:
            f.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "SectionFetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
