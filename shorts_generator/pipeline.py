"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — runs on this machine. The video is NEVER downloaded as a whole:
                              captions (or Whisper on the audio track) give the transcript, the LLM
                              ranks the moments, and only the clips you render are fetched from the
                              video site, as their own time ranges (see local/source.py).

Local mode is split in steps so a UI can show the ranked candidates first and render only the ones a
person picks:
  analyze_local()    link -> transcript -> signals -> ranked candidates   (saved as a project)
  reanalyze_local()  same video, new focus / clip length (no new transcript)
  render_local()     fetch + reframe (+ captions, title, watermark) the chosen candidates
  publish_kit_local() title / description / hashtags for candidates
  upload_local()     send the queued, rendered clips to your YouTube channel (local/youtube.py, local/uploads.py)

Everything heavy is imported inside the functions, so api mode (and ``--help``) stays light.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

from . import config
from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


# --------------------------------------------------------------------------------------------- #
# Local mode
# --------------------------------------------------------------------------------------------- #
@dataclass
class AnalyzeOptions:
    candidates: int = 10                 # how many ranked candidates to keep
    language: Optional[str] = None       # caption / speech language (None = the video's own)
    focus: str = ""                      # free text: "funny moments", "key insights"...
    min_seconds: int = 20
    max_seconds: int = 60
    use_audio: bool = True               # voice-energy signal (fetches the audio track, never the video)
    use_replay: bool = True              # YouTube "most replayed" signal when the video has one
    refresh: bool = False                # ignore cached transcript / signals

    def stored(self) -> Dict:
        d = asdict(self)
        d.pop("refresh", None)
        return d


MAX_CANDIDATES = 40


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _check_disk(need_mb: int) -> None:
    """Fail early, with a clear message, instead of half-way through a render."""
    from pathlib import Path

    from .config import LOCAL_OUTPUT_DIR
    from .local.health import require_free_space

    require_free_space(Path(LOCAL_OUTPUT_DIR), need_mb)


def _validate_analyze(o: AnalyzeOptions) -> None:
    if not (5 <= o.min_seconds < o.max_seconds <= 180):
        raise ValueError("Clip length must satisfy 5 <= minimum < maximum <= 180 seconds.")
    longest = int(config.SHORT_SECONDS)                       # moments are picked to fit a Short, not cut to fit later
    o.max_seconds = min(o.max_seconds, longest)
    o.min_seconds = min(o.min_seconds, longest - 5)
    o.candidates = max(3, min(MAX_CANDIDATES, int(o.candidates)))
    o.focus = " ".join((o.focus or "").split())[:300]


def _kept_clips(project_id: str, clips: Optional[List[Dict]] = None) -> List[Dict]:
    """Clips rendered for an earlier analysis of this video stay in the gallery, detached from the new candidates."""
    from .local.project import load_project

    if clips is None:
        try:
            clips = load_project(project_id).get("clips", [])
        except KeyError:
            clips = []
    return [{**c, "candidate": None} for c in clips if c.get("file")]


def _build_project(meta: Dict, get_source: Callable, opts: AnalyzeOptions, old_clips: Optional[List[Dict]] = None) -> Dict:
    """Shared by analyze_local and reanalyze_local. ``get_source()`` resolves the link only if the network is needed."""
    from .highlights import Options
    from .local import control
    from .local.audio import energy_pct_fn, load_energy_db
    from .local.llm import call_local_llm
    from .local.project import cache_dir, new_project, save_project
    from .local.signals import heatmap_series, window_pct_fn
    from .local.transcript import get_transcript

    cdir = cache_dir(meta["id"])
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="signals")
    try:
        # the audio track is a network job: let it run while the captions and the LLM work
        energy_future = pool.submit(load_energy_db, get_source, cdir, opts.refresh) if opts.use_audio else None

        control.report("transcript", None, "Getting the transcript")
        transcript = get_transcript(get_source, cdir, language=opts.language, refresh=opts.refresh, duration=meta.get("duration"))

        signals: Dict = {}
        pending: Dict = {}
        replay = heatmap_series(meta.get("heatmap"), float(meta.get("duration") or transcript.get("duration") or 0)) \
            if opts.use_replay else None
        if replay is not None:
            signals["replay"] = window_pct_fn(replay)
            print("[signals] YouTube 'most replayed' data found", flush=True)
        if energy_future is not None:
            pending["energy"] = lambda: energy_pct_fn(energy_future.result())

        control.report("rank", 0.0, "Finding the best moments")
        hl = get_highlights(
            transcript, num_clips=opts.candidates, llm_fn=call_local_llm,
            options=Options(focus=opts.focus, min_seconds=opts.min_seconds, max_seconds=opts.max_seconds,
                            signals=signals, pending=pending),
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    candidates = sorted(hl.get("highlights", []), key=lambda h: int(h.get("score", 0)), reverse=True)
    if not candidates:
        raise RuntimeError("The LLM did not find any moments to suggest.")
    # "Moments to find" is a promise: an 11-hour stream used to come back with 280 (every chunk's candidates), unusable to browse
    candidates = candidates[: opts.candidates]
    stored = opts.stored()
    stored["signals"] = sorted({n for h in candidates for n in (h.get("signals") or {})})
    project = new_project(meta, transcript, candidates, stored)
    project["clips"] = _kept_clips(meta["id"], old_clips)       # re-analysing never empties the gallery
    path = save_project(project)
    print(f"[pipeline] {len(candidates)} candidates ranked; project saved: {path}", flush=True)
    return project


def analyze_local(target: str, options: Optional[AnalyzeOptions] = None) -> Dict:
    """Link (or local file) -> ranked candidates. Returns, and saves, a project dict."""
    from .local import control
    from .local.source import resolve

    opts = options or AnalyzeOptions()
    _validate_analyze(opts)
    _check_disk(100)
    control.report("source", None, "Reading the link")
    source = resolve(target)
    print(f"[source] {source.kind}: {source.title} ({_clock(source.duration)})", flush=True)
    return _build_project(source.meta(), lambda: source, opts)


def reanalyze_local(project: Dict, options: AnalyzeOptions) -> Dict:
    """Same video, new focus / clip length. Re-uses the cached transcript and signals (no network if cached)."""
    from .local.source import resolve

    _validate_analyze(options)
    meta = project["source"]
    holder: List = []

    def get_source():
        if not holder:
            holder.append(resolve(meta.get("local_path") or meta["url"]))
        return holder[0]

    return _build_project(meta, get_source, options, old_clips=project.get("clips", []))


def render_local(
    project: Dict,
    candidate_ids: List[int],
    options=None,
    *,
    quality: int = 1080,
    overrides: Optional[Dict[int, Tuple[float, float]]] = None,
    kit: bool = False,
) -> List[Dict]:
    """Render the chosen candidates of ``project`` and record them in project.json.

    Streamed videos: only the time range of each clip is fetched (at most ``quality`` pixels high,
    H.264), two ranges ahead while the previous clip renders. ``overrides`` = {candidate id: (start, end)}
    are the person's own trims; they are saved onto the candidate. Clips land in
    ``<output>/clips_<video id>/short_<NN>_<title>.mp4`` (NN = candidate id + 1), so re-rendering a
    candidate replaces its file instead of piling up copies.
    """
    from .local import control, romanize
    from .local import quality as qa
    from .local.clipper import RenderOptions, clip_filename, render_clip
    from .local.project import cache_dir, project_dir, update_project
    from .local.source import SectionFetcher, friendly_error, resolve

    options = options or RenderOptions()
    options.validate()
    _check_disk(400)
    ids = sorted({int(i) for i in candidate_ids})
    cands = project["candidates"]
    if not ids or any(i < 0 or i >= len(cands) for i in ids):
        raise ValueError("Pick existing candidates to render.")

    duration = float(project.get("duration") or 0)
    for cid, (s, e) in (overrides or {}).items():
        s, e = float(s), float(e)
        if not (0 <= s < e) or (duration and e > duration + 1) or not (3 <= e - s <= 180):
            raise ValueError(f"Clip #{int(cid) + 1}: use a start before the end, inside the video, 3 to 180 seconds long.")
        cands[int(cid)].update(start_time=s, end_time=e, edited=True)

    effects = options.active_effects()
    editing_on = any(effects.values())
    src = project["source"]
    local_path = src.get("local_path") if src.get("local_path") and os.path.isfile(src["local_path"]) else None
    transcript = None
    if options.captions or kit or editing_on:
        try:
            transcript = _transcript_for(project)
        except KeyError:
            if not editing_on:
                raise
            print("[clip] no transcript for this video: editing without captions", flush=True)
    segments = transcript["segments"] if (transcript and (options.captions or editing_on)) else None
    options = replace(options, watermark=_brand(options.watermark))      # only the person's own channel is shown
    source = None
    if local_path is None and transcript and editing_on and not src.get("channel"):
        source = resolve(src["url"])                     # an older project: find out whose video it is, for the titles
        if source.channel:
            src["channel"] = source.channel
            update_project(project, lambda fresh: fresh["source"].__setitem__("channel", source.channel))
            cands, src = project["candidates"], project["source"]
            for cid, (s, e) in (overrides or {}).items():
                cands[int(cid)].update(start_time=float(s), end_time=float(e), edited=True)
    wants_memes = bool(effects.get("memes") and _meme_menu())
    if transcript and (effects.get("popups") or effects.get("hook") or wants_memes):
        _edit_notes_for(project, ids, wants_memes)   # what appears when: one AI call for all clips; the edit works without it
        cands = project["candidates"]          # the project was re-read from disk: put the person's trims back on it
        for cid, (s, e) in (overrides or {}).items():
            cands[int(cid)].update(start_time=float(s), end_time=float(e), edited=True)
    out_dir = project_dir(project["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(float(cands[i]["start_time"]), float(cands[i]["end_time"])) for i in ids]
    if not _is_wide(options.aspect_ratio):               # a wide video is not a Short: its length is the person's choice
        for n, (start, end) in enumerate(ranges):
            fits = _short_end(start, end, (transcript or {}).get("segments"))
            if fits < end:
                print(f"[clip] #{ids[n] + 1} is {end - start:.0f}s: cut to {fits - start:.0f}s so the Short stays under a minute", flush=True)
                ranges[n] = (start, fits)
    roman = config.caption_script() == "roman"          # Hindi / Urdu letters become Roman ones on screen
    if roman:
        segments = _roman_text(project, ids, ranges, segments)

    records: List[Dict] = []
    fetcher = None
    try:
        if local_path is None:
            if source is None:
                control.report("source", None, "Contacting the video site")
                source = resolve(src["url"])
            fetcher = SectionFetcher(source, ranges, cache_dir(project["id"]) / "sections", max_height=int(quality))
        for n, cid in enumerate(ids):
            control.check()
            c = cands[cid]
            start, end = ranges[n]
            fname = clip_filename(cid + 1, c.get("title", ""))
            out_path = str(out_dir / fname)
            print(f"[clip] {n + 1}/{len(ids)}: {c.get('title', '')} ({_clock(start)}-{_clock(end)})", flush=True)
            err: Optional[str] = None
            info: Dict = {}
            hints = _hints_for(c, start, end)
            title = c.get("title", "")
            if roman:
                title, hints = romanize.text(title), romanize.hints(hints)
            with control.scale(n / len(ids), 1 / len(ids)):
                try:
                    if fetcher is not None:
                        control.report("fetch", None, "fetching just this part of the video")
                        section = fetcher.result(n)
                        try:
                            render_clip(str(section), out_path, start, end, options, transcript_segments=segments,
                                        title=title, precut=True, hints=hints, info=info)
                        finally:
                            section.unlink(missing_ok=True)
                    else:
                        render_clip(local_path, out_path, start, end, options, transcript_segments=segments,
                                    title=title, precut=False, hints=hints, info=info)
                    if info.get("size"):                        # never keep (or upload) a broken clip
                        problems = qa.verify(out_path, info["size"], info["duration"], info.get("has_audio", True))
                        if problems:
                            raise RuntimeError("Quality check failed: " + "; ".join(problems))
                except Exception as e:  # noqa: BLE001 — one failed clip must not lose the others
                    err = friendly_error(e) if "yt_dlp" in type(e).__module__ else (str(e) or type(e).__name__)
                    print(f"[clip] {n + 1} failed: {err}", flush=True)
            records.append({
                "candidate": cid, "file": None if err else fname, "title": title,
                "start": start, "end": end, "layout": options.layout, "captions": bool(options.captions),
                "caption_style": options.caption_style, "title_card": bool(options.title_card),
                "watermark": options.watermark, "aspect_ratio": options.aspect_ratio, "quality": int(quality),
                "error": err, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "edit": options.edit, "edit_summary": info.get("edit"), "framing": info.get("why"), "encoder": info.get("encoder"),
            })
    finally:
        if fetcher is not None:
            fetcher.close()
        done = set(ids)       # keep what was finished even if the job was cancelled half-way
        trims = {int(k): (float(v[0]), float(v[1])) for k, v in (overrides or {}).items()}

        def merge(fresh: Dict) -> None:
            # The job ran for minutes on its own copy. Re-read the saved project and change only what this job owns,
            # so a clip the person added (or deleted) in the meantime is not undone.
            for cid, (s, e) in trims.items():
                if cid < len(fresh["candidates"]):
                    fresh["candidates"][cid].update(start_time=s, end_time=e, edited=True)
            fresh["clips"] = [c for c in fresh.get("clips", []) if c.get("candidate") not in done] + records

        update_project(project, merge)

    if kit:
        missing = [r["candidate"] for r in records if r["file"] and not (project["candidates"][r["candidate"]].get("kit") or {}).get("title")]
        publish_kit_local(project, missing)
    return records


def _short_end(start: float, end: float, segments) -> float:
    """Where a clip ends so that the Short stays under a minute: after the last sentence that still fits."""
    limit = start + config.SHORT_SECONDS
    if end <= limit:
        return end
    stops = [float(s["end"]) for s in segments or [] if limit - 15 <= s["end"] <= limit]
    return max(stops) if stops else limit


def _is_wide(aspect_ratio: str) -> bool:
    width, _, height = str(aspect_ratio).partition(":")
    try:
        return float(width) > float(height)
    except ValueError:
        return False


def _brand(typed: str = "") -> str:
    """The branding on a clip: what the person typed ("-" = none), else the name of their connected YouTube channel."""
    from .local import youtube

    typed = (typed or "").strip()
    if typed:
        return "" if typed == "-" else typed
    return str((youtube.status().get("channel") or {}).get("title") or "").strip()[:40]


def _transcript_for(project: Dict) -> Dict:
    """The project's captions; saved ones that were cleaned away are fetched again (about 3 seconds). KeyError if none."""
    from .local.project import cache_dir, load_transcript
    from .local.source import resolve
    from .local.transcript import get_transcript

    try:
        return load_transcript(project)
    except (KeyError, ValueError, OSError):
        pass
    url = project.get("source", {}).get("url")
    if not url or project.get("source", {}).get("kind") == "file":
        raise KeyError("transcript")
    print("[transcript] the saved captions are gone: fetching them again", flush=True)
    try:
        return get_transcript(lambda: resolve(url), cache_dir(project["id"]), language=(project.get("transcript_info") or {}).get("language"),
                              allow_whisper=False)
    except Exception as e:  # noqa: BLE001 - the caller decides whether it can go on without them
        print(f"[transcript] could not fetch the captions ({str(e)[:160]})", flush=True)
        raise KeyError("transcript") from e


def _roman_text(project: Dict, ids: List[int], ranges, segments):
    """Spell everything these clips will show in Roman letters; returns the caption segments rewritten that way."""
    from .local import control, romanize
    from .local.llm import call_local_llm

    texts: List[str] = []
    for i, (start, end) in zip(ids, ranges):
        c = project["candidates"][i]
        kit = c.get("kit") or {}
        texts += [c.get("title", ""), kit.get("title", ""), kit.get("hook", ""), kit.get("description", "")]
        texts += [p.get("text") or "" for p in kit.get("popups") or []] + list(kit.get("keywords") or [])
        texts += [s.get("text", "") for s in segments or [] if s["end"] > start and s["start"] < end]
    if any(romanize.needed(t) for t in texts):
        control.report("kit", None, "Writing the subtitles in Roman letters")
        romanize.prepare(texts, call_local_llm)
    return romanize.segments(segments, ranges) if segments else segments


EDIT_NOTES_SECONDS = 75          # a busy AI service must not hold up the render: the edit has its own fallback


def _meme_menu():
    from .local import memes

    return memes.menu(memes.scan())


def _edit_notes_for(project: Dict, ids: List[int], wants_memes: bool = False) -> None:
    """Ask the AI for headline / popups / highlighted words / memes (with title and hashtags) for clips that have none yet."""
    from .local.publish import STYLE

    needed = ["popups"] + (["memes"] if wants_memes else [])
    missing = [i for i in ids if (project["candidates"][i].get("kit") or {}).get("style") != STYLE
               or any(k not in project["candidates"][i]["kit"] for k in needed)]
    if not missing:
        return
    from .local.llm import time_budget

    try:
        with time_budget(EDIT_NOTES_SECONDS):
            publish_kit_local(project, missing, with_edit=True)
    except Exception as e:  # noqa: BLE001 - the edit still happens, from the words and the loudness
        print(f"[edit] the AI could not plan the popups ({str(e)[:160]}); using the clip's own words and reactions", flush=True)


def _hints_for(candidate: Dict, start: float, end: float) -> Dict:
    kit = candidate.get("kit") or {}
    popups = [{"at": p["t"] - start, "emoji": p.get("emoji"), "text": p.get("text"), **({"mood": p["mood"]} if p.get("mood") else {})}
              for p in kit.get("popups") or [] if start <= p.get("t", -1) < end]
    hints = {"hook": kit.get("hook") or kit.get("title") or "", "popups": popups, "keywords": kit.get("keywords") or []}
    if "memes" in kit:                                   # the AI was asked which memes fit (an empty list: none did)
        hints["memes"] = [{"at": m["t"] - start, "id": m.get("id")} for m in kit["memes"] or [] if start <= m.get("t", -1) <= end + 0.5]
    return hints


def publish_kit_local(project: Dict, candidate_ids: List[int], with_edit: bool = False) -> Dict[int, Dict]:
    """Write a title, description and hashtags for each candidate and store them on the project.

    ``with_edit``: the same AI call also plans the headline, popups and highlighted words of the automatic edit.
    """
    from .local import control
    from .local.llm import call_local_llm
    from .local.project import update_project
    from .local.publish import make_publish_kit

    ids = sorted({int(i) for i in candidate_ids})
    if not ids:
        return {}
    control.report("kit", None, "Writing titles, descriptions and hashtags")
    kit = make_publish_kit(project, _transcript_for(project), ids, call_local_llm, with_edit=with_edit,
                           meme_menu=_meme_menu() if with_edit else (), brand=_brand())

    def put(fresh: Dict) -> None:
        for cid, k in kit.items():
            if cid < len(fresh["candidates"]):
                fresh["candidates"][cid]["kit"] = k

    update_project(project, put)
    return kit


def upload_local(project_id: str) -> Dict:
    """Upload the queued clips of a project to the connected YouTube channel (the queue lives in local/uploads.py).

    Titles, descriptions and hashtags come from the publish kit (written first, in one AI call, for clips that have none).
    Clips that cannot go now (daily cap, YouTube's quota, no internet) stay queued; the web app's watcher retries them.
    """
    from .local import uploads

    return uploads.drain(project_id, ensure_kit=publish_kit_local)


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    quality: str,
    language: Optional[str],
    layout: str = "auto",
    captions: bool = False,
    focus: str = "",
    min_seconds: int = 20,
    max_seconds: int = 60,
    caption_style: str = "classic",
    title_card: bool = False,
    watermark: str = "",
    normalize_audio: bool = True,
    kit: bool = False,
    refresh: bool = False,
    edit: str = "off",
) -> Dict:
    from .local.clipper import RenderOptions

    project = analyze_local(youtube_url, AnalyzeOptions(
        candidates=num_clips, language=language, focus=focus, min_seconds=min_seconds, max_seconds=max_seconds,
        refresh=refresh))
    top_ids = list(range(min(num_clips, len(project["candidates"]))))     # candidates are ranked best-first
    print(f"[pipeline] rendering the top {len(top_ids)} of {len(project['candidates'])} candidates", flush=True)

    shorts = render_local(
        project, top_ids,
        RenderOptions(aspect_ratio=aspect_ratio, layout=layout, captions=captions, edit=edit,
                      caption_style=caption_style or ("pop" if edit != "off" else "classic"),
                      title_card=title_card, watermark=watermark, normalize_audio=normalize_audio),
        quality=int(quality) if str(quality).isdigit() else 1080, kit=kit,
    )
    out_dir = os.path.join(os.environ.get("LOCAL_OUTPUT_DIR", "output"), f"clips_{project['id']}")
    results = []
    for rec in shorts:
        c = project["candidates"][rec["candidate"]]
        results.append({**c, "clip_url": os.path.join(out_dir, rec["file"]) if rec["file"] else None, "error": rec["error"]})
    return {
        "mode": "local",
        "project_id": project["id"],
        "source_video_url": project["source"]["url"],
        "transcript_source": project.get("transcript_info", {}).get("source"),
        "highlights": project["candidates"],
        "shorts": results,
    }


# --------------------------------------------------------------------------------------------- #
# API mode (MuAPI does everything server-side)
# --------------------------------------------------------------------------------------------- #
def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    layout: str = "auto",
    captions: bool = False,
    focus: str = "",
    min_seconds: int = 20,
    max_seconds: int = 60,
    caption_style: Optional[str] = None,
    title_card: bool = False,
    watermark: str = "",
    normalize_audio: bool = True,
    quality: str = "1080",
    kit: bool = False,
    refresh: bool = False,
    edit: str = "off",
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: link (local mode also accepts a file path).
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: api mode only: resolution MuAPI downloads the source at.
        language: ISO-639-1 caption / speech language (default: the video's own).
        mode: "api" (default, MuAPI) or "local" (nothing is downloaded as a whole; see the module doc).
        Local mode only:
          layout: "auto" picks per clip: "track" (one stable crop following a talking head), "split"
              (facecam on top, gameplay below) or "fit" (whole frame over a blurred background).
          captions / caption_style: burn big-type captions ("pop", "classic", "yellow", "box", "karaoke").
          edit: automatic editing, "off" | "clean" | "viral" (jump cuts, zooms, popups, sounds, headline, progress bar).
          title_card / watermark: the clip's title for its first seconds / small corner text.
          focus: steer the search ("funny moments", "key insights"); min_seconds / max_seconds: clip length.
          quality: tallest source picture to fetch per clip ("720", "1080", ...). normalize_audio: -14 LUFS.
          kit: also write a title, description and hashtags per clip. refresh: ignore cached transcript.

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # the top `num_clips`, each with clip_url / error
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(youtube_url, num_clips, aspect_ratio, quality, language, layout, captions, focus,
                          min_seconds, max_seconds, caption_style, title_card, watermark, normalize_audio, kit, refresh, edit)
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language)
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
