"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py, then extended:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt, steerable with a free-text *focus* ("funny moments", "key insights"...)
  - a clip-length range the editor asked for (also enforced after the LLM answers)
  - one GLOBAL re-rank over every chunk's candidates (chunk scores are not comparable)
  - blending of signals that do not depend on speech recognition: voice energy, YouTube "most replayed"
  - score-based dedupe with overlap suppression, and clip edges snapped to sentence boundaries

The LLM call is pluggable via the `llm_fn` argument so the same prompts can drive either MuAPI
(--mode api) or a direct local LLM client (--mode local). This module knows nothing about video files.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import muapi


LLMFn = Callable[[str], str]
PctFn = Callable[[float, float], float]


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}
{focus_note}
Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
- {duration_rule}
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality)
- {num_clips_instruction}
- For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
- Explain in one sentence why this clip is viral ("virality_reason")

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}"""


FOCUS_NOTE = """
EDITOR'S BRIEF for this video: "{focus}"
Look for moments that fit this brief and rank them above generic viral moments. Skip moments that do not fit it, even if they are entertaining.
"""


RERANK_PROMPT = """You are the head editor choosing which clips from ONE long video get published as YouTube Shorts / Reels.
{n} candidate clips were found by scanning the video in separate chunks, so their scores are NOT comparable with each other (every chunk rated its own best moments 85-95). Compare the candidates directly with each other and choose the best {k}.

Video: {content_type} | length about {minutes} minutes
{focus_note}
{virality_criteria}
{signal_notes}
Judge each candidate from its transcript excerpt as well as from its title:
- it opens with a hook within the first 3 seconds
- it is self-contained: a viewer who has not seen the rest of the video understands it
- it has a clear payoff: a laugh, surprise, twist, strong opinion or insight
- it is NOT filler: greetings, intro/outro, sponsor reads, menu/loading chatter, or set-up without a payoff
- the transcript comes from automatic speech recognition and may be garbled: do not punish garbled words when the title and reason clearly describe a strong moment, but do not reward invented drama either

Use the full 0-100 scale and be harsh. At most two clips may score above 90; an average candidate is 50-60; filler is below 40.
When two candidates are about equally good, prefer the one from a different part of the video.

Candidates:
{candidates}

Respond ONLY with valid JSON (no markdown): {{"ranking":[{{"id":int,"score":int,"reason":"string"}}]}}
Each reason is at most 15 plain words and never contains a double quotation mark.
List the best {m} candidates, best first. Use the ids given above."""


# name -> (label shown to the LLM, prompt note, blend weight)
SIGNAL_INFO = {
    "energy": (
        "voice energy",
        'Each candidate also shows its "voice energy": how loud its audio is compared with every other stretch of the same length in this video (95 = louder than 95% of them, 50 = typical).\n'
        "In gaming, streaming, vlog and reaction content the best moments are the speaker's own excited reactions (shouting, laughing, panic), so strongly prefer high energy. Treat a quiet candidate (below 30) as probably game dialogue, a cutscene, narration or silence, and pick it only if the text is exceptional.",
        0.30,
    ),
    "replay": (
        "most replayed",
        'Each candidate also shows "most replayed": how much real viewers re-watched that stretch of the video compared with every other stretch of the same length (95 = replayed more than 95% of them). This is genuine audience data: treat it as strong evidence of a moment people care about.',
        0.35,
    ),
}
MAX_SIGNAL_WEIGHT = 0.6          # the LLM always keeps at least 40% of the final score

CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
MAX_RERANK_CANDIDATES = 40
SNAP_TOLERANCE_SECONDS = 4.0    # how far a clip edge may move to reach a sentence boundary
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # cap LLM polls at 5 min — a wedged call should fail fast
MAX_HIGHLIGHT_API_ATTEMPTS = 3


@dataclass
class Options:
    """What the person asked for. Defaults suit YouTube Shorts / Reels."""
    focus: str = ""                              # free text steering the search ("funny moments", "key insights")
    min_seconds: int = 20
    max_seconds: int = 60
    signals: Dict[str, PctFn] = field(default_factory=dict)      # "energy" / "replay" -> pct(start, end)
    # Signals still being computed in the background (e.g. the audio loudness, a network-bound job that
    # runs while the LLM reads the transcript): zero-argument resolvers returning a PctFn or None.
    pending: Dict[str, Callable[[], Optional[PctFn]]] = field(default_factory=dict)

    def resolve_signals(self) -> Dict[str, PctFn]:
        """Wait for pending signals (a failed one is simply dropped) and return all that are available."""
        for name, resolver in list(self.pending.items()):
            self.pending.pop(name)
            try:
                fn = resolver()
            except Exception:           # a bonus signal must never fail the analysis (Cancelled is a BaseException)
                fn = None
            if fn is not None:
                self.signals[name] = fn
        return self.signals

    def duration_rule(self) -> str:
        mid = (self.min_seconds + self.max_seconds) // 2
        return (f"Duration: between {self.min_seconds} and {self.max_seconds} seconds, ideally around {mid}. "
                f"Never longer than {self.max_seconds}s and never shorter than {self.min_seconds}s — end on a complete thought")

    def focus_note(self) -> str:
        focus = " ".join(self.focus.split())[:300]
        return FOCUS_NOTE.format(focus=focus) if focus else ""


def call_muapi_llm(prompt: str) -> str:
    """Default LLM backend: MuAPI gpt-5-mini."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def _parse_json_loose(raw: str) -> Dict:
    """Models sometimes wrap JSON in markdown fences — strip and parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sanitize_highlights(raw_highlights: object, duration: float) -> List[Dict]:
    """Normalize model output into the expected shape; skip invalid entries."""
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue

        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip(),
                "start_time": start,
                "end_time": end,
                "score": max(0, min(100, _coerce_int(item.get("score"), default=0))),
                "hook_sentence": str(item.get("hook_sentence") or "").strip(),
                "virality_reason": str(item.get("virality_reason") or "").strip(),
            }
        )

    return cleaned


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        # Shift timestamps to be relative to the chunk start. get_highlights() adds
        # `_offset` back afterwards, and _sanitize_highlights() clamps to the chunk
        # duration — both assume chunk-relative times. Without this, every chunk
        # after the first had its highlights discarded (absolute time > duration).
        chunk_segs = [
            {"start": s["start"] - start, "end": s["end"] - start, "text": s["text"]}
            for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = {k: v for k, v in transcript.items() if k != "segments"}
            chunk["segments"] = chunk_segs
            # Include the overlap tail so highlights there aren't clamped away.
            chunk["duration"] = max(end - start, chunk_segs[-1]["end"])
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
    options: Optional[Options] = None,
) -> Dict:
    options = options or Options()
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        focus_note=options.focus_note(),
        duration_rule=options.duration_rule(),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
    )
    base_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"
    empty_answers = 0

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            if isinstance(parsed, list):  # some models return the bare array
                parsed = {"highlights": parsed}
            items = parsed.get("highlights")
            highlights = _sanitize_highlights(items, duration=duration)
            if highlights:
                return {"highlights": highlights}
            last_error = "no valid highlights in response"
            if isinstance(items, list) and not items:
                empty_answers += 1
                # Asked twice (the second time firmly) and still nothing: a stretch of a long stream can simply have
                # no moment (waiting screen, music, silence). That is an answer, not an error.
                if is_chunk and empty_answers >= 2:
                    return {"highlights": []}
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(
                f"[highlights] invalid model output on attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS}; retrying",
                flush=True,
            )
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, score, hook_sentence, virality_reason."
                + " No markdown fences, no commentary."
                + (f" (try {attempt + 1})" if attempt >= 2 else "")      # a new question: not the saved answer again
            )

    message = f"Highlight generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    if is_chunk:                                  # one part of a long video: the caller carries on with the others
        return {"highlights": [], "failed": message}
    raise RuntimeError(message)


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _excerpt(transcript: Dict, start: float, end: float, limit: int = 450) -> str:
    parts = [
        s["text"].strip()
        for s in transcript.get("segments", [])
        if s["end"] > start and s["start"] < end and s.get("text", "").strip()
    ]
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def signal_weights(names) -> Dict[str, float]:
    """Blend weight per available signal, scaled so the LLM keeps at least 40% of the final score."""
    raw = {n: SIGNAL_INFO[n][2] for n in names if n in SIGNAL_INFO}
    total = sum(raw.values())
    scale = min(1.0, MAX_SIGNAL_WEIGHT / total) if total else 1.0
    return {n: w * scale for n, w in raw.items()}


def blend_signals(highlights: List[Dict], signals: Dict[str, PctFn]) -> List[Dict]:
    """Attach ``signals: {name: percentile}`` to each candidate and blend them into its score.

    Used where there is no global re-rank (videos short enough for a single LLM call, or when the
    re-rank failed): the LLM's own score keeps at least 40% of the final value.
    """
    if not signals:
        return highlights
    weights = signal_weights(list(signals))
    w_total = sum(weights.values())
    out = []
    for h in highlights:
        vals = {n: fn(float(h["start_time"]), float(h["end_time"])) for n, fn in signals.items()}
        llm = int(h.get("score", 0))
        score = int(round((1 - w_total) * llm + sum(weights[n] * v for n, v in vals.items() if n in weights)))
        out.append({**h, "llm_score": llm, "score": score, "signals": {n: round(v) for n, v in vals.items()}})
    return sorted(out, key=lambda x: x["score"], reverse=True)


def _ranking_entries(raw: object, n: int) -> List[Dict]:
    """Valid {id, score, reason} entries (ids inside 0..n-1, each once) from a parsed ranking list."""
    seen = set()
    out: List[Dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        idx = _coerce_int(item.get("id"), default=-1)
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            out.append({"id": idx, "score": max(0, min(100, _coerce_int(item.get("score"), default=0))),
                        "reason": str(item.get("reason") or "").strip()})
    return out


def salvage_ranking(raw: str, n: int) -> List[Dict]:
    """Pull ids and scores out of a ranking answer that is not valid JSON.

    One unescaped quote inside a reason breaks the whole document (it did, for an 11-hour stream with Hinglish reasons), yet
    every ``"id": 3 ... "score": 87`` pair in it is still perfectly usable.
    """
    out: List[Dict] = []
    seen = set()
    for obj in (raw or "").split("{")[1:]:
        m_id = re.search(r'"id"\s*:\s*(\d+)', obj)
        m_score = re.search(r'"score"\s*:\s*(\d+)', obj)
        if not (m_id and m_score):
            continue
        idx = int(m_id.group(1))
        if not (0 <= idx < n) or idx in seen:
            continue
        seen.add(idx)
        m_reason = re.search(r'"reason"\s*:\s*"(.*?)(?:"\s*[,}\]]|$)', obj, re.S)
        out.append({"id": idx, "score": max(0, min(100, int(m_score.group(1)))),
                    "reason": (m_reason.group(1).strip() if m_reason else "")[:300]})
    return out


def rerank_highlights(
    highlights: List[Dict],
    transcript: Dict,
    num_clips: int,
    content_info: Dict,
    llm_fn: LLMFn,
    options: Optional[Options] = None,
) -> List[Dict]:
    """Rank the candidates of a chunked (long) video against each other.

    Chunks are scored independently, so their scores are not comparable — five
    chunks can each hand out a 95 and the final top-N is then arbitrary. One more
    LLM call sees every candidate (with its transcript excerpt and signals) side by side and
    returns a single, calibrated ranking. On any failure the chunk scores are
    kept, so this step can never lose the earlier work.
    """
    options = options or Options()
    options.resolve_signals()
    if len(highlights) <= num_clips:
        return blend_signals(highlights, options.signals)

    cands = sorted(highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:MAX_RERANK_CANDIDATES]
    cands = sorted(cands, key=lambda h: float(h["start_time"]))     # chronological ids
    names = list(options.signals)
    lines = []
    sig_values: List[Dict[str, float]] = []
    for i, h in enumerate(cands):
        s, e = float(h["start_time"]), float(h["end_time"])
        vals = {n: options.signals[n](s, e) for n in names}
        sig_values.append(vals)
        sig_txt = "".join(f" | {SIGNAL_INFO[n][0]} {v:.0f}" for n, v in vals.items())
        lines.append(
            f"[id {i}] {_clock(s)}-{_clock(e)} ({e - s:.0f}s) | chunk score {h.get('score')}{sig_txt}\n"
            f"title: {h.get('title', '')}\n"
            f"hook: {h.get('hook_sentence', '')}\n"
            f"why: {h.get('virality_reason', '')}\n"
            f"transcript: {_excerpt(transcript, s, e)}"
        )
    duration = float(transcript.get("duration", 0) or 0)
    prompt = RERANK_PROMPT.format(
        n=len(cands),
        k=num_clips,
        m=min(len(cands), max(num_clips * 3, 8)),
        content_type=content_info.get("content_type", "other"),
        minutes=max(1, int(round(duration / 60))),
        focus_note=options.focus_note(),
        virality_criteria=VIRALITY_CRITERIA,
        signal_notes="\n".join(SIGNAL_INFO[n][1] for n in names if n in SIGNAL_INFO),
        candidates="\n\n".join(lines),
    )

    from .local.llm import LLMError          # imported here: highlights.py stays importable without the local package

    ranking: List[Dict] = []
    last_error = "unknown"
    need = min(num_clips, len(cands))
    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        text = ""
        try:
            text = llm_fn(prompt)
            parsed = _parse_json_loose(text)
            ranking = _ranking_entries(parsed.get("ranking") if isinstance(parsed, dict) else parsed, len(cands))
            if len(ranking) < need:
                last_error = "ranking had too few valid entries"
        except LLMError:
            raise                          # the AI service is down or busy: do not quietly settle for the unreliable chunk scores
        except Exception as e:  # noqa: BLE001 — a damaged answer must not lose the chunk results
            last_error = str(e)
            ranking = salvage_ranking(text, len(cands))
            if len(ranking) >= need:
                print(f"[highlights] re-rank answer was not valid JSON; recovered {len(ranking)} scores from it", flush=True)
        if len(ranking) >= need:
            break
        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(f"[highlights] global re-rank invalid on attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS}; retrying", flush=True)

    if len(ranking) < min(num_clips, len(cands)):
        print(f"[highlights] global re-rank failed ({last_error[:120]}); keeping per-chunk scores", flush=True)
        return blend_signals(highlights, options.signals)

    weights = signal_weights(names)
    for r in ranking:
        r["llm_score"] = r["score"]
        vals = sig_values[r["id"]]
        if vals:      # the LLM tends to ignore the hints, so blend the signals in as well
            w_total = sum(weights.values())
            r["score"] = int(round((1 - w_total) * r["llm_score"] + sum(weights[n] * vals[n] for n in vals)))
    ranking.sort(key=lambda r: r["score"], reverse=True)
    floor = ranking[-1]["score"]
    out: List[Dict] = []
    ranked_ids = set()
    for r in ranking:
        ranked_ids.add(r["id"])
        h = dict(cands[r["id"]])
        h.update({"original_score": h.get("score"), "score": r["score"], "llm_score": r["llm_score"],
                  "rerank_reason": r["reason"]})
        if sig_values[r["id"]]:
            h["signals"] = {n: round(v) for n, v in sig_values[r["id"]].items()}
        out.append(h)
    # candidates the editor did not shortlist (or that fell outside the top 40) rank below every shortlisted one
    shortlisted = [cands[i] for i in ranked_ids]
    for h in sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True):
        if any(h is c for c in shortlisted):
            continue
        h = dict(h)
        h.update({"original_score": h.get("score"), "score": max(0, min(int(h.get("score", 0)), floor - 1))})
        out.append(h)
    print(
        "[highlights] global re-rank: top = "
        + ", ".join(f"{_clock(o['start_time'])} ({o['score']})" for o in out[: max(num_clips, 3)]),
        flush=True,
    )
    return out


def snap_to_sentences(highlights: List[Dict], transcript: Dict) -> List[Dict]:
    """Move clip edges onto transcript segment boundaries so clips don't start or end mid-sentence.

    The LLM's timestamps drift by a few seconds. Edges only move when a boundary is
    within SNAP_TOLERANCE_SECONDS; the start gets 0.2s of lead-in and the end 0.4s of tail.
    """
    segs = transcript.get("segments", [])
    if not segs:
        return highlights
    duration = float(transcript.get("duration", 0) or segs[-1]["end"])
    out: List[Dict] = []
    for h in highlights:
        start, end = float(h["start_time"]), float(h["end_time"])
        for s in segs:                                   # segment containing the start
            if s["start"] <= start + 0.5 < s["end"] and start - s["start"] <= SNAP_TOLERANCE_SECONDS:
                start = s["start"]
                break
        for s in segs:                                   # first segment ending at/after the end
            if s["end"] >= end - 0.5:
                if s["end"] - end <= SNAP_TOLERANCE_SECONDS and s["start"] < end:
                    end = s["end"]
                break
        start, end = max(0.0, start - 0.2), min(duration, end + 0.4)
        if end > start:
            out.append({**h, "start_time": start, "end_time": end})
        else:
            out.append(h)
    return out


def enforce_duration(highlights: List[Dict], transcript: Dict, min_s: float, max_s: float) -> List[Dict]:
    """Make every clip fit the requested length range, ending on a sentence boundary where possible.

    The prompt asks for the range, but models drift: a clip that is too long is cut back to the last
    sentence end inside the limit; one that is too short is extended to the first sentence end that
    reaches the minimum (bounded by the maximum).
    """
    segs = transcript.get("segments", [])
    duration = float(transcript.get("duration", 0) or (segs[-1]["end"] if segs else 0))
    out: List[Dict] = []
    for h in highlights:
        start, end = float(h["start_time"]), float(h["end_time"])
        length = end - start
        if length > max_s:
            ends = [s["end"] for s in segs if start + min_s <= s["end"] <= start + max_s]
            end = max(ends) if ends else start + max_s
        elif length < min_s:
            ends = [s["end"] for s in segs if start + min_s <= s["end"] <= start + max_s]
            end = min(ends) if ends else start + min_s
        if duration:
            end = min(end, duration)
        out.append({**h, "end_time": end} if end > start else h)
    return out


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
    options: Optional[Options] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable. `options` carries the editor's focus,
    clip-length range and ranking signals.
    """
    llm_fn = llm_fn or call_muapi_llm
    options = options or Options()
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        unreadable: List[str] = []
        for i, chunk in enumerate(chunks):
            from .local import control   # cancel / progress (no-ops outside the UI)

            control.check()
            control.report("rank", i / max(1, len(chunks)), f"reading part {i + 1} of {len(chunks)}")
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            result = call_highlight_api(text, content_info, chunk["duration"], num_clips=num_clips, is_chunk=True,
                                        llm_fn=llm_fn, options=options)
            if result.get("failed"):
                unreadable.append(result["failed"])
                print(f"[highlights] part {i + 1} could not be read; going on without it", flush=True)
            elif not result.get("highlights"):
                print(f"[highlights] part {i + 1}: nothing worth a clip", flush=True)
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        if unreadable and (len(unreadable) * 2 > len(chunks) or not all_highlights):
            raise RuntimeError(unreadable[-1])               # most of the video unread: that is a fault, say so
        highlights = dedupe_highlights(all_highlights)
        # Chunk scores aren't comparable across chunks: pick the best from the WHOLE video.
        highlights = rerank_highlights(highlights, transcript, num_clips, content_info, llm_fn, options)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn, options=options)
        highlights = blend_signals(dedupe_highlights(result.get("highlights", [])), options.resolve_signals())

    highlights = snap_to_sentences(highlights, transcript)
    highlights = enforce_duration(highlights, transcript, options.min_seconds, options.max_seconds)
    return {"highlights": highlights}
