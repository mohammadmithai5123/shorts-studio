"""Whisper transcription (faster-whisper) — the FALLBACK for videos that have no captions.

faster-whisper is an optional dependency (``pip install faster-whisper``): most videos never need it
because their captions are fetched instead (see transcript.py). Caching is done by the caller.
"""
import time
from typing import Dict, Optional

from ..config import (
    LOCAL_WHISPER_BEAM_SIZE,
    LOCAL_WHISPER_DEVICE,
    LOCAL_WHISPER_THREADS,
    LOCAL_WHISPER_VAD_FILTER,
    LOCAL_WHISPER_VAD_PARAMETERS,
    whisper_model,
)
from . import control


def _clock(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _media_duration(path: str) -> float:
    """Container duration in seconds (0.0 if unknown). Uses PyAV, a faster-whisper dependency."""
    try:
        import av  # type: ignore

        with av.open(path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
    except Exception:
        pass
    return 0.0


def _pick_beam(setting: str, duration: float) -> int:
    """'auto' -> beam 5 up to 30 min of audio, greedy (1) beyond; otherwise the number given."""
    if setting not in ("", "auto"):
        try:
            return max(1, int(setting))
        except ValueError:
            pass
    return 5 if duration <= 1800 else 1


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")      # catches missing cuBLAS/cuDNN libs
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Run faster-whisper on a local audio/video file -> {duration, language, segments}."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "This video has no usable captions, so its speech has to be transcribed with Whisper, which is "
            "not installed. Install it with:  pip install faster-whisper"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    total = _media_duration(media_path)
    beam = _pick_beam(LOCAL_WHISPER_BEAM_SIZE, total)
    print(
        f"[transcribe/local] faster-whisper model={whisper_model()} device={device} beam={beam} "
        f"threads={LOCAL_WHISPER_THREADS or 'default'} audio={_clock(total)}",
        flush=True,
    )
    # cpu_threads: 0 lets CTranslate2 choose (4). Do not default to os.cpu_count(): hyperthreads fight over cores.
    model = WhisperModel(whisper_model(), device=device, compute_type=compute_type, cpu_threads=LOCAL_WHISPER_THREADS)

    kwargs = {"audio": media_path, "language": language, "beam_size": beam, "condition_on_previous_text": False,
              "vad_filter": bool(LOCAL_WHISPER_VAD_FILTER)}
    if LOCAL_WHISPER_VAD_FILTER:
        kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    segments_iter, info = model.transcribe(**kwargs)
    total = total or float(getattr(info, "duration", 0.0))

    segments = []
    started = last_report = time.time()
    for s in segments_iter:
        control.check()
        segments.append({"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()})
        now = time.time()
        if total > 0 and now - last_report >= 5:
            last_report = now
            done = min(float(s.end), total)
            elapsed = now - started
            eta = elapsed * (total - done) / done if done > 0 else 0
            control.report("transcribe", done / total, f"{_clock(done)} / {_clock(total)} · about {_clock(eta)} left")
            print(f"[transcribe/local] {done / total:5.1%}  {_clock(done)} / {_clock(total)}  "
                  f"elapsed {_clock(elapsed)}  eta ~{_clock(eta)}", flush=True)

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    return {"duration": duration, "language": getattr(info, "language", language) or "", "segments": segments}
