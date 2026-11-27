"""Cooperative cancellation and structured progress, shared by every long-running step.

The web UI runs one job at a time. Before it starts a job it calls ``bind()``; the pipeline
then calls ``check()`` at safe points (between clips, inside the transcription / render loops)
and ``report()`` to say what it is doing. When nothing is bound (CLI, tests) both are no-ops, so
library code never needs to know whether a UI exists.
"""
import contextlib
import threading
from typing import Callable, Iterator, Optional, Tuple

ProgressSink = Callable[[str, Optional[float], str], None]


class Cancelled(BaseException):
    """Raised by check() when the user pressed Cancel.

    A BaseException on purpose: the pipeline has plenty of best-effort ``except Exception`` blocks, and
    none of them may swallow a cancel.
    """


_lock = threading.Lock()
_cancel: Optional[threading.Event] = None
_sink: Optional[ProgressSink] = None
_scale: Tuple[float, float] = (0.0, 1.0)        # (base, span): where the current step sits inside the whole job


def bind(cancel_event: Optional[threading.Event], sink: Optional[ProgressSink]) -> None:
    global _cancel, _sink, _scale
    with _lock:
        _cancel, _sink, _scale = cancel_event, sink, (0.0, 1.0)


def unbind() -> None:
    bind(None, None)


def check() -> None:
    """Raise Cancelled if the current job was cancelled."""
    ev = _cancel
    if ev is not None and ev.is_set():
        raise Cancelled("Cancelled")


def cancelled() -> bool:
    ev = _cancel
    return ev is not None and ev.is_set()


@contextlib.contextmanager
def scale(base: float, span: float) -> Iterator[None]:
    """Inside the block, report(fraction) means ``base + span * fraction`` of the whole job.

    Lets "clip 2 of 3" be one third of the render stage without the clip renderer knowing about it.
    """
    global _scale
    prev_base, prev_span = _scale
    _scale = (prev_base + prev_span * base, prev_span * span)
    try:
        yield
    finally:
        _scale = (prev_base, prev_span)


def report(stage: str, fraction: Optional[float] = None, note: str = "") -> None:
    """Tell whoever is listening which stage we are in (fraction 0..1, or None if unknown)."""
    sink = _sink
    if sink is not None:
        if fraction is not None:
            fraction = _scale[0] + _scale[1] * max(0.0, min(1.0, fraction))
        try:
            sink(stage, fraction, note)
        except Exception:
            pass        # progress is best-effort, it must never break the work
