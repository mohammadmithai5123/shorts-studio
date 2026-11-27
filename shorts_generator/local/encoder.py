"""Which H.264 encoder to use: the graphics chip when it works (much faster), otherwise x264 on the CPU.

``LOCAL_ENCODER`` = auto (default) | cpu | qsv | nvenc | amf. "auto" tries the hardware encoders once per process with a
tiny test encode, because being listed by ffmpeg does not mean the driver can do it. If a real render fails on a hardware
encoder, :func:`disable_hardware` switches the rest of the process to x264 and the render is repeated.
"""
import subprocess
import threading
from typing import Dict, List, Optional

from .. import config
from ..logsetup import get as _get_logger

log = _get_logger("encoder")

# pix_fmt is what the encoder wants; the pipe always carries yuv420p and ffmpeg converts where needed.
# The bitrate cap matters: the picture is upscaled, so a higher rate only makes files huge (busy game footage hit 12+ Mbps).
ENCODERS: Dict[str, Dict] = {
    "qsv": {"codec": "h264_qsv", "label": "Intel Quick Sync (graphics chip)", "pix": "nv12",
            "args": ["-preset", "faster", "-global_quality", "23", "-look_ahead", "0", "-maxrate", "10M", "-bufsize", "20M"]},
    "nvenc": {"codec": "h264_nvenc", "label": "NVIDIA NVENC (graphics card)", "pix": "yuv420p",
              "args": ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0", "-maxrate", "10M", "-bufsize", "20M"]},
    "amf": {"codec": "h264_amf", "label": "AMD AMF (graphics card)", "pix": "yuv420p",
            "args": ["-quality", "balanced", "-rc", "cqp", "-qp_i", "22", "-qp_p", "22"]},
    "cpu": {"codec": "libx264", "label": "x264 on the processor", "pix": "yuv420p",
            "args": ["-preset", "veryfast", "-crf", "21", "-maxrate", "8M", "-bufsize", "16M"]},
}
AUTO_ORDER = ("nvenc", "qsv", "amf")

_LOCK = threading.Lock()
_chosen: Optional[str] = None
_hardware_off = False


def _works(name: str) -> bool:
    enc = ENCODERS[name]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=720x1280:r=30", "-frames:v", "5",
           "-c:v", enc["codec"], *enc["args"], "-pix_fmt", enc["pix"], "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def choose() -> str:
    """The encoder key to use for the next render."""
    global _chosen
    wanted = (config.env("LOCAL_ENCODER") or "auto").lower()
    if _hardware_off or wanted == "cpu":
        return "cpu"
    if wanted in ENCODERS:
        return wanted
    with _LOCK:
        if _chosen is None:
            _chosen = next((n for n in AUTO_ORDER if _works(n)), "cpu")
            log.info("video encoder: %s", ENCODERS[_chosen]["label"])
        return _chosen


def video_args(name: str) -> List[str]:
    enc = ENCODERS[name]
    return ["-c:v", enc["codec"], *enc["args"], "-pix_fmt", enc["pix"]]


def disable_hardware(reason: str) -> None:
    global _hardware_off
    _hardware_off = True
    log.warning("hardware encoding switched off for this session: %s", reason[:300])


def reset() -> None:                         # for tests
    global _chosen, _hardware_off
    _chosen, _hardware_off = None, False


def describe() -> str:
    return ENCODERS[choose()]["label"]
