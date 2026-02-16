from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .utils import LjudanteckningError, run


@dataclass(frozen=True)
class ProbeInfo:
    duration_s: float
    has_audio: bool
    ok: bool  # ffprobe succeeded


def ffprobe_info(path: Path) -> ProbeInfo:
    """
    Probe media using ffprobe:
      - duration (seconds)
      - whether at least one audio stream exists

    If ffprobe fails, return ok=False (caller decides what to do).
    """
    try:
        p = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
    except subprocess.CalledProcessError:
        return ProbeInfo(duration_s=0.0, has_audio=False, ok=False)

    try:
        data = json.loads(p.stdout)
    except Exception as e:
        raise LjudanteckningError(f"ffprobe returned invalid JSON for {path}: {e!r}") from None

    fmt = data.get("format", {}) or {}
    dur_raw = fmt.get("duration", "0")
    try:
        duration = float(dur_raw)
    except Exception:
        duration = 0.0

    streams = data.get("streams", []) or []
    has_audio = any((s.get("codec_type") == "audio") for s in streams)

    return ProbeInfo(duration_s=max(0.0, duration), has_audio=has_audio, ok=True)
