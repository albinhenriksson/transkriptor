from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class LjudanteckningError(RuntimeError):
    pass


def which(bin_name: str) -> str | None:
    return shutil.which(bin_name)


def require_bins(*bins: str) -> None:
    missing = [b for b in bins if which(b) is None]
    if missing:
        raise LjudanteckningError(f"Missing required binaries in PATH: {', '.join(missing)}")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def detect_gpu_ids() -> list[str]:
    """
    Detect NVIDIA GPU indices using `nvidia-smi -L`.
    Returns ["0","1",...].
    """
    require_bins("nvidia-smi")
    p = run(["nvidia-smi", "-L"])
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    # Each GPU usually prints as: "GPU 0: ...", "GPU 1: ..."
    ids: list[str] = []
    for ln in lines:
        if ln.startswith("GPU "):
            # GPU 0: ...
            n = ln.split(":", 1)[0].split()[1]
            if n.isdigit():
                ids.append(n)
    return ids


def normalize_path(p: Path) -> Path:
    return p.expanduser().resolve()
