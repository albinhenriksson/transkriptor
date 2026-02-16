from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import time

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .utils import LjudanteckningError, normalize_path, require_bins

__all__ = ["ChunkSpec", "workdir_for", "split_to_chunks", "cleanup_workdir"]


@dataclass(frozen=True)
class ChunkSpec:
    chunk_seconds: int
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class ChunkCacheMeta:
    version: int
    input_path: str
    input_size: int
    input_mtime_ns: int
    chunk_seconds: int
    sample_rate: int
    channels: int
    created_at: float


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def workdir_for(media_path: Path, workdir_name: str) -> Path:
    """
    Stable per-file workdir next to the media file:
      <media_dir>/.ljudanteckning/<stem>__<ext>__<hash>/
    """
    media_path = normalize_path(media_path)
    base = media_path.parent / workdir_name
    base.mkdir(parents=True, exist_ok=True)

    stem = media_path.stem[:80].strip() or "file"
    ext = (media_path.suffix.lstrip(".") or "noext")[:16]
    h = _short_hash(str(media_path))

    d = base / f"{stem}__{ext}__{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_path(workdir: Path) -> Path:
    return workdir / "chunk_meta.json"


def _read_meta(workdir: Path) -> ChunkCacheMeta | None:
    p = _meta_path(workdir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ChunkCacheMeta(**data)
    except Exception:
        return None


def _write_meta(workdir: Path, meta: ChunkCacheMeta) -> None:
    _meta_path(workdir).write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")


def _existing_chunks(workdir: Path) -> list[Path]:
    return sorted(workdir.glob("chunk_*.wav"))


def _parse_out_time_to_seconds(s: str) -> float:
    # "HH:MM:SS.micro"
    hms, _, frac = s.partition(".")
    h, m, sec = hms.split(":")
    out = int(h) * 3600 + int(m) * 60 + int(sec)
    if frac:
        frac6 = frac[:6].ljust(6, "0")
        out += int(frac6) / 1_000_000
    return float(out)


def split_to_chunks(
    media_path: Path,
    workdir_name: str,
    spec: ChunkSpec,
    duration_s: float,
    force: bool,
) -> list[Path]:
    """
    Split input media into WAV chunks using ffmpeg segment muxer.
    Returns chunk paths.
    """
    require_bins("ffmpeg", "ffprobe")

    media_path = normalize_path(media_path)
    wd = workdir_for(media_path, workdir_name)

    # Cache validation
    st = media_path.stat()
    meta = _read_meta(wd)
    existing = _existing_chunks(wd)

    want = ChunkCacheMeta(
        version=1,
        input_path=str(media_path),
        input_size=st.st_size,
        input_mtime_ns=st.st_mtime_ns,
        chunk_seconds=spec.chunk_seconds,
        sample_rate=spec.sample_rate,
        channels=spec.channels,
        created_at=time(),
    )

    cache_ok = (
        (meta is not None)
        and (meta.version == want.version)
        and (meta.input_path == want.input_path)
        and (meta.input_size == want.input_size)
        and (meta.input_mtime_ns == want.input_mtime_ns)
        and (meta.chunk_seconds == want.chunk_seconds)
        and (meta.sample_rate == want.sample_rate)
        and (meta.channels == want.channels)
        and bool(existing)
    )

    if cache_ok and not force:
        return existing

    # Remove old chunks if forcing or cache mismatch
    for p in wd.glob("chunk_*.wav"):
        p.unlink(missing_ok=True)

    out_pattern = wd / "chunk_%05d.wav"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        str(spec.channels),
        "-ar",
        str(spec.sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(spec.chunk_seconds),
        "-reset_timestamps",
        "1",
        "-progress",
        "pipe:1",
        "-loglevel",
        "error",
        str(out_pattern),
    ]

    columns = [
        TextColumn("[bold]chunking[/bold]"),
        TextColumn("{task.fields[name]}", justify="left"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[speed]}", justify="right"),
    ]

    with Progress(*columns, transient=True) as progress:
        task = progress.add_task(
            "chunk",
            total=max(duration_s, 0.001),
            name=media_path.name,
            speed="",
        )

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        out_time_s = 0.0
        assert p.stdout is not None

        for line in p.stdout:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)

            if k == "out_time":
                try:
                    out_time_s = _parse_out_time_to_seconds(v)
                except Exception:
                    continue
                progress.update(task, completed=min(out_time_s, duration_s))

            elif k == "speed":
                sp = v.strip()
                if sp:
                    progress.update(task, speed=f"[dim]{sp}[/dim]")

            elif k == "progress" and v.strip() == "end":
                progress.update(task, completed=duration_s)
                break

        stderr = ""
        if p.stderr is not None:
            stderr = p.stderr.read()

        rc = p.wait()
        if rc != 0:
            msg = stderr.strip() or "ffmpeg failed (no stderr captured)"
            raise LjudanteckningError(f"ffmpeg chunking failed for {media_path}:\n{msg}")

    chunks = _existing_chunks(wd)
    if not chunks:
        raise LjudanteckningError(f"No chunks produced for {media_path} (workdir: {wd})")

    _write_meta(wd, want)
    return chunks


def cleanup_workdir(workdir: Path, policy: str) -> None:
    """
    policy:
      - "none": keep chunks + per-chunk json
      - "json": delete per-chunk json only
      - "all": delete chunks + per-chunk json
    """
    if policy == "none":
        return

    if policy in ("json", "all"):
        for p in workdir.glob("chunk_*.whisper.json"):
            p.unlink(missing_ok=True)

    if policy == "all":
        for p in workdir.glob("chunk_*.wav"):
            p.unlink(missing_ok=True)
