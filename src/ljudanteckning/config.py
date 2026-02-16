from __future__ import annotations

import configparser
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from platformdirs import user_config_dir

from .utils import LjudanteckningError, normalize_path


@dataclass(frozen=True)
class Settings:
    # general
    verbosity: int = 1
    root: Path = Path(".")
    exclude: list[str] = field(default_factory=list)

    # temp
    workdir_name: str = ".ljudanteckning"

    # ffmpeg
    chunk_seconds: int = 600
    sample_rate: int = 16000
    channels: int = 1

    # whisper
    model: str = "medium"
    device: str = "cuda"  # GPU-first
    compute_type: str | None = None
    language: str | None = None
    vad: bool = True
    beam_size: int = 5

    # gpu scheduling
    gpus: str | None = None  # "0,1,2"
    jobs: int = 0  # 0 => auto

    # output
    write_srt: bool = True
    write_vtt: bool = True
    write_json: bool = True
    write_txt: bool = True
    cleanup: str = "json"  # none|json|all


def default_config_paths() -> list[Path]:
    """
    INI search order (low -> high priority):
      1) ~/.config/ljudanteckning/ljudanteckning.ini
      2) ./ljudanteckning.ini
    """
    xdg = Path(user_config_dir("ljudanteckning")) / "ljudanteckning.ini"
    local = Path.cwd() / "ljudanteckning.ini"
    return [xdg, local]


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _read_many(cp: configparser.ConfigParser, paths: Iterable[Path]) -> None:
    for p in paths:
        if p.exists():
            cp.read(p, encoding="utf-8")


def load_settings(explicit_config: Path | None) -> Settings:
    """
    Load Settings from INI files. Later files override earlier ones.
    """
    cp = configparser.ConfigParser()

    paths = default_config_paths()
    if explicit_config is not None:
        paths = [*paths, explicit_config]

    _read_many(cp, paths)

    def get(section: str, key: str, fallback=None):
        return cp.get(section, key, fallback=fallback) if cp.has_section(section) else fallback

    def getint(section: str, key: str, fallback: int):
        return cp.getint(section, key, fallback=fallback) if cp.has_section(section) else fallback

    def getbool(section: str, key: str, fallback: bool):
        return (
            cp.getboolean(section, key, fallback=fallback) if cp.has_section(section) else fallback
        )

    exclude = _split_csv(get("ljudanteckning", "exclude", ""))

    s = Settings(
        verbosity=getint("ljudanteckning", "verbosity", 1),
        root=normalize_path(Path(get("ljudanteckning", "root", "."))),
        exclude=exclude,
        workdir_name=get("temp", "workdir_name", ".ljudanteckning"),
        chunk_seconds=getint("ffmpeg", "chunk_seconds", 600),
        sample_rate=getint("ffmpeg", "sample_rate", 16000),
        channels=getint("ffmpeg", "channels", 1),
        model=get("whisper", "model", "medium"),
        device=get("whisper", "device", "cuda"),
        compute_type=(lambda x: x if x else None)(get("whisper", "compute_type", "")),
        language=(lambda x: x if x else None)(get("whisper", "language", "")),
        vad=getbool("whisper", "vad", True),
        beam_size=getint("whisper", "beam_size", 5),
        gpus=(lambda x: x if x else None)(get("gpu", "gpus", "")),
        jobs=getint("gpu", "jobs", 0),
        write_srt=getbool("output", "write_srt", True),
        write_vtt=getbool("output", "write_vtt", True),
        write_json=getbool("output", "write_json", True),
        write_txt=getbool("output", "write_txt", True),
        cleanup=get("output", "cleanup", "json"),
    )

    validate_settings(s)
    return s


def apply_overrides(s: Settings, **overrides) -> Settings:
    """
    Apply CLI overrides cleanly. Pass only keys with non-None values.
    """
    clean = {k: v for k, v in overrides.items() if v is not None}
    s2 = replace(s, **clean)
    validate_settings(s2)
    return s2


def validate_settings(s: Settings) -> None:
    if s.verbosity < 0 or s.verbosity > 2:
        raise LjudanteckningError("verbosity must be 0..2")

    if s.chunk_seconds <= 0:
        raise LjudanteckningError("ffmpeg.chunk_seconds must be > 0")

    if s.sample_rate not in (8000, 12000, 16000, 22050, 24000, 44100, 48000):
        # keep it sane (whisper usually likes 16k)
        raise LjudanteckningError("ffmpeg.sample_rate looks invalid/unexpected")

    if s.channels not in (1, 2):
        raise LjudanteckningError("ffmpeg.channels must be 1 or 2")

    if s.device != "cuda":
        raise LjudanteckningError("This project is GPU-first: whisper.device must be 'cuda'")

    if s.beam_size < 1:
        raise LjudanteckningError("whisper.beam_size must be >= 1")

    if s.cleanup not in ("none", "json", "all"):
        raise LjudanteckningError("output.cleanup must be one of: none|json|all")
