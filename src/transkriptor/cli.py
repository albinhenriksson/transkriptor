from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from .config import apply_overrides, load_settings
from .logging_setup import setup_logging
from .media import discover_media
from .probe import ffprobe_info
from .utils import (
    TranskriptorError,
    detect_gpu_ids,
    normalize_path,
    require_bins,
)

app = typer.Typer(
    add_completion=True, help="GPU-parallel media transcription (Whisper) -> SRT/VTT/JSON/TXT."
)
console = Console()
log = logging.getLogger(__name__)

BANNER = r"""
┏┳┓┏┓┏┓┏┓┏┓┓┏┏┓┏┓┳┓
 ┃ ┣┫┃┓┗┓┃ ┣┫┣┫┃┃┣┫
 ┻ ┛┗┗┛┗┛┗┛┛┗┛┗┗┛┗┛
"""

# -----------------------------------------------------------------------------
# Typer parameter specs (module-level singletons to satisfy Ruff B008)
# -----------------------------------------------------------------------------

ARG_PATH = typer.Argument(Path("."), help="File or directory to process.")

OPT_CONFIG = typer.Option(None, "--config", "-c", help="INI file (highest priority).")
OPT_NOCLI = typer.Option(False, "--nocli", help="Batch mode (no interactive UI).")
OPT_VERBOSITY = typer.Option(None, "--verbosity", "-v", help="Override verbosity 0..2.")

OPT_MODEL = typer.Option(None, "--model", help="Whisper model (e.g. medium, large-v3).")
OPT_LANGUAGE = typer.Option(None, "--language", help="Force language code (e.g. sv, en).")
OPT_COMPUTE_TYPE = typer.Option(None, "--compute-type", help="CTranslate2 compute type.")
OPT_CHUNK_SECONDS = typer.Option(None, "--chunk-seconds", help="FFmpeg chunk length in seconds.")

OPT_GPUS = typer.Option(None, "--gpus", help='GPU list like "0,1,2". Empty = auto-detect.')
OPT_JOBS = typer.Option(None, "--jobs", help="Parallel workers. 0 = auto (usually #GPUs).")

OPT_VAD = typer.Option(True, "--vad/--no-vad", help="Enable/disable VAD filter.")
OPT_BEAM_SIZE = typer.Option(None, "--beam-size", help="Beam size (>=1).")

OPT_WRITE_SRT = typer.Option(None, "--write-srt/--no-srt", help="Write SRT output.")
OPT_WRITE_VTT = typer.Option(None, "--write-vtt/--no-vtt", help="Write VTT output.")
OPT_WRITE_JSON = typer.Option(None, "--write-json/--no-json", help="Write JSON output.")
OPT_WRITE_TXT = typer.Option(None, "--write-txt/--no-txt", help="Write TXT output.")

OPT_CLEANUP = typer.Option(None, "--cleanup", help="Cleanup policy: none|json|all.")


def _parse_csv_ids(s: str) -> list[str]:
    ids = [x.strip() for x in s.split(",") if x.strip()]
    for x in ids:
        if not x.isdigit():
            raise TranskriptorError(
                f"Invalid GPU id '{x}'. Expected comma-separated integers like 0,1,2."
            )
    return ids


@app.command()
def run(
    path: Path = ARG_PATH,
    config: Path | None = OPT_CONFIG,
    nocli: bool = OPT_NOCLI,
    verbosity: int | None = OPT_VERBOSITY,
    model: str | None = OPT_MODEL,
    language: str | None = OPT_LANGUAGE,
    compute_type: str | None = OPT_COMPUTE_TYPE,
    chunk_seconds: int | None = OPT_CHUNK_SECONDS,
    gpus: str | None = OPT_GPUS,
    jobs: int | None = OPT_JOBS,
    vad: bool = OPT_VAD,
    beam_size: int | None = OPT_BEAM_SIZE,
    write_srt: bool | None = OPT_WRITE_SRT,
    write_vtt: bool | None = OPT_WRITE_VTT,
    write_json: bool | None = OPT_WRITE_JSON,
    write_txt: bool | None = OPT_WRITE_TXT,
    cleanup: str | None = OPT_CLEANUP,
) -> None:
    """
    Run a transcription job.

    Pipeline implementation lands in milestones; this command already validates config,
    resolves GPU IDs, and prints a deterministic execution plan.
    """
    try:
        require_bins("ffmpeg", "ffprobe", "nvidia-smi")

        settings = load_settings(config)
        settings = apply_overrides(
            settings,
            verbosity=verbosity,
            model=model,
            language=language,
            compute_type=compute_type,
            chunk_seconds=chunk_seconds,
            gpus=gpus,
            jobs=jobs,
            vad=vad,
            beam_size=beam_size,
            write_srt=write_srt,
            write_vtt=write_vtt,
            write_json=write_json,
            write_txt=write_txt,
            cleanup=cleanup,
        )

        setup_logging(settings.verbosity)

        p = normalize_path(path)
        if not p.exists():
            raise TranskriptorError(f"Path does not exist: {p}")

        console.print(
            Panel.fit(BANNER, title="transkriptor", subtitle="GPU-first transcription pipeline")
        )

        gpu_ids = _parse_csv_ids(settings.gpus) if settings.gpus else detect_gpu_ids()
        if not gpu_ids:
            raise TranskriptorError("No GPUs detected. Check NVIDIA driver / nvidia-smi.")

        workers = settings.jobs if settings.jobs and settings.jobs > 0 else len(gpu_ids)

        log.info("Mode: %s", "batch (--nocli)" if nocli else "interactive (not implemented yet)")
        log.info("Target: %s", p)
        log.info("Model: %s", settings.model)
        log.info("Language: %s", settings.language or "(auto)")
        log.info("Compute type: %s", settings.compute_type or "(auto)")
        log.info("Chunk seconds: %s", settings.chunk_seconds)
        log.info("GPUs: %s", ",".join(gpu_ids))
        log.info("Workers: %s", workers)
        log.info(
            "Outputs: srt=%s vtt=%s json=%s txt=%s",
            settings.write_srt,
            settings.write_vtt,
            settings.write_json,
            settings.write_txt,
        )
        log.info("Cleanup: %s", settings.cleanup)

        media = discover_media(p, settings.exclude)
        if not media:
            raise TranskriptorError("No files found (after filtering).")

        log.info("Discovered files (candidates): %d", len(media))

        valid = 0
        bad = 0
        unprobeable = 0
        total_dur = 0.0

        for m in media:
            info = ffprobe_info(m.path)

            if not info.ok:
                unprobeable += 1
                continue

            if not info.has_audio or info.duration_s <= 0:
                bad += 1
                continue

            valid += 1
            total_dur += info.duration_s

        if unprobeable:
            log.info("Skipped %d files (ffprobe couldn't read).", unprobeable)
        if bad:
            log.info("Skipped %d files (no audio stream or zero duration).", bad)

        if valid == 0:
            raise TranskriptorError("No valid media-with-audio files found.")

        log.info("Valid media files: %d", valid)
        log.info("Total duration (valid): %.1f minutes", total_dur / 60.0)

        log.info(
            "Status: discovery + ffprobe OK. Next milestone is chunking + multi-GPU transcription."
        )

    except TranskriptorError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=2) from None


@app.command(name="show-config")
def show_config(
    config: Path | None = OPT_CONFIG,
) -> None:
    """Print the effective configuration after reading INI files."""
    try:
        s = load_settings(config)
        console.print("[bold]Effective settings:[/bold]")
        console.print(f"verbosity = {s.verbosity}")
        console.print(f"root = {s.root}")
        console.print(f"exclude = {s.exclude}")
        console.print(f"workdir_name = {s.workdir_name}")
        console.print(f"chunk_seconds = {s.chunk_seconds}")
        console.print(f"sample_rate = {s.sample_rate}")
        console.print(f"channels = {s.channels}")
        console.print(f"model = {s.model}")
        console.print(f"device = {s.device}")
        console.print(f"compute_type = {s.compute_type}")
        console.print(f"language = {s.language}")
        console.print(f"vad = {s.vad}")
        console.print(f"beam_size = {s.beam_size}")
        console.print(f"gpus = {s.gpus}")
        console.print(f"jobs = {s.jobs}")
        console.print(f"write_srt = {s.write_srt}")
        console.print(f"write_vtt = {s.write_vtt}")
        console.print(f"write_json = {s.write_json}")
        console.print(f"write_txt = {s.write_txt}")
        console.print(f"cleanup = {s.cleanup}")

    except TranskriptorError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=2) from None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
