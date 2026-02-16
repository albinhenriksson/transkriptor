from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from .chunking import ChunkSpec, cleanup_workdir, split_to_chunks, workdir_for
from .config import apply_overrides, load_settings
from .export import (
    detect_language_from_chunks,
    merge_chunks,
)
from .export import (
    write_srt as export_srt,
)
from .export import (
    write_txt as export_txt,
)
from .export import (
    write_vtt as export_vtt,
)
from .export import (
    write_whisper_json as export_whisper_json,
)
from .logging_setup import setup_logging
from .media import discover_media
from .probe import ffprobe_info
from .transcribe import TranscribeTask, transcribe_tasks
from .utils import (
    LjudanteckningError,
    detect_gpu_ids,
    normalize_path,
    require_bins,
)

app = typer.Typer(
    add_completion=True, help="GPU-parallel media transcription (Whisper) -> SRT/VTT/JSON/TXT."
)
console = Console()
log = logging.getLogger(__name__)

TITLE = "ljudanteckning"
TAGLINE = "Turn audio + video into searchable text."

BANNER = r"""

████████╗██████╗  █████╗ ███╗   ██╗███████╗██╗  ██╗██████╗ ██╗██████╗ ████████╗ ██████╗ ██████╗ 
╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║ ██╔╝██╔══██╗██║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
   ██║   ██████╔╝███████║██╔██╗ ██║███████╗█████╔╝ ██████╔╝██║██████╔╝   ██║   ██║   ██║██████╔╝
   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔═██╗ ██╔══██╗██║██╔═══╝    ██║   ██║   ██║██╔══██╗
   ██║   ██║  ██║██║  ██║██║ ╚████║███████║██║  ██╗██║  ██║██║██║        ██║   ╚██████╔╝██║  ██║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝
""".strip("\n")


def _max_line_len(s: str) -> int:
    lines = s.splitlines() or [""]
    return max(len(line) for line in lines)


def print_banner() -> None:
    # Panel width should fit banner + title/subtitle if possible.
    desired = max(_max_line_len(BANNER), len(TITLE), len(TAGLINE)) + 4  # borders/padding
    width = min(console.size.width, desired)

    # If the terminal is too narrow for the subtitle, drop it instead of truncating.
    subtitle = TAGLINE if width >= (len(TAGLINE) + 4) else None

    console.print(
        Panel(
            BANNER,
            title=TITLE,
            subtitle=subtitle,
            expand=False,
            width=width,
            padding=(0, 1),
            border_style="cyan",
        )
    )

    if subtitle is None:
        console.print(f"[dim]{TAGLINE}[/dim]")


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

OPT_RESPLIT = typer.Option(
    False, "--resplit", help="Recreate WAV chunks even if they already exist."
)

OPT_RETRANSCRIBE = typer.Option(
    False, "--retranscribe", help="Redo chunk transcriptions even if JSON exists."
)

OPT_TELEMETRY = typer.Option(
    True, "--telemetry/--no-telemetry", help="Show GPU telemetry (NVML) during transcription."
)


def _parse_csv_ids(s: str) -> list[str]:
    ids = [x.strip() for x in s.split(",") if x.strip()]
    for x in ids:
        if not x.isdigit():
            raise LjudanteckningError(
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
    resplit: bool = OPT_RESPLIT,
    retranscribe: bool = OPT_RETRANSCRIBE,
    telemetry: bool = OPT_TELEMETRY,
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
            raise LjudanteckningError(f"Path does not exist: {p}")

        print_banner()

        gpu_ids = _parse_csv_ids(settings.gpus) if settings.gpus else detect_gpu_ids()
        if not gpu_ids:
            raise LjudanteckningError("No GPUs detected. Check NVIDIA driver / nvidia-smi.")

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
            raise LjudanteckningError("No files found (after filtering).")

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
            raise LjudanteckningError("No valid media-with-audio files found.")

        log.info("Valid media files: %d", valid)
        log.info("Total duration (valid): %.1f minutes", total_dur / 60.0)
        # Chunking stage
        spec = ChunkSpec(
            chunk_seconds=settings.chunk_seconds,
            sample_rate=settings.sample_rate,
            channels=settings.channels,
        )
        log.info(
            "Chunking stage: %ds chunks @ %d Hz, %d ch",
            spec.chunk_seconds,
            spec.sample_rate,
            spec.channels,
        )

        # Build jobs: per-file workdir + chunks
        jobs_list: list[tuple[Path, Path]] = []  # (media_path, workdir)
        transcribe_q: list[TranscribeTask] = []

        for m in media:
            info = ffprobe_info(m.path)
            if not info.ok or not info.has_audio or info.duration_s <= 0:
                continue

            wd = workdir_for(m.path, settings.workdir_name)
            chunks = split_to_chunks(
                media_path=m.path,
                workdir_name=settings.workdir_name,
                spec=spec,
                duration_s=info.duration_s,
                force=resplit,
            )

            # Queue per-chunk transcription tasks
            for wav in chunks:
                out_json = wav.with_suffix(".whisper.json")
                if out_json.exists() and not retranscribe:
                    continue
                transcribe_q.append(TranscribeTask(wav_path=wav, out_json_path=out_json))

            jobs_list.append((m.path, wd))
            log.info("Chunks: %s -> %d", m.path.name, len(chunks))

        if not jobs_list:
            raise LjudanteckningError("No valid media jobs after chunking.")

        log.info("Chunks queued for transcription: %d", len(transcribe_q))

        # Transcription stage (multi-GPU)
        if transcribe_q:
            transcribe_tasks(
                tasks=transcribe_q,
                gpu_ids=gpu_ids,
                jobs=workers,
                model_name=settings.model,
                language=settings.language,
                compute_type=settings.compute_type,
                vad=settings.vad,
                beam_size=settings.beam_size,
                telemetry=telemetry,
            )

            log.info("Transcription stage: OK")
        else:
            log.info("Transcription stage: nothing to do (all chunk JSON exists)")

        # Merge + export stage (per media file)
        for media_path, wd in jobs_list:
            segments = merge_chunks(wd, settings.chunk_seconds)
            if not segments:
                log.warning("No segments for %s (skipping export)", media_path.name)
                continue

            out_base = media_path.with_suffix("")
            if settings.write_srt:
                export_srt(segments, out_base.with_suffix(".srt"))
            if settings.write_vtt:
                export_vtt(segments, out_base.with_suffix(".vtt"))
            if settings.write_txt:
                export_txt(segments, out_base.with_suffix(".txt"))

            det_lang = detect_language_from_chunks(wd)
            final_lang = settings.language or det_lang

            if settings.write_json:
                export_whisper_json(
                    segments,
                    out_base.with_suffix(".whisper.json"),
                    source=media_path,
                    model=settings.model,
                    language=final_lang,
                )

            cleanup_workdir(wd, settings.cleanup)
            log.info("Exported: %s", media_path.name)

        log.info(
            "Status: transcription + export OK. Next milestone is nicer progress + GPU telemetry."
        )

        log.info(
            "Chunking stage: %ds chunks @ %d Hz, %d ch",
            spec.chunk_seconds,
            spec.sample_rate,
            spec.channels,
        )

        # Re-probe each valid file to get per-file duration for progress bars
        # (we can optimize this later by caching probe results in memory)
        chunk_total = 0
        for m in media:
            info = ffprobe_info(m.path)
            if not info.ok or not info.has_audio or info.duration_s <= 0:
                continue

            chunks = split_to_chunks(
                media_path=m.path,
                workdir_name=settings.workdir_name,
                spec=spec,
                duration_s=info.duration_s,
                force=resplit,
            )
            chunk_total += len(chunks)
            log.info("Chunks: %s -> %d", m.path.name, len(chunks))

        log.info("Total chunks created/reused: %d", chunk_total)

    except LjudanteckningError as e:
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

    except LjudanteckningError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=2) from None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
