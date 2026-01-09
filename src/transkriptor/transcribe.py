from __future__ import annotations

import json
import os
import queue
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing import Queue, get_context
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .utils import TranskriptorError


@dataclass(frozen=True)
class TranscribeTask:
    wav_path: Path
    out_json_path: Path


def _compute_type_candidates(prefer: str | None) -> list[str]:
    if prefer:
        return [prefer]

    # Sensible CUDA-first defaults. Pascal often likes int8/int8_float16.
    return ["int8_float16", "int8", "float16", "float32"]


def _worker(
    task_q: Queue,
    result_q: Queue,
    gpu_id: str,
    model_name: str,
    language: str | None,
    compute_type_prefer: str | None,
    vad: bool,
    beam_size: int,
) -> None:
    """
    One worker == one GPU.
    We bind CUDA_VISIBLE_DEVICES before importing faster_whisper.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    try:
        from faster_whisper import WhisperModel  # import inside worker
    except Exception as e:
        result_q.put(("fatal", f"GPU {gpu_id}: import faster_whisper failed: {e!r}"))
        return

    model = None
    last_err: Exception | None = None
    for ct in _compute_type_candidates(compute_type_prefer):
        try:
            model = WhisperModel(model_name, device="cuda", compute_type=ct)
            result_q.put(("info", f"GPU {gpu_id}: model={model_name} compute_type={ct}"))
            break
        except Exception as e:
            last_err = e

    if model is None:
        result_q.put(("fatal", f"GPU {gpu_id}: could not init model ({last_err!r})"))
        return

    while True:
        item = task_q.get()
        if item is None:
            return

        wav_path, out_json_path = item
        try:
            segments, info = model.transcribe(
                str(wav_path),
                language=language,
                vad_filter=vad,
                beam_size=beam_size,
            )

            data = {
                "source": str(wav_path),
                "language": getattr(info, "language", None),
                "duration": getattr(info, "duration", None),
                "segments": [
                    {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
                    for s in segments
                    if (s.text or "").strip()
                ],
            }
            out_json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result_q.put(("done", str(wav_path)))
        except Exception as e:
            result_q.put(("fail", f"{wav_path}: {e!r}"))


def transcribe_tasks(
    tasks: Iterable[TranscribeTask],
    gpu_ids: list[str],
    jobs: int,
    model_name: str,
    language: str | None,
    compute_type: str | None,
    vad: bool,
    beam_size: int,
) -> None:
    tasks_list = list(tasks)
    if not tasks_list:
        return

    if not gpu_ids:
        raise TranskriptorError("No GPUs available for transcription.")

    workers_n = min(max(1, jobs), len(gpu_ids))

    ctx = get_context("spawn")
    task_q: Queue = ctx.Queue()
    result_q: Queue = ctx.Queue()

    procs = []
    for i in range(workers_n):
        gpu_id = gpu_ids[i]
        p = ctx.Process(
            target=_worker,
            args=(
                task_q,
                result_q,
                gpu_id,
                model_name,
                language,
                compute_type,
                vad,
                beam_size,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Feed tasks
    for t in tasks_list:
        task_q.put((t.wav_path, t.out_json_path))
    for _ in procs:
        task_q.put(None)

    total = len(tasks_list)
    done = 0
    failed = 0
    fatal: str | None = None

    with Progress(
        TextColumn("[bold]transcribing[/bold]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[detail]}", justify="left"),
        transient=True,
    ) as progress:
        task = progress.add_task("transcribe", total=total, detail="")

        while done + failed < total and fatal is None:
            try:
                kind, payload = result_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if kind == "info":
                progress.update(task, detail=f"[dim]{payload}[/dim]")
            elif kind == "done":
                done += 1
                progress.update(task, advance=1)
            elif kind == "fail":
                failed += 1
                progress.update(task, advance=1, detail=f"[red]{payload}[/red]")
            elif kind == "fatal":
                fatal = payload
            else:
                # unknown message
                continue

    # Ensure workers are stopped
    for p in procs:
        p.join(timeout=2)

    if fatal:
        raise TranskriptorError(f"Fatal worker error: {fatal}")

    if failed:
        raise TranskriptorError(
            f"{failed} chunk(s) failed transcription. Fix and rerun with --retranscribe."
        )
