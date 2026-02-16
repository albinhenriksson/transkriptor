from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def _fmt_srt(t: float) -> str:
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(t: float) -> str:
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def merge_chunks(workdir: Path, chunk_seconds: int) -> list[Segment]:
    jsons = sorted(workdir.glob("chunk_*.whisper.json"))
    if not jsons:
        return []

    out: list[Segment] = []
    for jf in jsons:
        m = re.search(r"chunk_(\d+)\.whisper\.json$", jf.name)
        if not m:
            continue
        idx = int(m.group(1))
        offset = idx * float(chunk_seconds)

        data = json.loads(jf.read_text(encoding="utf-8"))
        segs = data.get("segments", []) or []
        for s in segs:
            text = str(s.get("text", "")).strip()
            if not text:
                continue
            out.append(
                Segment(
                    start=float(s["start"]) + offset,
                    end=float(s["end"]) + offset,
                    text=text,
                )
            )

    out.sort(key=lambda x: (x.start, x.end))
    return out


def write_srt(segments: list[Segment], out_path: Path) -> None:
    lines: list[str] = []
    for i, s in enumerate(segments, start=1):
        lines += [
            str(i),
            f"{_fmt_srt(s.start)} --> {_fmt_srt(s.end)}",
            s.text,
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[Segment], out_path: Path) -> None:
    lines: list[str] = ["WEBVTT", ""]
    for s in segments:
        lines += [
            f"{_fmt_vtt(s.start)} --> {_fmt_vtt(s.end)}",
            s.text,
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: list[Segment], out_path: Path) -> None:
    # Timestamped lines are great for grep and human scanning.
    lines = [f"{_fmt_vtt(s.start)}\t{s.text}" for s in segments]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_whisper_json(
    segments: list[Segment],
    out_path: Path,
    source: Path,
    model: str,
    language: str | None,
) -> None:
    data = {
        "source": str(source),
        "model": model,
        "language": language,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_language_from_chunks(workdir: Path) -> str | None:
    for jf in sorted(workdir.glob("chunk_*.whisper.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        lang = data.get("language")
        if lang:
            return str(lang)
    return None
