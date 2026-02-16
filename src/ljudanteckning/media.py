from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

from .utils import normalize_path

SKIP_DIR_NAMES = {
    ".ljudanteckning",
    ".git",
    ".venv",
    "__pycache__",
}


@dataclass(frozen=True)
class MediaFile:
    path: Path
    ext: str  # normalized lower-case suffix (may be empty)


def _is_excluded(p: Path, base: Path, patterns: list[str]) -> bool:
    """
    Exclude logic:
    - If pattern contains a path separator, match against relative path (posix-style).
    - Otherwise, match against basename.
    """
    if not patterns:
        return False

    try:
        rel = p.relative_to(base)
        rel_s = rel.as_posix()
    except Exception:
        rel_s = p.as_posix()

    name = p.name

    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue

        if "/" in pat or "\\" in pat:
            if fnmatch.fnmatch(rel_s, pat):
                return True
        else:
            if fnmatch.fnmatch(name, pat):
                return True

    return False


def discover_media(target: Path, exclude_patterns: list[str]) -> list[MediaFile]:
    """
    Discover candidate files under `target` (file or directory).

    This is intentionally extension-agnostic:
    - We include any file not excluded by patterns.
    - The actual "is it media with audio?" decision happens in ffprobe stage.
    """
    target = normalize_path(target)

    if target.is_file():
        if _is_excluded(target, target.parent, exclude_patterns):
            return []
        return [MediaFile(path=target, ext=target.suffix.lower())]

    base = target
    out: list[MediaFile] = []

    stack = [base]
    while stack:
        d = stack.pop()
        if d.name in SKIP_DIR_NAMES:
            continue

        try:
            for child in d.iterdir():
                if child.is_dir():
                    if child.name in SKIP_DIR_NAMES:
                        continue
                    stack.append(child)
                    continue

                if not child.is_file():
                    continue

                if _is_excluded(child, base, exclude_patterns):
                    continue

                out.append(MediaFile(path=child, ext=child.suffix.lower()))
        except PermissionError:
            continue

    out.sort(key=lambda m: m.path.as_posix())
    return out
