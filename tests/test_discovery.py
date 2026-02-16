from pathlib import Path

from ljudanteckning.media import discover_media


def test_discover_media_filters_exts(tmp_path: Path):
    (tmp_path / "a.txt").write_text("nope", encoding="utf-8")
    (tmp_path / "b.mkv").write_text("fake", encoding="utf-8")
    (tmp_path / "c.wav").write_text("fake", encoding="utf-8")

    found = discover_media(tmp_path, exclude_patterns=["*.txt"])
    exts = sorted([m.ext for m in found])
    assert exts == [".mkv", ".wav"]
