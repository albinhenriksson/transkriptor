def test_cli_imports():
    import transkriptor.cli  # noqa: F401


def test_chunking_exports():
    from transkriptor.chunking import ChunkSpec  # noqa: F401


def test_transcribe_imports():
    import transkriptor.transcribe  # noqa: F401
