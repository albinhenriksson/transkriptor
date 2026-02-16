from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(verbosity: int) -> None:
    """
    Configure global logging.

    verbosity:
      0 -> WARNING
      1 -> INFO
      2+ -> DEBUG
    """
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
