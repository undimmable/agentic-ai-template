"""Expressive logging helpers.

Follows ``expressive-logging-for-agentic-debugging.md``: logs are namespaced
under ``agentic_ai`` so they can be turned up or down as one unit, and every
runtime step is traceable via a short run id.
"""

from __future__ import annotations

import logging
import sys

_ROOT_NAME = "agentic_ai"


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure the ``agentic_ai`` logger and return it.

    Logging goes to stderr so that agent answers on stdout stay pipe-friendly.
    """

    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logger = logging.getLogger(_ROOT_NAME)
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
