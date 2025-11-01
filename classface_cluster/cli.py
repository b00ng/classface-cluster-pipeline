"""
Command‑line entry point for the classface clustering pipeline.

This module parses command line arguments, constructs a :class:`RunConfig`
object and dispatches either to the pipeline runner or the Gradio UI.
"""

from __future__ import annotations

from pathlib import Path
import sys

from .config import parse_args, RunConfig
from .pipeline import run_pipeline
from .ui import launch_ui


def main(argv: list[str] | None = None) -> None:
    """Entry point called by the ``classface`` script."""
    cfg = parse_args(argv)
    if cfg.extra.get("launch_ui"):
        host = cfg.extra.get("ui_host", "127.0.0.1")
        port = cfg.extra.get("ui_port", 7860)
        launch_ui(cfg.db_path, cfg.embeddings_dir, host=host, port=port)
    else:
        run_pipeline(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])