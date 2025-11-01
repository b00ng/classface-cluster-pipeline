"""
Command‑line entry point for the classface clustering pipeline.

This module parses command line arguments, constructs a :class:`RunConfig`
object and dispatches either to the pipeline runner or the Gradio UI.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _gpu_preflight() -> None:
    """Best-effort check for ONNX Runtime GPU availability and provide guidance.

    This does not stop execution; it only warns when CUDA is not available so
    users know how to enable GPU acceleration.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        # ONNX Runtime not importable — InsightFace will error later if missing.
        return
    providers = set(getattr(ort, "get_available_providers", lambda: [])())
    wants_cuda = "CUDAExecutionProvider" in {"CUDAExecutionProvider"}  # clarity
    if "CUDAExecutionProvider" not in providers:
        msg = (
            "GPU not detected by ONNX Runtime; falling back to CPU.\n"
            "To enable GPU: uninstall CPU onnxruntime and install CUDA build:\n"
            "  pip uninstall -y onnxruntime && pip install onnxruntime-gpu\n"
            "Ensure NVIDIA drivers + CUDA toolkit are correctly installed.\n"
            "Verify with: python -c 'import onnxruntime as o; print(o.get_available_providers())'\n"
            "On WSL2, also verify 'nvidia-smi' works inside your shell."
        )
        print(msg, file=sys.stderr)

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
        _gpu_preflight()
        run_pipeline(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
