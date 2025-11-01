# Classface Cluster Pipeline

## Overview

This project contains a flexible pipeline for organizing class photo archives.  
Given a folder full of photographs spanning multiple school years, the pipeline detects faces, generates embeddings, clusters them into student identities, and produces tidy folders and HTML albums for each student.  

Key features include:

* **InsightFace/AdaFace embeddings** – choose from multiple models (e.g. iResNet100 glint360k or AdaFace) via a command‑line flag.
* **SQLite tracking** – every run is recorded in a central database with metadata (input/output paths, parameters, face counts, runtime, etc.).
* **Parquet storage** – each run’s face embeddings are saved as Parquet parts to enable incremental resumes.
* **Resume support** – interrupted runs can be continued later; only new images are processed and embeddings appended.
* **Optimised thumbnails** – JPEG thumbnails are generated in parallel and cached to speed up subsequent album builds.
* **Interactive UI** – a Gradio‑based interface lets you browse run history, inspect parameters and results, and resume incomplete runs.

See `pyproject.toml` for a list of dependencies and optional extras.  The package installs a console entry‑point `classface` for running the pipeline or launching the UI.

## Quick start

Install the package (for development you can use `-e` to install in editable mode):

```bash
pip install -e .
```

Run a new analysis on a folder of images:

```bash
classface \
  --input /path/to/photos \
  --output /path/to/output_root \
  --db /path/to/classface_runs.sqlite \
  --embeddings-dir /path/to/embeddings \
  --model iresnet100 \
  --use-phash            # optional, deduplicate identical photos by perceptual hash

# If you re-run a job and want thumbnails rebuilt, add:
#   --thumb-regen
```

Launch the Gradio UI to manage historical runs:

```bash
classface --ui \
  --db /path/to/classface_runs.sqlite \
  --embeddings-dir /path/to/embeddings

## Duplicates and rebuilds

- Content duplicates: pass `--use-phash` to compute a perceptual hash per image and collapse
  visually identical files (even if they have different filenames). This reduces repeated
  copies in each student folder.
- Album rebuilds: if you change clustering parameters or re-run the same input, use
  `--thumb-regen` to force regeneration of thumbnails. The album builder also cleans per-student
  folders to remove stale files from previous runs.

Note for WSL/Windows environments: the album stage copies original images into the student
folders (hardlinks are avoided for cross-filesystem compatibility). Ensure the output root
is writable from your shell.

## GPU setup (optional but recommended)

The pipeline automatically uses GPU if available via ONNX Runtime. If it falls back to CPU
and you have an NVIDIA GPU, install the CUDA build of ONNX Runtime and verify providers:

1) Install GPU build (and remove CPU build if present):

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

2) Ensure NVIDIA drivers and CUDA runtime are installed for your platform. On WSL2, install
the Windows NVIDIA driver with WSL support and confirm `nvidia-smi` works inside WSL.

3) Verify providers:

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

You should see `['CUDAExecutionProvider', 'CPUExecutionProvider']`. If not, the CLI will print a
hint on how to enable GPU and continue on CPU.
```

## Repository layout

```
classface_cluster/
├── __init__.py
├── album.py        # HTML album generation
├── cli.py          # Console entry point and argument parsing
├── clustering.py    # kNN graph and clustering logic
├── config.py       # Data classes for run parameters
├── db.py           # SQLite schema and helper functions
├── embeddings_io.py# Read/write Parquet embedding parts
├── embedders.py    # Model loading and face embedding
├── images.py       # Image scanning, EXIF reading and hashing
├── pipeline.py     # Top‑level run/resume orchestration
├── thumbnails.py    # Thumbnail caching and generation
└── ui.py           # Gradio run manager
```

Each module has detailed docstrings and is designed to be reusable.  Feel free to inspect individual files for more implementation details.
