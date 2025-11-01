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
  --model iresnet100
```

Launch the Gradio UI to manage historical runs:

```bash
classface --ui \
  --db /path/to/classface_runs.sqlite \
  --embeddings-dir /path/to/embeddings
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