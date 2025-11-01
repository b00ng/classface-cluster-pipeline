# Repository Guidelines

## Project Structure & Module Organization
The `classface_cluster/` package is the heart of the project. Key modules include `cli.py` for the console entry point, `pipeline.py` for orchestration, `embedders.py` for model loading, `db.py` for SQLite helpers, `album.py` and `thumbnails.py` for deliverables, plus `ui.py` for the Gradio dashboard. Shared configuration lives in `config.py`, which exposes the `RunConfig` dataclass used across the stack. Add new modules alongside these to keep imports flat; place fixtures or synthetic image assets under `classface_cluster/tests_data/` if you introduce them. Future automated tests should live in a top-level `tests/` directory mirroring the package layout.

## Build, Test, and Development Commands
Create an isolated environment before hacking:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .  # add [adaface] to pull optional embedder extras
```

An editable install gives you the `classface` CLI. Common flows:

```bash
classface --input ~/photos --output ./runs --db ./runs.sqlite --embeddings-dir ./embeddings
classface --ui --db ./runs.sqlite --embeddings-dir ./embeddings  # launch Gradio manager
pytest  # once a suite exists
```

Document bespoke scripts in the README before relying on them in reviews.

## Coding Style & Naming Conventions
Stick to Python ≥3.9, 4-space indentation, and Black-compatible line widths (the existing modules sit comfortably under 100 characters). Functions and modules follow `snake_case`; classes use `PascalCase`. Preserve rich docstrings and type hints (`RunConfig`, dict annotations, etc.) to keep CLI contracts explicit. When adding CLI flags, update both `config.py` and `cli.py`, and record defaults in the docstring so the UI and pipeline stay in sync.

## Testing Guidelines
Adopt `pytest` with temporary directories for IO-heavy paths (`tmp_path` works well for embeddings and thumbnails). Name files `test_<module>.py` and mimic the package structure (`tests/test_pipeline.py`). Target lightweight unit tests for helpers plus integration smoke tests that stub heavy models via dependency injection (e.g., fake embedder returning deterministic vectors). Aim for meaningful coverage on data migration, resume logic, and SQLite writes; document any skipped tests when GPU or ONNX runtimes are unavailable.

## Commit & Pull Request Guidelines
Commits in history use short, imperative verbs (`Add pipeline resume handling`). Keep each commit scoped to one concern and mention affected modules when useful (`Tweak embeddings_io buffer size`). Pull requests should link tracking issues, outline the CLI command used for validation, and attach screenshots when UI behaviour changes. Flag optional dependencies or migration steps in the PR body so downstream automation knows when to install extras or run one-off scripts.
