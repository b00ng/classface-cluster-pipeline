"""
Top‑level package for the classface clustering pipeline.

Exposes the public API and imports the console entry point in :mod:`classface_cluster.cli`.

The actual functionality is organised into smaller modules:

- :mod:`classface_cluster.config` – dataclasses for run configuration.
- :mod:`classface_cluster.db` – SQLite schema and helpers for recording runs.
- :mod:`classface_cluster.images` – scanning input folders, reading EXIF and computing perceptual hashes.
- :mod:`classface_cluster.embedders` – wrappers around InsightFace and AdaFace to produce face embeddings.
- :mod:`classface_cluster.embeddings_io` – reading/writing Parquet embedding parts, with support for resumable runs.
- :mod:`classface_cluster.clustering` – building a k‑NN graph and clustering faces into identities.
- :mod:`classface_cluster.thumbnails` – generating and caching thumbnails for HTML albums.
- :mod:`classface_cluster.album` – building HTML albums from cluster results.
- :mod:`classface_cluster.pipeline` – orchestrates a full run or resume, tying together all modules.
- :mod:`classface_cluster.ui` – Gradio interface for managing historical runs.

You can run the pipeline from the command line using the `classface` script installed by this
package.  See the README for usage details.
"""

__all__ = [
    "config",
    "db",
    "images",
    "embedders",
    "embeddings_io",
    "clustering",
    "thumbnails",
    "album",
    "pipeline",
    "ui",
]
