"""
Configuration structures for running the classface clustering pipeline.

We use :class:`dataclasses.dataclass` to describe the parameters accepted by the
command line interface and stored in the SQLite database.  Each field
corresponds to a user‑controllable tuning parameter, with sensible defaults.

The :func:`parse_args` function converts command line arguments into a
:class:`RunConfig` instance.  The command line options are intentionally kept
concise; for more advanced usage (e.g. passing a fully customised model or
embedding module) users can modify the resulting config object before
invoking the pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class RunConfig:
    """Parameters controlling a single pipeline run.

    Attributes
    ----------
    input_dir: Path
        Directory containing photos to process.  This must exist and may
        contain nested subdirectories.  Only files that can be read by
        OpenCV/Pillow will be considered.
    output_root: Path
        Root directory where per‑student folders and HTML albums will be
        created.  For each run, a subdirectory named ``run_<id>`` will be
        created underneath this root.
    db_path: Path
        Path to a SQLite database used to record run metadata.  The database
        will be created automatically if it does not exist.
    embeddings_dir: Path
        Directory where Parquet embedding parts are written for each run.
        Each run will have a subdirectory named ``run_<id>`` and may
        contain multiple ``part-*.parquet`` files if the run is resumed.
    model_name: str
        Name of the embedding model.  Supported values include
        ``"iresnet100"`` (the default), ``"antelopev2"`` (an InsightFace
        model package), and ``"adaface"``.  See
        :mod:`classface_cluster.embedders` for details.
    adaface_weights: Optional[Path]
        Optional path to a custom weight file when using AdaFace.  Ignored
        for InsightFace models.
    min_face_size: int
        Minimum bounding box side length (in pixels) for detected faces to
        be processed.  Faces smaller than this are discarded.
    sim_threshold: float
        Cosine similarity threshold for creating edges in the k‑NN graph.
        Lower values connect more faces but risk over‑linking; higher values
        yield more conservative clusters.
    merge_threshold: float
        Cosine similarity threshold used when merging small clusters into
        larger ones based on centroid similarity.
    min_cluster_size: int
        Minimum number of faces in a cluster to be considered a valid
        identity.  Smaller clusters will be placed in an ``Unknown``
        category.
    knn_k: int
        Number of nearest neighbours to query per face when constructing
        the k‑NN graph.  Typical values are between 30 and 50.
    start_month: int
        Month (1–12) considered as the start of an academic year.
        Photographs are grouped by academic year for album purposes.
    hdbscan: bool
        Whether to use HDBSCAN for clustering instead of connected
        components.  HDBSCAN can detect non‑spherical clusters but is
        slower on large datasets.
    hdbscan_min_cluster_size: int
        Minimum cluster size parameter for HDBSCAN.  Ignored when
        ``hdbscan`` is ``False``.
    hdbscan_min_samples: int
        Minimum samples parameter for HDBSCAN (controls cluster persistence).
    thumbs_size: int
        Maximum side length of generated thumbnails (in pixels).  Thumbnails
        preserve aspect ratio and are used in the HTML albums.
    thumbs_workers: int
        Number of worker processes used to generate thumbnails in parallel.
    thumb_regen: bool
        Whether to regenerate thumbnails even if they exist.  Default is
        ``False`` to reuse cached thumbnails.
    use_phash: bool
        Whether to compute perceptual hashes of images and skip duplicates.
        Hash computation adds overhead but can reduce processing of
        identical photos.
    command_line: Optional[str]
        Full original command line invocation, recorded for reproducibility.
    """
    input_dir: Path
    output_root: Path
    db_path: Path
    embeddings_dir: Path
    model_name: str = "iresnet100"
    adaface_weights: Optional[Path] = None
    min_face_size: int = 80
    sim_threshold: float = 0.55
    merge_threshold: float = 0.58
    min_cluster_size: int = 5
    knn_k: int = 50
    start_month: int = 6
    hdbscan: bool = False
    hdbscan_min_cluster_size: int = 8
    hdbscan_min_samples: int = 2
    thumbs_size: int = 420
    thumbs_workers: int = 4
    thumb_regen: bool = False
    use_phash: bool = False
    command_line: Optional[str] = None
    # Additional fields can be stored as needed
    extra: Dict[str, Any] = field(default_factory=dict)

def parse_args(argv: Optional[list[str]] = None) -> RunConfig:
    """Parse command line arguments and return a :class:`RunConfig` instance.

    Parameters
    ----------
    argv: list of str, optional
        List of command line arguments.  If omitted, :mod:`sys.argv` will be
        used.  This parameter facilitates testing.

    Returns
    -------
    RunConfig
        Populated configuration object.
    """
    parser = argparse.ArgumentParser(
        description="Classface clustering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", dest="input_dir", type=Path, required=False,
                        help="Path to input folder containing images")
    parser.add_argument("--output", dest="output_root", type=Path, required=False,
                        help="Output root folder (subfolder per run will be created)")
    parser.add_argument("--db", dest="db_path", type=Path, required=False,
                        help="Path to SQLite database file")
    parser.add_argument("--embeddings-dir", dest="embeddings_dir", type=Path, required=False,
                        help="Directory to store Parquet embedding parts per run")
    parser.add_argument("--model", dest="model_name", type=str, default="iresnet100",
                        choices=["iresnet100", "antelopev2", "adaface"],
                        help="Embedding model to use")
    parser.add_argument("--adaface-weights", dest="adaface_weights", type=Path, required=False,
                        help="Path to custom AdaFace weight file")
    parser.add_argument("--min-face-size", dest="min_face_size", type=int, default=80,
                        help="Discard faces smaller than this many pixels")
    parser.add_argument("--sim-threshold", dest="sim_threshold", type=float, default=0.55,
                        help="Cosine similarity threshold for kNN graph")
    parser.add_argument("--merge-threshold", dest="merge_threshold", type=float, default=0.58,
                        help="Cosine similarity threshold for merging clusters")
    parser.add_argument("--min-cluster-size", dest="min_cluster_size", type=int, default=5,
                        help="Minimum cluster size to consider as a valid identity")
    parser.add_argument("--knn-k", dest="knn_k", type=int, default=50,
                        help="Number of nearest neighbours to query for kNN graph")
    parser.add_argument("--start-month", dest="start_month", type=int, default=6,
                        help="Month considered as the start of an academic year (1–12)")
    parser.add_argument("--hdbscan", dest="hdbscan", action="store_true",
                        help="Use HDBSCAN for clustering instead of connected components")
    parser.add_argument("--hdbscan-min-cluster-size", dest="hdbscan_min_cluster_size", type=int,
                        default=8, help="Minimum cluster size parameter for HDBSCAN")
    parser.add_argument("--hdbscan-min-samples", dest="hdbscan_min_samples", type=int, default=2,
                        help="Minimum samples parameter for HDBSCAN")
    parser.add_argument("--thumb-size", dest="thumbs_size", type=int, default=420,
                        help="Maximum side length of generated thumbnails (pixels)")
    parser.add_argument("--thumbs-workers", dest="thumbs_workers", type=int, default=4,
                        help="Number of parallel workers for thumbnail generation")
    parser.add_argument("--thumb-regen", dest="thumb_regen", action="store_true",
                        help="Regenerate thumbnails even if cached versions exist")
    parser.add_argument("--use-phash", dest="use_phash", action="store_true",
                        help="Compute perceptual hashes of images to skip duplicates")
    parser.add_argument("--resume", dest="resume_run", type=int, default=None,
                        help="ID of a previous run to resume (ignore --input/--model/etc.)")
    parser.add_argument("--ui", dest="launch_ui", action="store_true",
                        help="Launch the Gradio UI instead of running a pipeline")
    parser.add_argument("--ui-host", dest="ui_host", type=str, default="127.0.0.1",
                        help="Host address for the UI server")
    parser.add_argument("--ui-port", dest="ui_port", type=int, default=7860,
                        help="Port for the UI server")
    args = parser.parse_args(argv)

    # When launching the UI, only db_path and embeddings_dir need to be provided.
    if args.launch_ui:
        if args.db_path is None:
            parser.error("--db must be provided when launching the UI")
        if args.embeddings_dir is None:
            parser.error("--embeddings-dir must be provided when launching the UI")
        return RunConfig(
            input_dir=Path("."),
            output_root=Path("."),
            db_path=args.db_path,
            embeddings_dir=args.embeddings_dir,
            command_line=" ".join(parser.prog + " " + " ".join(argv or [])),
            extra={"launch_ui": True, "ui_host": args.ui_host, "ui_port": args.ui_port}
        )

    # For a regular run, ensure required fields are present
    if args.resume_run is None:
        missing = [name for name in ("input_dir", "output_root", "db_path", "embeddings_dir")
                   if getattr(args, name) is None]
        if missing:
            parser.error("Missing required arguments: " + ", ".join(missing))

    return RunConfig(
        input_dir=args.input_dir if args.input_dir else Path("."),
        output_root=args.output_root if args.output_root else Path("."),
        db_path=args.db_path if args.db_path else Path("classface_runs.sqlite"),
        embeddings_dir=args.embeddings_dir if args.embeddings_dir else Path("embeddings"),
        model_name=args.model_name,
        adaface_weights=args.adaface_weights,
        min_face_size=args.min_face_size,
        sim_threshold=args.sim_threshold,
        merge_threshold=args.merge_threshold,
        min_cluster_size=args.min_cluster_size,
        knn_k=args.knn_k,
        start_month=args.start_month,
        hdbscan=args.hdbscan,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        thumbs_size=args.thumbs_size,
        thumbs_workers=args.thumbs_workers,
        thumb_regen=args.thumb_regen,
        use_phash=args.use_phash,
        command_line=" ".join(parser.prog + " " + " ".join(argv or [])),
        extra={"resume_run": args.resume_run}
    )
