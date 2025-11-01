"""
Parquet input/output for face embeddings.

Embeddings are stored per run in a dedicated directory.  Each run may
consist of multiple ``part-*.parquet`` files appended over time when a
resumable run is interrupted and later continued.  Each row corresponds to
a detected face and includes the embedding vector along with associated
metadata (image ID, bounding box, timestamps, etc.).

We use PyArrow's Parquet support to efficiently write and read these files.
Embeddings are stored as lists of floats in a ``embedding`` column.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Dict, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _run_dir(base: Path, run_id: int) -> Path:
    return base / f"run_{run_id:06d}"


def next_part_index(run_path: Path) -> int:
    """Determine the next part index for a run directory.

    Scans existing files matching ``part-*.parquet`` and returns an
    incrementing index.  If no parts exist, returns 0.
    """
    max_idx = -1
    for f in run_path.glob("part-*.parquet"):
        try:
            idx = int(f.stem.split("-")[1])
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    return max_idx + 1


def count_existing_rows(run_path: Path) -> int:
    """Count the total number of rows across existing parts.

    Used to determine the starting ``emb_row`` index when resuming a run.
    """
    total = 0
    for f in run_path.glob("part-*.parquet"):
        try:
            table = pq.read_table(f, columns=["embedding"])
            total += table.num_rows
        except Exception:
            continue
    return total


def write_embedding_part(base: Path, run_id: int, emb_records: Iterable[Dict[str, Any]]) -> None:
    """Write a batch of face embedding records to a new Parquet part.

    Parameters
    ----------
    base: Path
        Embeddings root directory (should correspond to ``--embeddings-dir``).
    run_id: int
        Identifier of the current run.
    emb_records: iterable of dict
        Each element must contain serialisable keys: ``emb_row``, ``image_id``,
        ``path``, ``timestamp``, ``bbox_x1``, ``bbox_y1``, ``bbox_x2``,
        ``bbox_y2``, ``det_score``, ``pose_yaw``, ``pose_pitch``,
        ``pose_roll``, and ``embedding`` (list of floats).
    """
    run_path = _run_dir(base, run_id)
    run_path.mkdir(parents=True, exist_ok=True)
    part_index = next_part_index(run_path)
    part_path = run_path / f"part-{part_index:03d}.parquet"
    # Build a PyArrow table from the records
    df = pd.DataFrame(list(emb_records))
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, part_path)


def read_embeddings(base: Path, run_id: int) -> pd.DataFrame:
    """Read all embedding parts for a run and return a concatenated DataFrame."""
    run_path = _run_dir(base, run_id)
    dfs: List[pd.DataFrame] = []
    for part in sorted(run_path.glob("part-*.parquet")):
        try:
            df = pq.read_table(part).to_pandas()
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()