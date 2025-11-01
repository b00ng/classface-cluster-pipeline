"""
Database layer for the classface clustering pipeline.

We maintain a SQLite database with a small number of normalized tables to
record metadata about each pipeline run and its associated images, faces
and clusters.  This enables resumable processing and allows the Gradio UI
to present historical results.

The tables are created automatically if they do not exist when connecting.
All interactions are implemented using SQLAlchemy Core for clarity and
compatibility with multiple versions of SQLAlchemy.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    Table, Column, Integer, String, Float, DateTime, Boolean, JSON, MetaData,
    ForeignKey, create_engine, select, insert, update, text
)
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError, OperationalError


def _make_metadata() -> MetaData:
    """Define and return SQLAlchemy metadata with our table definitions."""
    metadata = MetaData()
    # Table recording each run
    Table(
        "runs", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("input_dir", String, nullable=False),
        Column("output_root", String, nullable=False),
        Column("embeddings_dir", String, nullable=False),
        Column("model_name", String, nullable=False),
        Column("parameters", JSON, nullable=False),
        Column("status", String, nullable=False, default="running"),
        Column("start_time", DateTime, nullable=False),
        Column("end_time", DateTime, nullable=True),
        Column("n_images", Integer, nullable=True),
        Column("n_faces", Integer, nullable=True),
        Column("album_path", String, nullable=True),
        Column("command_line", String, nullable=True),
        Column("notes", String, nullable=True),
    )
    # Table of images processed in a run
    Table(
        "images", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", Integer, ForeignKey("runs.id"), nullable=False),
        Column("path", String, nullable=False),
        Column("timestamp", Float, nullable=False),  # seconds since epoch
        Column("exif_datetime", String, nullable=True),
        Column("width", Integer, nullable=True),
        Column("height", Integer, nullable=True),
        Column("phash", String, nullable=True),
        Column("face_count", Integer, nullable=True),
    )
    # Table of faces detected
    Table(
        "faces", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", Integer, ForeignKey("runs.id"), nullable=False),
        Column("image_id", Integer, ForeignKey("images.id"), nullable=False),
        Column("emb_row", Integer, nullable=False),  # index within the Parquet file
        Column("bbox_x1", Float, nullable=False),
        Column("bbox_y1", Float, nullable=False),
        Column("bbox_x2", Float, nullable=False),
        Column("bbox_y2", Float, nullable=False),
        Column("det_score", Float, nullable=True),
        Column("pose_yaw", Float, nullable=True),
        Column("pose_pitch", Float, nullable=True),
        Column("pose_roll", Float, nullable=True),
        Column("timestamp", Float, nullable=False),
    )
    # Table of clusters produced per run
    Table(
        "clusters", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", Integer, ForeignKey("runs.id"), nullable=False),
        Column("label", String, nullable=False),
        Column("size", Integer, nullable=False),
        Column("student_name", String, nullable=True),
        Column("centroid", JSON, nullable=True),  # list of floats
        Column("sample_face_id", Integer, ForeignKey("faces.id"), nullable=True),
    )
    return metadata


def init_db(db_path: Path) -> Engine:
    """Initialize the database and create tables if they do not exist.

    Parameters
    ----------
    db_path: Path
        Location of the SQLite database file.

    Returns
    -------
    sqlalchemy.Engine
        Connected engine instance.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    metadata = _make_metadata()
    metadata.create_all(engine)
    return engine


def _bulk_insert_with_ids(conn: Connection, table: Table, rows: List[Dict[str, Any]]) -> List[int]:
    """Insert many rows and return their primary keys.

    Uses ``INSERT ... RETURNING`` when available, otherwise falls back to
    row-by-row inserts to remain compatible with older SQLite versions.
    """
    if not rows:
        return []
    stmt = insert(table)
    try:
        result = conn.execute(stmt.returning(table.c.id), rows)
        ids = [int(pk) for pk in result.scalars()]
        conn.commit()
        return ids
    except OperationalError as exc:
        # RETURNING not supported on this SQLite build; fall back to per-row inserts
        if conn.in_transaction():
            conn.rollback()
        message = str(exc).upper()
        if "RETURNING" not in message:
            raise
    except SQLAlchemyError:
        if conn.in_transaction():
            conn.rollback()
        raise

    inserted_ids: List[int] = []
    for row in rows:
        single_result = conn.execute(insert(table).values(**row))
        inserted_ids.append(int(single_result.inserted_primary_key[0]))
    conn.commit()
    return inserted_ids


def record_run_start(conn: Connection, input_dir: Path, output_root: Path, embeddings_dir: Path,
                      model_name: str, parameters: Dict[str, Any], command_line: Optional[str] = None) -> int:
    """Insert a new run row and return its ID.

    Parameters
    ----------
    conn: Connection
        Open database connection.
    input_dir, output_root, embeddings_dir: Path
        Paths recorded for the run.
    model_name: str
        Name of the embedding model.
    parameters: dict
        JSON‑serialisable dictionary of configuration parameters.
    command_line: str, optional
        Original command line invocation.

    Returns
    -------
    int
        ID of the newly created run.
    """
    runs = _make_metadata().tables["runs"]
    now = _dt.datetime.utcnow()
    result = conn.execute(
        insert(runs).values(
            input_dir=str(input_dir),
            output_root=str(output_root),
            embeddings_dir=str(embeddings_dir),
            model_name=model_name,
            parameters=parameters,
            status="running",
            start_time=now,
            command_line=command_line,
        )
    )
    conn.commit()
    return int(result.inserted_primary_key[0])


def record_run_end(conn: Connection, run_id: int, status: str,
                   n_images: int, n_faces: int, album_path: Optional[str] = None,
                   notes: Optional[str] = None) -> None:
    """Update a run row to mark it as finished.

    Parameters
    ----------
    conn: Connection
        Open database connection.
    run_id: int
        Primary key of the run to update.
    status: str
        Final status: ``"done"``, ``"interrupted"``, ``"no_faces"``.
    n_images: int
        Number of unique images processed.
    n_faces: int
        Number of faces detected.
    album_path: str, optional
        Relative path to the generated album directory.
    notes: str, optional
        Additional notes to store (e.g. error messages).
    """
    runs = _make_metadata().tables["runs"]
    conn.execute(
        update(runs)
        .where(runs.c.id == run_id)
        .values(
            status=status,
            end_time=_dt.datetime.utcnow(),
            n_images=n_images,
            n_faces=n_faces,
            album_path=album_path,
            notes=notes,
        )
    )
    conn.commit()


def insert_images(conn: Connection, run_id: int, image_records: Iterable[Dict[str, Any]]) -> List[int]:
    """Bulk insert image metadata and return inserted primary keys.

    Each record must include at least ``path``, ``timestamp`` and may include
    ``exif_datetime``, ``width``, ``height``, ``phash``, ``face_count``.

    Returns
    -------
    list of int
        Primary keys of inserted rows in order.
    """
    images_table = _make_metadata().tables["images"]
    rows = [dict(run_id=run_id, **rec) for rec in image_records]
    return _bulk_insert_with_ids(conn, images_table, rows)


def insert_faces(conn: Connection, run_id: int, face_records: Iterable[Dict[str, Any]]) -> List[int]:
    """Bulk insert face metadata and return inserted primary keys.

    Each record must include ``image_id`` and ``emb_row`` plus bounding box fields
    ``bbox_x1``, ``bbox_y1``, ``bbox_x2``, ``bbox_y2``.  Additional fields such
    as detection score and pose angles may be supplied.
    """
    faces_table = _make_metadata().tables["faces"]
    rows = [dict(run_id=run_id, **rec) for rec in face_records]
    return _bulk_insert_with_ids(conn, faces_table, rows)


def insert_clusters(conn: Connection, run_id: int, cluster_records: Iterable[Dict[str, Any]]) -> List[int]:
    """Bulk insert cluster metadata and return inserted primary keys.

    Each record must include ``label`` and ``size``; optional fields are
    ``student_name``, ``centroid`` (list) and ``sample_face_id``.
    """
    clusters_table = _make_metadata().tables["clusters"]
    rows = [dict(run_id=run_id, **rec) for rec in cluster_records]
    return _bulk_insert_with_ids(conn, clusters_table, rows)


def get_run(conn: Connection, run_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single run by ID as a dictionary, or ``None`` if not found."""
    runs = _make_metadata().tables["runs"]
    row = conn.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
    return dict(row) if row else None


def list_runs(conn: Connection, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return a list of run records, optionally filtered by status."""
    runs = _make_metadata().tables["runs"]
    query = select(runs)
    if status:
        query = query.where(runs.c.status == status)
    rows = conn.execute(query.order_by(runs.c.start_time.desc())).mappings().all()
    return [dict(row) for row in rows]


def update_run_status(conn: Connection, run_id: int, status: str, notes: Optional[str] = None) -> None:
    """Update the status (and optionally notes) for a run.

    This function is typically used to mark a run as interrupted when
    resuming fails or upon cancellation.
    """
    runs = _make_metadata().tables["runs"]
    conn.execute(
        update(runs)
        .where(runs.c.id == run_id)
        .values(status=status, notes=notes)
    )
    conn.commit()
