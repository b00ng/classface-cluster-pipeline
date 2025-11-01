"""
High‑level orchestration of the classface clustering pipeline.

This module ties together the lower‑level components: scanning images,
detection/embedding, clustering, thumbnail generation, and album creation.
It interacts with the SQLite database to record run metadata and supports
resuming interrupted runs.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .config import RunConfig
from .db import (
    init_db, record_run_start, record_run_end, update_run_status,
    insert_images, insert_faces, insert_clusters, get_run
)
from sqlalchemy import text
from .images import scan_images
from .embedders import get_embedder
from .embeddings_io import write_embedding_part, read_embeddings, count_existing_rows
from .clustering import cluster_embeddings, assign_labels
from .album import build_album


def run_pipeline(config: RunConfig) -> None:
    """Run or resume a clustering pipeline based on the provided configuration.

    This function manages the overall control flow: connecting to the database,
    loading an embedder, scanning images, computing embeddings, clustering,
    copying files and writing the HTML album.  It records progress and status
    in the database so that interrupted runs can be resumed later.

    Parameters
    ----------
    config: RunConfig
        Configuration settings for this run.
    """
    # Initialise or connect to the database
    engine = init_db(config.db_path)
    with engine.connect() as conn:
        if config.extra.get("resume_run"):
            run_id = int(config.extra["resume_run"])
            run_record = get_run(conn, run_id)
            if not run_record:
                raise RuntimeError(f"Run {run_id} does not exist.")
        else:
            # Create a new run record
            params_json = {
                "min_face_size": config.min_face_size,
                "sim_threshold": config.sim_threshold,
                "merge_threshold": config.merge_threshold,
                "min_cluster_size": config.min_cluster_size,
                "knn_k": config.knn_k,
                "start_month": config.start_month,
                "hdbscan": config.hdbscan,
                "hdbscan_min_cluster_size": config.hdbscan_min_cluster_size,
                "hdbscan_min_samples": config.hdbscan_min_samples,
                "thumbs_size": config.thumbs_size,
                "thumbs_workers": config.thumbs_workers,
                "thumb_regen": config.thumb_regen,
                "use_phash": config.use_phash,
            }
            run_id = record_run_start(
                conn,
                input_dir=config.input_dir,
                output_root=config.output_root,
                embeddings_dir=config.embeddings_dir,
                model_name=config.model_name,
                parameters=params_json,
                command_line=config.command_line,
            )
            run_record = get_run(conn, run_id)
        # Determine existing embeddings for resume
        resume = config.extra.get("resume_run") is not None
        emb_offset = 0
        emb_df = None
        # Load images already processed in this run
        processed_paths = set()
        if resume:
            # Count existing embedding rows
            emb_offset = count_existing_rows(Path(run_record["embeddings_dir"]), run_id)
            # Read embeddings for clustering
            emb_df = read_embeddings(Path(run_record["embeddings_dir"]), run_id)
            # Determine processed image paths from DB
            # We cannot query DB in SQLAlchemy elegantly without ORM; we can use raw SQL
            res = conn.execute(text("SELECT path FROM images WHERE run_id = :rid"), {"rid": run_id}).fetchall()
            processed_paths = {Path(row[0]) for row in res}
        # Instantiate the embedder
        embedder = get_embedder(
            model_name=config.model_name,
            adaface_weights=str(config.adaface_weights) if config.adaface_weights else None,
            min_face_size=config.min_face_size,
            use_gpu=True,
        )
        # Process images
        face_records: List[Dict[str, Any]] = []
        emb_records: List[Dict[str, Any]] = []
        images_records: List[Dict[str, Any]] = []
        face_counter = emb_offset
        image_id_map: Dict[Path, int] = {}
        try:
            for img_path, meta in scan_images(config.input_dir, use_phash=config.use_phash):
                if resume and img_path in processed_paths:
                    continue
                # Read image using cv2 (BGR)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                # Insert image into DB and get its ID
                image_rec = {
                    "path": str(img_path),
                    "timestamp": meta["timestamp"],
                    "exif_datetime": meta.get("exif_datetime"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "phash": meta.get("phash"),
                    "face_count": None,
                }
                img_ids = insert_images(conn, run_id, [image_rec])
                image_id = img_ids[0]
                image_id_map[img_path] = image_id
                images_records.append({
                    "id": image_id,
                    "path": str(img_path),
                    "timestamp": meta["timestamp"],
                })
                # Detect and embed faces
                faces = embedder.extract_faces(img)
                if not faces:
                    continue
                # Update face_count in DB
                conn.execute(text("UPDATE images SET face_count = :fc WHERE id = :iid"), {"fc": len(faces), "iid": image_id})
                conn.commit()
                for face in faces:
                    record = {
                        "emb_row": face_counter,
                        "image_id": image_id,
                        "bbox_x1": face["bbox"][0],
                        "bbox_y1": face["bbox"][1],
                        "bbox_x2": face["bbox"][2],
                        "bbox_y2": face["bbox"][3],
                        "det_score": face.get("det_score"),
                        "pose_yaw": face["pose"].get("yaw"),
                        "pose_pitch": face["pose"].get("pitch"),
                        "pose_roll": face["pose"].get("roll"),
                        "timestamp": meta["timestamp"],
                    }
                    face_records.append(record)
                    emb_records.append({
                        "emb_row": face_counter,
                        "image_id": image_id,
                        "path": str(img_path),
                        "timestamp": meta["timestamp"],
                        "bbox_x1": face["bbox"][0],
                        "bbox_y1": face["bbox"][1],
                        "bbox_x2": face["bbox"][2],
                        "bbox_y2": face["bbox"][3],
                        "det_score": face.get("det_score"),
                        "pose_yaw": face["pose"].get("yaw"),
                        "pose_pitch": face["pose"].get("pitch"),
                        "pose_roll": face["pose"].get("roll"),
                        "embedding": face["embedding"].tolist(),
                    })
                    face_counter += 1
            # After processing all images, insert faces and embeddings
            if face_records:
                insert_faces(conn, run_id, face_records)
                # Write embeddings part to Parquet
                write_embedding_part(Path(config.embeddings_dir), run_id, emb_records)
        except Exception as exc:
            # Roll back any pending transaction before updating status
            if conn.in_transaction():
                conn.rollback()
            # Mark run as interrupted and store exception
            update_run_status(conn, run_id, "interrupted", notes=str(exc))
            traceback.print_exc()
            return
        # Combine embeddings (existing and new) for clustering
        if resume and emb_df is not None:
            # Combine existing and new embeddings
            new_df = read_embeddings(Path(config.embeddings_dir), run_id)
            df = new_df
        else:
            df = read_embeddings(Path(config.embeddings_dir), run_id)
        if df.empty:
            # No faces detected
            record_run_end(conn, run_id, status="no_faces", n_images=len(images_records), n_faces=0)
            return
        # Data for clustering
        embeddings = np.vstack(df["embedding"]).astype(np.float32)
        image_ids = df["image_id"].tolist()
        # Cluster
        clusters = cluster_embeddings(
            embeddings=embeddings,
            image_ids=image_ids,
            sim_threshold=config.sim_threshold,
            merge_threshold=config.merge_threshold,
            min_cluster_size=config.min_cluster_size,
            knn_k=config.knn_k,
            use_hdbscan=config.hdbscan,
            hdbscan_min_cluster_size=config.hdbscan_min_cluster_size,
            hdbscan_min_samples=config.hdbscan_min_samples,
        )
        labels = assign_labels(clusters)
        # Build cluster map for album generation and DB insertion
        cluster_map: Dict[str, List[int]] = {}
        cluster_records: List[Dict[str, Any]] = []
        for cluster, label in zip(clusters, labels):
            cluster_map[label] = cluster
            # Choose sample face as the first face index
            sample_face_idx = cluster[0]
            sample_face_row = df.iloc[sample_face_idx]
            cluster_records.append({
                "label": label,
                "size": len(cluster),
                "student_name": None,
                "centroid": None,
                "sample_face_id": int(sample_face_row["emb_row"]),
            })
        # Insert clusters into DB (delete existing cluster rows first if resuming)
        # Delete existing cluster rows for this run
        conn.execute(text("DELETE FROM clusters WHERE run_id = :rid"), {"rid": run_id})
        conn.commit()
        insert_clusters(conn, run_id, cluster_records)
        # Build album
        # Fetch images_records list for album: we need all images from DB not just newly processed
        rows = conn.execute(text("SELECT id, path, timestamp FROM images WHERE run_id = :rid"), {"rid": run_id}).mappings().all()
        all_images_records = [dict(r) for r in rows]
        # faces_records for album: list of dict for each face (we need image_id and timestamp)
        # Build list in same order as embedding DataFrame (emb_df) indices
        faces_meta = []
        for _, row in df.iterrows():
            faces_meta.append({"image_id": int(row["image_id"]), "timestamp": float(row["timestamp"]), "emb_row": int(row["emb_row"]),})
        album_rel = build_album(
            run_id=run_id,
            output_root=config.output_root,
            cluster_map=cluster_map,
            faces_records=faces_meta,
            images_records=all_images_records,
            thumbs_size=config.thumbs_size,
            thumbs_workers=config.thumbs_workers,
            thumb_regen=config.thumb_regen,
            start_month=config.start_month,
        )
        # Finish run
        n_images = len(all_images_records)
        n_faces = len(df)
        record_run_end(conn, run_id, status="done", n_images=n_images, n_faces=n_faces, album_path=album_rel)
