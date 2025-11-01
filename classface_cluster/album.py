"""
HTML album generation.

This module builds minimal static HTML pages for browsing the results of a
pipeline run.  For each student cluster, it creates a directory with
hard‑linked copies of the original photos, organised by academic year, and
generates a simple index page showing thumbnail grids.  A class‑level
index lists all students with a sample thumbnail and cluster size.
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

from datetime import datetime

from .thumbnails import build_thumbnails


def _academic_year(ts: float, start_month: int) -> str:
    """Return an academic year string (e.g. "2019-2020") for a timestamp."""
    dt = datetime.utcfromtimestamp(ts)
    year_start = dt.year if dt.month >= start_month else dt.year - 1
    return f"{year_start}-{year_start + 1}"


def build_album(run_id: int, output_root: Path, cluster_map: Dict[str, List[int]],
                faces_records: List[Dict[str, any]], images_records: List[Dict[str, any]],
                thumbs_size: int = 420, thumbs_workers: int = 4, thumb_regen: bool = False,
                start_month: int = 6) -> str:
    """Generate an HTML album and return the relative path to the root index.

    Parameters
    ----------
    run_id: int
        Unique identifier of the run (used for folder names).
    output_root: Path
        Root directory for all runs; a subfolder ``run_<id>`` will be created.
    cluster_map: dict mapping label to list of face indices
        Each key is a student label and each value is a list of face indices
        referencing entries in ``faces_records``.
    faces_records: list of dict
        Metadata for each face including ``image_id`` and ``timestamp``.
    images_records: list of dict
        Metadata for each image including ``path`` and ``timestamp``.
    thumbs_size, thumbs_workers, thumb_regen: int, int, bool
        Parameters for thumbnail generation.
    start_month: int
        Month considered as the start of the academic year.

    Returns
    -------
    str
        Relative path to the generated index.html file.
    """
    run_dir = output_root / f"run_{run_id:06d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Directory for thumbnails
    thumbs_dir = run_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)
    # Build thumbnails list: (src, dst)
    thumb_pairs = []
    for img in images_records:
        src = Path(img["path"])
        dst = thumbs_dir / (src.name + ".jpg")
        thumb_pairs.append((src, dst))
    build_thumbnails(thumb_pairs, max_size=thumbs_size, workers=thumbs_workers, regen=thumb_regen)
    # Build student directories and per‑student index
    student_links = []
    for label, face_indices in cluster_map.items():
        # Determine all unique image IDs for this cluster
        img_ids = sorted({faces_records[i]["image_id"] for i in face_indices}, key=lambda i: images_records[i]["timestamp"])
        if not img_ids:
            continue
        student_dir = run_dir / label
        student_dir.mkdir(parents=True, exist_ok=True)
        # Copy images into academic year subfolders and collect HTML entries
        entries: List[str] = []
        for img_id in img_ids:
            img_rec = images_records[img_id]
            img_path = Path(img_rec["path"])
            ts = img_rec["timestamp"]
            year_folder_name = _academic_year(ts, start_month)
            dest_folder = student_dir / year_folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_path = dest_folder / img_path.name
            # Create hardlink if possible, otherwise copy
            try:
                if not dest_path.exists():
                    # Try to create a hard link; fall back to copying
                    try:
                        dest_path.hardlink_to(img_path)
                    except Exception:
                        shutil.copy2(img_path, dest_path)
            except Exception:
                # Fallback to copying on any error
                try:
                    shutil.copy2(img_path, dest_path)
                except Exception:
                    pass
            # Thumbnail path relative to run_dir
            thumb_name = img_path.name + ".jpg"
            thumb_rel = f"thumbnails/{thumb_name}"
            entry = f'<a href="{label}/{year_folder_name}/{html.escape(img_path.name)}"><img src="../{thumb_rel}" alt="{html.escape(img_path.name)}" loading="lazy" /></a>'
            entries.append(entry)
        # Write per‑student HTML
        student_index = student_dir / "index.html"
        with student_index.open("w", encoding="utf-8") as fh:
            fh.write("<html><head><meta charset='utf-8'/><title>{}</title></head><body>\n".format(html.escape(label)))
            fh.write(f"<h1>{html.escape(label)}</h1>\n")
            fh.write("<div style='display:flex;flex-wrap:wrap;gap:8px;'>\n")
            for e in entries:
                fh.write(e + "\n")
            fh.write("</div>\n")
            fh.write("</body></html>")
        # Determine sample thumbnail (first image)
        sample_thumb = Path(img_rec["path"]).name + ".jpg"
        student_links.append((label, len(img_ids), sample_thumb))
    # Write top‑level index
    index_path = run_dir / "index.html"
    with index_path.open("w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'/><title>Classface Run {}</title></head><body>\n".format(run_id))
        fh.write(f"<h1>Run {run_id:06d} – Class Roster</h1>\n")
        fh.write("<ul style='list-style:none;padding:0;'>\n")
        for label, size, thumb in sorted(student_links, key=lambda x: x[1], reverse=True):
            fh.write("<li style='margin-bottom:1em;'>\n")
            fh.write(f"<a href='{html.escape(label)}/index.html' style='text-decoration:none;color:black;'>\n")
            fh.write(f"<img src='thumbnails/{html.escape(thumb)}' style='width:64px;height:64px;object-fit:cover;margin-right:8px;vertical-align:middle;' alt='{html.escape(label)}'/>\n")
            fh.write(f"<span style='font-size:1.2em;'>{html.escape(label)} ({size} images)</span></a>\n")
            fh.write("</li>\n")
        fh.write("</ul>\n")
        fh.write("</body></html>")
    # Return relative path to the index
    return f"run_{run_id:06d}/index.html"