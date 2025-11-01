"""
Image scanning and metadata extraction.

This module provides utility functions to iterate over all image files in a
directory tree, read basic metadata (dimensions, EXIF timestamps) and
optionally compute perceptual hashes to identify duplicates.  It is
intentionally kept decoupled from the detection/embedding logic so it can
be reused in other contexts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Dict, Any

from PIL import Image, ExifTags
import imagehash


# Map EXIF tag names to their numerical IDs
_DATETIME_TAG = None
for k, v in ExifTags.TAGS.items():
    if v == "DateTimeOriginal":
        _DATETIME_TAG = k
        break


def iter_image_paths(root: Path) -> Iterator[Path]:
    """Yield all files under ``root`` that have an image‑like extension."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                yield Path(dirpath) / fn


def read_image_metadata(path: Path, compute_phash: bool = False) -> Optional[Tuple[float, Optional[str], int, int, Optional[str]]]:
    """Read metadata for an image.

    Returns a tuple ``(timestamp, exif_datetime, width, height, phash)`` or
    ``None`` if the file cannot be opened.  The timestamp is taken from
    the EXIF ``DateTimeOriginal`` tag if available, otherwise from the
    file's modification time.  The perceptual hash is a hexadecimal string
    computed with imagehash; it is optional and only computed when
    ``compute_phash`` is ``True``.
    """
    try:
        with Image.open(path) as im:
            width, height = im.size
            exif_datetime = None
            timestamp = None
            if hasattr(im, "_getexif"):
                exif = im._getexif() or {}
                if _DATETIME_TAG and _DATETIME_TAG in exif:
                    exif_datetime = str(exif[_DATETIME_TAG])
            # Fallback timestamp from file stat
            stat_ts = path.stat().st_mtime
            if exif_datetime:
                # Parse EXIF datetime into timestamp (assume local timezone, not critical)
                try:
                    import time
                    ts = time.strptime(exif_datetime, "%Y:%m:%d %H:%M:%S")
                    timestamp = time.mktime(ts)
                except Exception:
                    timestamp = stat_ts
            else:
                timestamp = stat_ts
            phash = None
            if compute_phash:
                # Downscale for performance
                hash_value = imagehash.phash(im)
                phash = str(hash_value)
            return timestamp, exif_datetime, width, height, phash
    except Exception:
        return None


def scan_images(root: Path, use_phash: bool = False) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Iterate over image files and yield their metadata.

    This helper drives the high‑level pipeline: it walks the directory tree
    under ``root``, reads EXIF/dimension/hash information for each file and
    yields a tuple ``(path, meta)``.  The ``meta`` dictionary contains
    ``timestamp``, ``exif_datetime``, ``width``, ``height``, and
    optionally ``phash`` if ``use_phash`` is true.

    Files that cannot be opened as images are silently skipped.
    """
    seen_hashes = set() if use_phash else None
    for path in iter_image_paths(root):
        meta = read_image_metadata(path, compute_phash=use_phash)
        if meta is None:
            continue
        timestamp, exif_datetime, width, height, phash = meta
        # Skip duplicate photos if using perceptual hashes
        if use_phash and phash:
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)
        yield path, {
            "timestamp": float(timestamp),
            "exif_datetime": exif_datetime,
            "width": width,
            "height": height,
            "phash": phash,
        }