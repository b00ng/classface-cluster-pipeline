"""
Thumbnail generation and caching.

To speed up album rendering and reduce bandwidth, we generate resized JPEG
thumbnails for each original image.  Thumbnails preserve aspect ratio and
are saved alongside the album output.  This module provides functions for
generating thumbnails in parallel and skipping already cached files.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Iterable, Tuple, List

from PIL import Image


def _generate_thumbnail(src: Path, dst: Path, max_size: int) -> None:
    """Generate a thumbnail JPEG for a single image."""
    try:
        with Image.open(src) as im:
            im.thumbnail((max_size, max_size), Image.ANTIALIAS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=85, optimize=True, progressive=True)
    except Exception:
        # If thumbnail fails (e.g. unsupported image), skip silently
        pass


def build_thumbnails(items: Iterable[Tuple[Path, Path]], max_size: int = 420,
                     workers: int = 4, regen: bool = False) -> None:
    """Generate thumbnails for a collection of source/destination path pairs.

    Parameters
    ----------
    items: iterable of (src, dst)
        Each tuple defines the original image path and the path where the
        thumbnail should be stored.  It is assumed that src paths exist.
    max_size: int
        Maximum dimension (width or height) of the generated thumbnail.
    workers: int
        Number of worker processes used for parallel thumbnail generation.
    regen: bool
        If False (default), skip creating thumbnails for which the dst file
        already exists.  If True, regenerate thumbnails unconditionally.
    """
    def work(pair: Tuple[Path, Path]) -> None:
        src, dst = pair
        if not regen and dst.exists():
            return
        _generate_thumbnail(src, dst, max_size)

    # Convert items to list to avoid issues with multiple iteration
    pairs = list(items)
    if not pairs:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(work, pairs))