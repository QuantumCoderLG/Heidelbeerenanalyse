from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps


def load_image(path: Path) -> Tuple["np.ndarray", int, int]:
    """
    Load an image from disk.

    Returns
    -------
    (np.ndarray, width, height)
        Image as a numpy array (RGB, dtype=uint8) with shape (H, W, 3), and its width/height.
    """
    if not isinstance(path, (str, Path)):
        raise ValueError("path must be a string or pathlib.Path")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    with Image.open(p) as im:
        # Honor EXIF orientation so that the pixel array matches what
        # annotators saw in labeling tools/viewers. This fixes rotated overlays
        # when some cameras store orientation only in EXIF.
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            # If EXIF is missing or Pillow can't transpose, proceed as-is.
            pass
        im = im.convert("RGB")
        width, height = im.size
        arr = np.asarray(im, dtype=np.uint8)
    return arr, width, height


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory of the given path exists.
    """
    p = Path(path)
    parent = p.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def atomic_save_pil_image(image: Image.Image, path: Path, **save_kwargs) -> None:
    """
    Atomically save a PIL image by writing to a temporary file and moving it into place.

    Parameters
    ----------
    image:
        PIL.Image.Image to save.
    path:
        Destination file path.
    save_kwargs:
        Additional keyword arguments passed to PIL.Image.Image.save().
        For PNGs, you may pass 'pnginfo' to embed metadata.
    """
    p = Path(path)
    ensure_parent_dir(p)
    suffix = "".join(p.suffixes) or ".png"
    tmp_dir = p.parent
    
    # Use context manager for proper file descriptor cleanup
    fd = None
    tmp_path = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=tmp_dir)
        # Close the file descriptor immediately as PIL will open its own
        os.close(fd)
        fd = None  # Mark as closed
        
        tmp_path = Path(tmp_name)
        image.save(tmp_path, **save_kwargs)
        
        # Ensure the temp file is properly flushed to disk before atomic rename
        # This is important on some filesystems
        tmp_path.touch()
        
        # os.replace is atomic on POSIX and Windows (same filesystem).
        os.replace(tmp_path, p)
        tmp_path = None  # Mark as successfully moved
        
    except Exception as e:
        # Clean up file descriptor if still open
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        
        # Clean up temporary file if it exists and wasn't moved
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as cleanup_err:
                # Log cleanup failure but re-raise original exception
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to clean up temporary file %s: %s", tmp_path, cleanup_err
                )
        
        # Re-raise the original exception
        raise e
