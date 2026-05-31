from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

# Use unsigned 32-bit space for compatibility with uint32 masks and SAM 2.1
# This provides ~4 billion unique IDs which is more than sufficient
_MAX_UINT32 = 0xFFFFFFFF


def _sha256_uint32(text: str) -> int:
    """
    Convert a string to a deterministic integer via SHA256, reduced into 32-bit unsigned range.
    
    Using SHA256 instead of MD5 for better collision resistance,
    especially important when reducing to 32-bit space.

    Parameters
    ----------
    text:
        Input text to hash.

    Returns
    -------
    int
        Deterministic non-negative integer in [0, 2^32-1].
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Take first 4 bytes for uint32 (using more bytes with XOR for better distribution)
    value = 0
    for i in range(0, len(digest), 4):
        chunk = int.from_bytes(digest[i:i+4], byteorder="big", signed=False)
        value ^= chunk
    return value & _MAX_UINT32


def stable_image_id(image_stem: str) -> int:
    """
    Compute a deterministic image ID based on an image stem (filename without extension).

    Notes
    -----
    * Case-insensitive: the stem is lower-cased before hashing.
    * Uses SHA256 → uint32 for better collision resistance in 32-bit space.
    * Collisions are extremely unlikely but theoretically possible with any hash.

    Parameters
    ----------
    image_stem:
        Filename without extension, e.g. "IMG_0001".

    Returns
    -------
    int
        Stable, deterministic image identifier in [0, 2^32-1].
    """
    if not image_stem or not isinstance(image_stem, str):
        raise ValueError("image_stem must be a non-empty string.")
    stem = image_stem.strip().lower()
    image_id = _sha256_uint32(stem)
    logger.debug("Computed stable image id %d for stem '%s'", image_id, image_stem)
    return image_id


def stable_annotation_id(image_id: int, index: int) -> int:
    """
    Compute a deterministic annotation ID from an image ID and a per-image index.

    Parameters
    ----------
    image_id:
        The (stable) image identifier.
    index:
        Zero-based index of the annotation within the image.

    Returns
    -------
    int
        Stable, deterministic annotation identifier in [0, 2^32-1].

    Raises
    ------
    ValueError
        If index is negative or image_id is out of range.
    """
    if index < 0:
        raise ValueError("index must be non-negative.")
    if image_id < 0 or image_id > _MAX_UINT32:
        raise ValueError(f"image_id must be in range [0, {_MAX_UINT32}]")
    # Compose a unique string and hash it.
    token = f"{image_id}:{index}"
    ann_id = _sha256_uint32(token)
    logger.debug("Computed stable annotation id %d for token '%s'", ann_id, token)
    return ann_id
