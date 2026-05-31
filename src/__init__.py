from __future__ import annotations

"""
berries.src package

Kern- und Orchestrierungs-Module für das Heidelbeer-Datenset.
Dieses Paket hat keine Seiteneffekte beim Import.

Enthaltene Subpakete:
- config: Konfiguration & Pfad-Helfer
- data: Datensätze, COCO-Exports & Hilfen zur Annotation
- evaluation: Auswertung, Metriken & Postprocessing
- pipelines: CLI-Pipelines rund um Datenaufbereitung
- training: Modellaufbau, Losses & Trainingsschleifen
- utils: Allgemeine Helferfunktionen
"""

import logging

# Keine vorab-Imports der Subpakete, um runpy-Warnungen zu vermeiden.
# Submodule können weiterhin explizit importiert werden, z. B.:
#   import src.training.train_segmentation
# oder
#   from src import training

__all__ = ["config", "data", "evaluation", "pipelines", "training", "utils"]

# Paketversion (Fallback, wenn nicht installiert)
__version__ = "0.0.0"

try:
    from importlib.metadata import PackageNotFoundError, version as _version  # type: ignore

    try:
        __version__ = _version("berries")
    except PackageNotFoundError:
        pass
except Exception:
    # importlib.metadata nicht verfügbar (z. B. ältere Python-Version) → bleibe bei Fallback
    pass

# Verhindere "No handler found" Warnungen, ohne Logging global zu konfigurieren.
logging.getLogger(__name__).addHandler(logging.NullHandler())
