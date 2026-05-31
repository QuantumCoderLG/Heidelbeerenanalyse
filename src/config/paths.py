from __future__ import annotations

from pathlib import Path


def detect_project_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    base = here.parent if here.is_file() else here
    for parent in [base] + list(base.parents):
        try:
            if parent.name.startswith("berries") and (parent / "src").is_dir():
                return parent.parent if parent.parent.exists() else parent
            for child in parent.iterdir():
                if child.is_dir() and child.name.startswith("berries") and (child / "src").is_dir():
                    return parent
            if (parent / "src").is_dir() and (parent / "data").is_dir():
                return parent
        except (PermissionError, NotADirectoryError):
            continue
    parents = list(base.parents)
    return parents[1] if len(parents) >= 2 else base


def project_root() -> Path:
    return detect_project_root()


__all__ = ["project_root", "detect_project_root"]
