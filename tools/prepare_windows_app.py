"""Create the Windows app folder structure under a given destination.

This script copies the launch scripts, inference code and assets into the
expected layout::

    <dest>/  # typically "Projekt zur Bewertung von Heidelbeeren"
        run_windows.bat
        run_windows_no_install.bat
        Thresholds.json
        nicht_anfassen/
            inference_gui.py
            inference_single.py
            inference_assets/...

Optionally a virtual environment can be copied as well.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import os

try:
    from .build_thresholds import build as build_thresholds
except ImportError:
    from build_thresholds import build as build_thresholds  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT_NAME = "Projekt zur Bewertung von Heidelbeeren"
LEGACY_ROOT_NAMES = {"berries2_app", "berries2-app", "berries2.0-app"}
LEGACY_ROOT_NAMES_LOWER = {name.lower() for name in LEGACY_ROOT_NAMES}


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Quelle fehlt: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, symlinks: bool = False, ignore=None) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Quelle fehlt: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=symlinks, ignore=ignore)


def _is_windows_mount_path(p: Path) -> bool:
    # Heuristik: WSL/Unix Pfad wie /mnt/c/...
    try:
        parts = p.resolve().parts
    except Exception:
        parts = p.parts
    return len(parts) >= 3 and parts[0] == os.sep and parts[1] == "mnt" and len(parts[2]) == 1


def _venv_looks_posix(venv: Path) -> bool:
    return (venv / "bin" / "python").exists() or (venv / "bin" / "activate").exists()


def _venv_looks_windows(venv: Path) -> bool:
    return (venv / "Scripts" / "python.exe").exists() or (venv / "Scripts" / "activate.bat").exists()


def _normalize_dest_path(dest: Path) -> Path:
    """Replace legacy folder names (e.g. berries2_app) with the new app root."""
    lower_name = dest.name.lower()
    if lower_name in LEGACY_ROOT_NAMES_LOWER:
        return dest.with_name(APP_ROOT_NAME)
    return dest


def prepare(dest: Path, *, copy_venv: bool, venv_path: Path, clean_dest: bool, force_venv_copy: bool = False) -> None:
    if clean_dest and dest.exists():
        try:
            shutil.rmtree(dest)
        except OSError as exc:
            # Unter WSL kann das Entfernen einzelner Dateien auf /mnt/c scheitern,
            # z.B. wenn eine Windows-Anwendung (oder Antivirus) Dateien wie cv2.pyd
            # noch geöffnet hat. In diesem Fall überspringen wir das harte Löschen
            # und kopieren die App-Inhalte einfach über das bestehende Verzeichnis.
            print(
                f"Warnung: Zielordner {dest} konnte nicht vollständig gelöscht werden ("  # noqa: E501
                f"{exc}). Bestehende Dateien bleiben bestehen; die App-Struktur wird "
                "darüber kopiert. Falls eine alte venv enthalten ist, kann sie unter "
                "Windows manuell entfernt werden."
            )
    dest.mkdir(parents=True, exist_ok=True)

    # Ensure Thresholds.json is up-to-date
    thresholds_tmp = ROOT / "build" / "thresholds" / "Thresholds.json"
    build_thresholds(thresholds_tmp)

    # Top-level files
    copy_file(ROOT / "run_windows.bat", dest / "run_windows.bat")
    copy_file(ROOT / "run_windows_no_install.bat", dest / "run_windows_no_install.bat")
    copy_file(thresholds_tmp, dest / "Thresholds.json")
    # Windows-spezifische README in den Zielordner legen
    readme_src = ROOT / "windows_app_README.md"
    if readme_src.exists():
        copy_file(readme_src, dest / "README.md")

    # Nested structure
    app_dir = dest / "nicht_anfassen"
    app_dir.mkdir(parents=True, exist_ok=True)
    copy_file(ROOT / "inference_gui.py", app_dir / "inference_gui.py")
    copy_file(ROOT / "inference_single.py", app_dir / "inference_single.py")

    copy_tree(ROOT / "inference_assets", app_dir / "inference_assets")

    if copy_venv:
        if not venv_path.exists():
            raise FileNotFoundError(f"Virtuelle Umgebung nicht gefunden: {venv_path}")

        # Verhindere den (sehr langsamen) Versuch, eine Linux-venv nach Windows zu kopieren
        dest_is_windows = _is_windows_mount_path(dest)
        venv_is_posix = _venv_looks_posix(venv_path)
        venv_is_windows = _venv_looks_windows(venv_path)

        if dest_is_windows and venv_is_posix and not force_venv_copy:
            raise RuntimeError(
                "Unerwartete venv-Kombination: Quelle ist eine Linux/POSIX-venv, Ziel liegt auf Windows.\n"
                "Das Kopieren waere extrem langsam und die Umgebung auf Windows ohnehin unbrauchbar.\n"
                "Erzeuge stattdessen eine Windows-venv (z.B. in PowerShell: 'py -3 -m venv .venv') und\n"
                "uebergib deren Pfad mit --venv-path. Alternativ ohne venv: --with-venv weglassen.\n"
                "Falls du es trotzdem erzwingen willst: --force-venv-copy verwenden."
            )

        # Kleinere Optimierung: offensichtliche Caches nicht mitkopieren
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache")

        # Wenn die venv ein lib64->lib Symlink enthaelt, vermeiden wir Doppelkopien, indem
        # wir Symlinks folgen (Standard) und lib64 explizit ignorieren.
        lib64 = venv_path / "lib64"
        if lib64.is_symlink() and (venv_path / "lib").exists():
            def ignore_func(src, names):
                if Path(src) == venv_path:
                    return set(["lib64"]).union(ignore("", names) or set())
                ignored = ignore(src, names)
                return set() if ignored is None else set(ignored)
            copy_tree(venv_path, app_dir / ".venv", ignore=ignore_func)
        else:
            copy_tree(venv_path, app_dir / ".venv", ignore=ignore)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erzeuge Windows-App-Struktur.")
    parser.add_argument(
        "--dest",
        required=True,
        help='Zielpfad (z.B. "/mnt/c/Users/<Name>/Downloads/Projekt zur Bewertung von Heidelbeeren")',
    )
    parser.add_argument(
        "--with-venv",
        action="store_true",
        help="Kopiert zusätzlich die virtuelle Umgebung (Standard: .venv).",
    )
    parser.add_argument(
        "--venv-path",
        default=".venv",
        help="Quelle der virtuellen Umgebung (relativ zum Repo oder absolut).",
    )
    parser.add_argument(
        "--clean-dest",
        action="store_true",
        help="Löscht das Zielverzeichnis vorab vollständig.",
    )
    parser.add_argument(
        "--force-venv-copy",
        action="store_true",
        help="Erzwingt das Kopieren der venv auch bei unguenstigen Kombinationen (nicht empfohlen).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dest = Path(args.dest).expanduser()
    if not dest.is_absolute():
        dest = (Path.cwd() / dest).resolve()
    normalized_dest = _normalize_dest_path(dest)
    if normalized_dest != dest:
        print(f'Hinweis: Zielordner wurde auf "{APP_ROOT_NAME}" angepasst.')
    dest = normalized_dest

    venv_path = Path(args.venv_path)
    if not venv_path.is_absolute():
        venv_path = (ROOT / venv_path).resolve()

    prepare(
        dest,
        copy_venv=args.with_venv,
        venv_path=venv_path,
        clean_dest=args.clean_dest,
        force_venv_copy=args.force_venv_copy,
    )
    print(f"Windows-App nach {dest} kopiert.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
