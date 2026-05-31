"""Lightweight checks to ensure the Windows bundle layout stays intact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def verify(bundle: Path) -> None:
    required = [
        "run_windows.bat",
        "run_windows_no_install.bat",
        "Thresholds.json",
        "nicht_anfassen/inference_gui.py",
        "nicht_anfassen/inference_single.py",
        "nicht_anfassen/inference_assets/manifest.json",
    ]
    missing = [p for p in required if not (bundle / p).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle unvollständig, folgende Dateien fehlen: {', '.join(missing)}")

    assets_dir = bundle / "nicht_anfassen" / "inference_assets"
    model_dir = assets_dir / "models"
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise FileNotFoundError("Keine Modelle unter nicht_anfassen/inference_assets/models gefunden.")

    for script_name in ("run_windows.bat", "run_windows_no_install.bat"):
        text = (bundle / script_name).read_text(encoding="utf-8", errors="ignore")
        if "inference_gui.py" not in text:
            raise RuntimeError(f"{script_name} verweist nicht auf inference_gui.py.")
        if "nicht_anfassen" not in text:
            raise RuntimeError(f"{script_name} verweist nicht auf das nicht_anfassen-Verzeichnis.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-Test für das Windows-Bundle.")
    parser.add_argument("--bundle", required=True, help="Pfad zum Bundle (Ausgabe von prepare_windows_app).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bundle = Path(args.bundle).expanduser()
    if not bundle.is_absolute():
        bundle = (Path.cwd() / bundle).resolve()
    verify(bundle)
    print(f"Bundle {bundle} geprüft.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
