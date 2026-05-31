"""Generate Thresholds.json from the editable base template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "configs" / "thresholds_base.json"
DEFAULT_OUTPUT_PATH = ROOT / "build" / "thresholds" / "Thresholds.json"


def build(dest: Path = DEFAULT_OUTPUT_PATH) -> Path:
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Basisdatei nicht gefunden: {BASE_PATH}")

    with BASE_PATH.open("r", encoding="utf-8") as fh:
        base = json.load(fh)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(base, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return dest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erzeuge Thresholds.json aus den Klassik-Schwellen.")
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Zieldatei für die KI-Schwellen (Standard: build/thresholds/Thresholds.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = (Path.cwd() / dest).resolve()
    dest_path = build(dest)
    print(f"Thresholds nach {dest_path} geschrieben.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
