from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOT_PATH = ROOT / "docs" / "windows_gui_inference_flowchart.dot"
PNG_PATH = ROOT / "docs" / "windows_gui_inference_flowchart.png"
SVG_PATH = ROOT / "docs" / "windows_gui_inference_flowchart.svg"
PNG_DPI = 192


def run_dot(*args: str) -> None:
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        raise SystemExit("Graphviz 'dot' was not found in PATH.")
    subprocess.run([dot_binary, *args], check=True)


def main() -> None:
    if not DOT_PATH.exists():
        raise SystemExit(f"Missing DOT source: {DOT_PATH}")

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # SVG stays sharp when zooming; PNG remains as a higher-resolution fallback.
    run_dot("-Tsvg", str(DOT_PATH), "-o", str(SVG_PATH))
    run_dot(f"-Gdpi={PNG_DPI}", "-Tpng", str(DOT_PATH), "-o", str(PNG_PATH))

    print(SVG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
