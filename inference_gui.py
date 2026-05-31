from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

# Reuse the single-file inference core
from inference_single import InferenceCore


APP_TITLE = "Blueberry QA – Inferenz"

def _discover_default_assets() -> Path:
    # Prefer PyInstaller one-file extraction dir
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            base = Path(getattr(sys, "_MEIPASS"))  # type: ignore[attr-defined]
            cand = base / "inference_assets"
            if cand.exists():
                return cand
    except Exception:
        pass
    # Then next to this file
    here = Path(__file__).parent
    cand = here / "inference_assets"
    if cand.exists():
        return cand
    # Finally, working dir
    cand = Path.cwd() / "inference_assets"
    return cand


@dataclass
class AppState:
    assets_dir: Path
    image_path: Optional[Path] = None
    overlay_path: Optional[Path] = None
    label_text: str = ""
    busy: bool = False
    core: Optional[InferenceCore] = None
    folder_path: Optional[Path] = None
    batch_paths: list[Path] = field(default_factory=list)


class InferenceApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(900, 650)
        try:
            # Improve text rendering on Windows
            self.tk.call("tk", "scaling", 1.25)
        except Exception:
            pass

        self.state = AppState(assets_dir=_discover_default_assets())
        self._preload_thread: Optional[threading.Thread] = None
        self._preload_target: Optional[Path] = None
        self._build_ui()
        self._update_selection_info()
        self._set_status("Bereit.")
        self._start_core_preload()

    # ------------------------------ UI Layout ------------------------------
    def _build_ui(self) -> None:
        # Top toolbar
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Bild öffnen", command=self._choose_image).pack(side=tk.LEFT)
        ttk.Button(top, text="Ordner wählen", command=self._choose_folder).pack(side=tk.LEFT, padx=(6, 0))
        self.btn_eval = ttk.Button(top, text="Bewerten", command=self._on_evaluate, state=tk.DISABLED)
        self.btn_eval.pack(side=tk.LEFT, padx=(6, 0))
        self.var_selection = tk.StringVar(value="Keine Auswahl")
        ttk.Label(top, textvariable=self.var_selection).pack(side=tk.LEFT, padx=(12, 0))

        # Center: image panel with scroll
        center = ttk.Frame(self, padding=(8, 4, 8, 4))
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center, bg="#222222", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel: info
        right = ttk.Frame(center)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        # Hinweis: Globales End-Label ausgeblendet. Es werden nur per-Instanz Labels im Overlay angezeigt.

        self.btn_save = ttk.Button(right, text="Overlay speichern…", command=self._save_overlay, state=tk.DISABLED)
        self.btn_save.pack(anchor=tk.W)

        # Overlay mode selector
        ttk.Label(right, text="Overlay").pack(anchor=tk.W, pady=(12, 0))
        self.var_mode = tk.StringVar()
        self.cmb_mode = ttk.Combobox(right, textvariable=self.var_mode, state="readonly", width=18)
        self.mode_options = [
            ("Seg", "seg"),
            ("Beere", "a1"),
            ("Never", "a2"),
            ("Red", "a3"),
            ("Green", "a4"),
        ]
        self.mode_labels = tuple(label for label, _ in self.mode_options)
        self.mode_lookup = {label: key for label, key in self.mode_options}
        self.cmb_mode["values"] = self.mode_labels
        if self.mode_labels:
            self.var_mode.set(self.mode_labels[0])
        self.cmb_mode.pack(anchor=tk.W, pady=(2, 0))
        self.cmb_mode.bind("<<ComboboxSelected>>", lambda e: self._render_current_overlay())

        ttk.Label(right, text="Fortschritt").pack(anchor=tk.W, pady=(16, 0))
        self.progress = ttk.Progressbar(right, mode="indeterminate")
        self.progress.pack(fill=tk.X)
        self.var_stage = tk.StringVar(value="–")
        ttk.Label(right, textvariable=self.var_stage, justify=tk.LEFT, wraplength=240).pack(anchor=tk.W, pady=(4, 0))

        # Status bar
        bottom = ttk.Frame(self, padding=(8, 4, 8, 8))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.var_status = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.var_status, anchor=tk.W).pack(side=tk.LEFT)

        self._update_buttons()

    # ------------------------------ Utilities -----------------------------
    def _set_status(self, text: str) -> None:
        self.var_status.set(text)
        self.update_idletasks()

    def _set_stage(self, text: str) -> None:
        self.var_stage.set(text)
        self.update_idletasks()

    def _update_selection_info(self) -> None:
        if self.state.folder_path and self.state.batch_paths:
            count = len(self.state.batch_paths)
            self.var_selection.set(
                f"Ordner: {self.state.folder_path.name} ({count} Bild{'er' if count != 1 else ''})"
            )
        elif self.state.image_path:
            self.var_selection.set(f"Bild: {self.state.image_path.name}")
        else:
            self.var_selection.set("Keine Auswahl")

    def _clear_batch_selection(self) -> None:
        self.state.folder_path = None
        self.state.batch_paths = []

    def _update_buttons(self) -> None:
        has_img = self.state.image_path is not None
        self.btn_eval.config(state=(tk.NORMAL if has_img and not self.state.busy else tk.DISABLED))
        self.btn_save.config(state=(tk.NORMAL if self.state.overlay_path and not self.state.busy else tk.DISABLED))

    def _start_core_preload(self) -> None:
        assets = self.state.assets_dir
        if not assets.exists():
            self._set_status(f"Assets-Ordner fehlt: {assets}")
            return
        if self._preload_thread and self._preload_thread.is_alive():
            if self._preload_target == assets:
                return
        self._preload_target = assets
        self._preload_thread = threading.Thread(target=self._preload_core_worker, args=(assets,), daemon=True)
        self._preload_thread.start()

    def _preload_core_worker(self, assets: Path) -> None:
        def _status(msg: str) -> None:
            self.after(0, lambda: self._set_status(msg))

        _status("Lade Modelle vor…")
        try:
            core = InferenceCore(assets)
        except Exception as exc:
            def _warn() -> None:
                if str(self.state.assets_dir) != str(assets):
                    return
                if not self.state.busy:
                    self._set_status("Vorab-Laden fehlgeschlagen.")
                messagebox.showwarning("Vorab-Laden fehlgeschlagen", f"Modelle konnten nicht geladen werden:\n{exc}")

            self.after(0, _warn)
            return

        def _assign() -> None:
            if str(self.state.assets_dir) != str(assets):
                return
            self.state.core = core
            if not self.state.busy:
                self._set_status("Modelle vorgeladen.")

        self.after(0, _assign)

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Bilder", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("Alle Dateien", "*.*")],
        )
        if not path:
            return
        self._clear_batch_selection()
        self.state.image_path = Path(path)
        self.state.overlay_path = None
        self._render_image(self.state.image_path)
        self._set_status(str(self.state.image_path))
        self._update_selection_info()
        self._update_buttons()

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory(title="Ordner mit Bildern auswählen")
        if not path:
            return
        folder = Path(path)
        supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported_ext])
        if not images:
            messagebox.showinfo("Keine Bilder gefunden", "Dieser Ordner enthält keine unterstützten Bildformate.")
            return
        self.state.folder_path = folder
        self.state.batch_paths = images
        self.state.image_path = images[0]
        self.state.overlay_path = None
        self._render_image(self.state.image_path)
        self._set_status(f"{len(images)} Bilder in {folder}")
        self._update_selection_info()
        self._update_buttons()

    def _render_image(self, path: Path) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Fehler", f"Bild konnte nicht geladen werden:\n{exc}")
            return
        self._render_pil(img)

    def _render_pil(self, img: Image.Image) -> None:
        # Fit into canvas
        cw = max(200, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        scale = max(0.05, min(1.0, scale))
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        img_resized = img.resize(new_size, Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

    def _save_overlay(self) -> None:
        # Build current overlay from cache
        if not self.state.core:
            messagebox.showinfo("Info", "Kein Overlay zum Speichern vorhanden.")
            return
        mode = self._current_mode_key()
        img = self.state.core.get_overlay(mode)
        if img is None:
            messagebox.showinfo("Info", "Kein Overlay zum Speichern vorhanden.")
            return
        initial = None
        if self.state.image_path:
            label = self.var_mode.get()
            safe_label = label if label else mode
            initial = f"{self.state.image_path.stem}_{safe_label}.png"
        dst = filedialog.asksaveasfilename(
            title="Overlay speichern",
            defaultextension=".png",
            initialfile=initial or "overlay.png",
            filetypes=[("PNG", "*.png"), ("Alle Dateien", "*.*")],
        )
        if not dst:
            return
        try:
            img.save(dst)
            self._set_status(f"Overlay gespeichert: {dst}")
        except Exception as exc:
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{exc}")

    # ------------------------------ Inference ------------------------------
    def _on_evaluate(self) -> None:
        if self.state.busy:
            return
        targets: list[Path] = []
        if self.state.batch_paths:
            targets = list(self.state.batch_paths)
        elif self.state.image_path:
            targets = [self.state.image_path]
        if not targets:
            return
        # Ensure assets dir exists
        assets_dir = self.state.assets_dir
        if not assets_dir.exists():
            messagebox.showerror("Fehler", f"Assets-Ordner existiert nicht:\n{assets_dir}")
            return

        if len(targets) > 1:
            self._set_status(f"Bewerte {len(targets)} Bilder…")
        else:
            self._set_status(str(targets[0]))

        # Start worker thread
        self.state.busy = True
        self._update_buttons()
        self.progress.start(12)
        self._set_stage("Initialisiere Modelle…")
        t = threading.Thread(target=self._worker_eval, args=(targets,), daemon=True)
        t.start()

    def _worker_eval(self, targets: list[Path]) -> None:
        try:
            # Lazy-init core (reload if assets path changed)
            if self.state.core is None or str(self.state.core.assets_root) != str(self.state.assets_dir):
                self.state.core = InferenceCore(self.state.assets_dir)
            total = len(targets)
            t_start = time.time()
            for idx, img_path in enumerate(targets, start=1):
                self.after(0, lambda p=img_path, i=idx: self._set_stage(f"Bild {i}/{total}: Lade {p.name}…"))
                def progress(text: str, i: int = idx) -> None:
                    self.after(0, lambda: self._set_stage(f"Bild {i}/{total}: {text}"))

                def _overlay_update(image):
                    # Schedule drawing the incremental overlay on the main thread
                    self.after(0, lambda img=image: self._render_pil(img))

                overlay, _meta = self.state.core.run_on_image(
                    img_path, margin=0.15, top_k=50, progress_cb=progress, overlay_cb=_overlay_update
                )

                # Save final overlay next to image (temporary) and display by mode
                out_path = img_path.with_name(img_path.stem + "_bewertet.png")
                overlay.save(out_path)
                # Update UI on main thread for this image
                def _after_image(img: Path = img_path, out: Path = out_path, index: int = idx, total_count: int = total) -> None:
                    self.state.image_path = img
                    self.state.overlay_path = out
                    self._render_current_overlay()
                    self._set_status(f"Bewertet: {img.name} ({index}/{total_count})")
                    self._update_selection_info()
                    self._update_buttons()

                self.after(0, _after_image)

            dt_total = time.time() - t_start

            # Update UI on main thread after batch
            def _done() -> None:
                self._set_status(f"Fertig ({total} Bild{'er' if total != 1 else ''}) in {dt_total:.2f}s")
                self._set_stage("Bereit.")
                self.progress.stop()
                self.state.busy = False
                self._update_buttons()

            self.after(0, _done)
        except Exception as exc:
            def _err() -> None:
                self.progress.stop()
                self.state.busy = False
                self._update_buttons()
                messagebox.showerror("Fehler bei der Inferenz", str(exc))
                self._set_stage("Fehler.")
            self.after(0, _err)

    @staticmethod
    def _label_color(label: str) -> str:
        mapping = {
            "unbekannt": "#999999",
            "Never": "#cc3333",
            "Red": "#ff0000",
            "Yellow": "#e0a800",
            "Green": "#16a34a",
        }
        return mapping.get(label, "#00aa00")

    def _render_current_overlay(self) -> None:
        if not self.state.core:
            return
        mode = self._current_mode_key()
        img = self.state.core.get_overlay(mode)
        if img is None:
            # Fallback to last saved overlay file if present
            if self.state.overlay_path and Path(self.state.overlay_path).exists():
                self._render_image(Path(self.state.overlay_path))
            return
        self._render_pil(img)

    def _current_mode_key(self) -> str:
        label = self.var_mode.get()
        return self.mode_lookup.get(label, "final")


def main() -> int:
    app = InferenceApp()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
