# Heidelbeer-Instanzsegmentierung

Dieses Projekt enthält die komplette Pipeline für binäre Heidelbeer-Instanzsegmentierung mit DeepLabV3+ (ResNet-50-Backbone).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datenaufbereitung

Rohdaten (`data/raw/images`, `data/raw/annotations`) werden mit `python -m src.pipelines.prepare_dataset` nach `data/processed/<split>/` exportiert. Bestehende Artefakte bleiben kompatibel.

## Training

```bash
python -m src.training.train_segmentation \
    --config configs/train.yaml  # optional
```

Wichtige CLI-Optionen (als Overrides möglich):

- `--lr-find` / `--no-lr-find`: logarithmische LR-Suche
- `--freeze-blocks layer1 layer2`: blockweises Einfrieren
- `--accum-steps 2`: Gradient Accumulation
- `--amp` / `--no-amp`: Mixed Precision
- `--config <pfad>` & `--override train.max_epochs=150`

### Cross-Validation

Aktiviere in der Config (`cv.enabled: true`, `cv.num_folds: 5`). Pro Fold werden Metriken (TensorBoard + CSV) sowie `best.pt`/`last.pt` gespeichert.

## Evaluation & Inferenz

```bash
python -m src.evaluation.evaluate_segmentation \
    --checkpoint outputs/checkpoints/fold_0_best.pt \
    --split val \
    --count-tune \
    --save-dir outputs/predictions
```

- Threshold wird aus Checkpoint oder Config geladen; optional per Count-Guided Tuning an Zielanzahl (Default 25) angepasst.
- Postprocessing: Schwellenwert-, Morphologie-, Connected-Components-, Circularity-Filter, Watershed.

### Klassische Yellow/Green-Entscheidung (ohne CNN)

Verwendet fünf geometrische Merkmale pro Crop (`relative_size`, äquivalenter Durchmesser `deq`, Circularity, Solidity, radiale Rauheit) aus `data/instance_crops/metadata/crops.csv`. Eine Beere gilt als green, sobald mindestens 4 von 5 Regeln erfüllt sind.

1) Thresholds aus grünen Beeren kalibrieren (Standard: 5.–95. Perzentil):

```bash
python -m src.evaluation.classical_yellow_green fit \
  --metadata data/instance_crops/metadata/crops.csv \
  --method quantile --q-low 0.05 --q-high 0.95 \
  --output outputs/classical/thresholds.json
```

2) Vorhersage + Auswertung gegen Ground-Truth (`yellow`/`green`):

```bash
python -m src.evaluation.classical_yellow_green predict \
  --metadata data/instance_crops/metadata/crops.csv \
  --thresholds outputs/classical/thresholds.json \
  --output-csv outputs/classical/predictions.csv \
  --output-metrics outputs/classical/metrics.json
```

Ausgabe:
- `outputs/classical/thresholds.json`: Multi-Feature-Regeln + Statistik
- `outputs/classical/predictions.csv`: Pro Beere Label (`pred_label`) + Begründung (`decision_reason`)
- `outputs/classical/metrics.json`: Konfusionsmatrix, Accuracy, Precision/Recall (yellow)
- Legacy-Dateien mit nur einem Feature werden abgelehnt – in diesem Fall bitte `fit` erneut ausführen.

### End-to-End: Segmentierung → A1 → A2 → A3 (Einzelbild)

Schnelltest für die komplette Filterkette (Segmentierung, dann A1 notberry-Filter, dann A2 never-Filter) auf einem Bild:

```bash
python -m src.evaluation.run_a1_a2_pipeline \
  --image outputs/Test/Inferenz_Test_Bild.JPG \
  --seg-checkpoint Kanditaten/fold_1_best.pt \
  --a1-checkpoint outputs/backbone_a/a1/fold_00/best.pt \
  --a2-checkpoint outputs/backbone_a/a2/fold_00/best.pt \
  --a3-checkpoint outputs/backbone_a/a3/fold_00/best.pt \
  --a3-config configs/backbone_a3.yaml \
  --amp
```

Standardpfade sind bereits so gesetzt, dass der obige Befehl auch ohne Argumente funktioniert, wenn die Dateien an den genannten Orten liegen.

Ausgabe-Struktur unter `outputs/pipeline_a1_a2/<bildname>/`:

- `00_seg/`: `overlay.png`, `instances_mask.png`
- `01_a1/`: `overlay_boxes.png`, `predictions.csv`,
  - `accepted_berry/crops/` (weitergereichte Beeren)
  - `rejected_notberry/crops/` (ausgeschiedene Nicht-Heidelbeeren)
- `02_a2/`: `overlay_boxes.png`, `predictions.csv`,
  - `accepted_candidates/crops/` (weitergereichte Kandidaten für Farbstufen)
  - `rejected_never/crops/` (ausgeschiedene "never")
- `03_a3/`: `overlay_boxes.png`, `predictions.csv`,
  - `accepted_red/crops/`
    - `rejected_not_red/crops/`
 - `meta.json`: Zusammenfassung inkl. verwendeter Thresholds


## Logging & Ausgaben

- TensorBoard unter `outputs/tensorboard`
- CSV-Logs in `outputs/logs/training.csv`
- Aggregierte Evaluationsmetriken in `outputs/logs/eval.json`

## Tests

```bash
python -m pytest
```

Enthaltene Tests decken Loss-Kombinationen, Postprocessing, Metriken und Dataset-Sanity ab.

## Backbone A (A1/A2/A3)

Training der Klassifikations-Köpfe für die Bewertungspipeline (A1: notberry, A2: never, A3: red):

```bash
# A1: notberry (1) vs. berry (0)
python -m src.training.train_backbone_a --task a1 --config configs/backbone_a.yaml

# A2: never (1) vs. {yellow, green, red} (0)
python -m src.training.train_backbone_a --task a2 --config configs/backbone_a.yaml

# A3: red (1) vs. {yellow, green} (0)
python tools/generate_a3_manual_split.py  # erzeugt data/instance_crops/splits/a3/{train,val,test}.txt
python -m src.training.train_backbone_a --task a3 --config configs/backbone_a3.yaml  # EfficientNet-B5 + Farbfeatures
```

`configs/backbone_a.yaml` erwartet manuelle Splits unter `data/instance_crops/splits/a2/`; bei Bedarf neu erzeugen:

```bash
python tools/generate_a2_manual_split.py
```

### Relabeling/Sortierung per Vorschlagsdatei

Wenn einzelne ausgeschnittene Heidelbeeren (Crops) falsch einsortiert wurden, kann eine einfache Vorschlagsdatei verwendet werden, um die Dateien zwischen den Ordnern (`yellow`, `green`, `red`, `never`) zu verschieben und die Metadaten (`data/instance_crops/metadata/crops.csv`) zu aktualisieren.

- Format der Vorschlagsdatei (Beispiel: `configs/suggestions_example.txt`):
  - Kopfzeile: abgekürzter Szenenname, z. B. `YELLOW211` für `Yellow_2_1_1`.
  - Danach pro Zeile: `Nummer <n> = <ziel>` mit `<ziel>` in `{yellow, green, red, never}` oder `ok` (keine Änderung).

Beispiel:

```
YELLOW211
Nummer 16 = never
Nummer 23 = red
```

Anwenden der Vorschläge (Dry‑Run zuerst):

```
python -m src.tools.relabel_from_suggestions configs/suggestions_example.txt --dry-run
```

Änderungen übernehmen (inkl. Backup der `crops.csv`) und optional die genutzten Gesamtbilder in einen Extra‑Ordner kopieren:

```
python -m src.tools.relabel_from_suggestions \
  configs/suggestions_example.txt \
  --backup \
  --split-sources-dir data/all_images/_split
```

Hinweise:
- Das Skript verschiebt die Crop‑Dateien (RGB + Maske) in `data/instance_crops/images/<ziel>/` bzw. `data/instance_crops/masks/<ziel>/` und passt `class_label`, `crop_path`, `mask_path` in `crops.csv` an.
- Unbekannte/uneindeutige Zeilen werden übersprungen; `ok` bedeutet „keine Änderung“.
- Die „Gesamtbilder“ werden standardmäßig nicht verschoben; mit `--split-sources-dir` können sie zusätzlich in einen Extra‑Unterordner kopiert werden.

Ausgaben & Logs:
- `outputs/backbone_a/<task>/fold_XX/` enthält `summary.json`, `best.pt`, `last.pt`.
- Hard-Negs liegen in `outputs/backbone_a/<task>/hard_negatives/fold_XX.csv` (Test-Set analog unter `hard_negatives_test`).
- Vorhersagen pro Fold landen in `outputs/backbone_a/predictions/<task>/<task>_foldXX.csv`.
- Trainingsmetriken werden an `outputs/backbone_a/training_log_<task>.csv` bzw. `outputs/backbone_a/metrics.csv` angehängt.

### Schneller End-to-End-Test (Segmentierung + A1 Klassifikation) für ein Bild:

```bash
python -m src.evaluation.test_backbone_a1 \
  --image data/all_images/Beispiel.jpg \
  --seg-checkpoint outputs/checkpoints/fold_0_best.pt \
  --clf-checkpoint outputs/backbone_a/a1/fold_00/best.pt
```

## Windows-Bundle bauen

Für Tester:innen unter Windows gibt es keine eingecheckte Kopie mehr. Stattdessen erzeugst du mithilfe der bestehenden Quellen jederzeit ein frisches Paket:

```bash
make windows-dist  # schreibt nach build/Heidelbeeren-Bewertung-App
```

Der Make-Task ruft `python -m tools.prepare_windows_app --dest build/Heidelbeeren-Bewertung-App` auf, liest `configs/thresholds_base.json` (enthält KI‑ und Klassik-Schwellen) und lässt `python tools/build_thresholds.py` die vollständige Datei bauen. Anschließend prüft `python -m tools.verify_windows_bundle`, ob `run_windows*.bat`, `Thresholds.json` und die Python-Dateien korrekt unter `build/Heidelbeeren-Bewertung-App/nicht_anfassen/` liegen. Das Ergebnis: Alle Schwellen (inklusive `classical_rules`) liegen zentral in `build/Heidelbeeren-Bewertung-App/Thresholds.json` und müssen nicht mehr in Unterordnern gesucht werden.

Falls du nur die Schwellen neu berechnen möchtest (z. B. nach einem Update der klassischen Regeln), verwende:

```bash
python tools/build_thresholds.py
```

Das Skript liest `configs/thresholds_base.json` und schreibt alle Schwellen (inkl. `classical_rules`) nach `build/thresholds/Thresholds.json`. Änderungen nimmst du daher ausschließlich an `configs/thresholds_base.json` vor; die Windows-App verwendet anschließend genau diese `Thresholds.json` im App-Hauptordner.

## Nicht versionierte Artefakte

Um das Repo schlank zu halten, werden virtuelle Umgebungen, generierte Daten und Windows-Builds nicht eingecheckt (`.venv/`, `build/`, `outputs/`, `data/processed/`, `dist_windows_min/`). Die README beschreibt sämtliche Befehle, um diese Artefakte bei Bedarf neu zu erzeugen.

## Veröffentlichung und Forschungsdaten

Quellcode liegt im GitHub-Repository. Große Forschungsdaten, Modellgewichte und
fertige Windows-Pakete werden separat über Zenodo beziehungsweise GitHub
Releases veröffentlicht. Hinweise zur Erstellung der Upload-Archive stehen in
[`docs/ZENODO_UPLOAD.md`](docs/ZENODO_UPLOAD.md).

Nach einem Quellcode-Clone fehlen die großen ONNX-Laufzeitmodelle bewusst.
Für Windows-Nutzung das fertige Release-Asset
`Heidelbeeren-Bewertung-App-v1.0.0.zip` verwenden. Für direkte Python-Nutzung
`blueberry-models-v1.0.0.zip` aus dem Zenodo-Software-Record herunterladen und
so entpacken, dass `inference_assets/models/` vorhanden ist.

Ausführliche Veröffentlichungsschritte stehen in
[`docs/PUBLISHING_GUIDE.md`](docs/PUBLISHING_GUIDE.md).

Zenodo-Records:

- Dataset DOI: `10.5281/zenodo.20479053`
- Software/Model DOI: `10.5281/zenodo.20479124`
