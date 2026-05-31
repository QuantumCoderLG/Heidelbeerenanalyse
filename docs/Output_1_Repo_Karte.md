# Output 1: Repo-Karte

## Ordnerstruktur (Tree)

```
berries2.0/
├── configs/                          # Konfigurationsdateien
│   ├── backbone_a.yaml               # A1/A2-Klassifikations-Config
│   ├── backbone_a3.yaml              # A3-Farbklassifikations-Config
│   ├── thresholds_base.json          # Zentrale Entscheidungsschwellen
│   ├── train.yaml                    # Segmentierungs-Trainings-Config
│   └── suggestions_user.txt          # Nutzervorschläge für Relabeling
│
├── data/
│   ├── all_images/                   # Rohbilder (JPG + CR3), ~222 Dateien
│   │   ├── Ampel/{Yellow,Green,Never,Red}/  # Qualitätsklassen-Bilder
│   │   ├── Heidelbeeren2/            # Versuchsplatten (frisch/gefroren)
│   │   └── _split/                   # Symlinks/Kopien nach Ampel-Klassen
│   ├── BBoxes_annotation_data/       # Bounding-Box-XML (7 Dateien, „Kaputte Stelle")
│   ├── raw/
│   │   ├── annotations/              # CVAT-Polygon-XML (27 Dateien)
│   │   ├── images/                   # Zugehörige JPG-Bilder (27 Dateien, 6000×4000)
│   │   └── masks/                    # Generierte Instanzmasken (PNG)
│   ├── processed/                    # COCO-Format: train/val/test mit Masks + JSON
│   └── instance_crops/
│       ├── images/{red,yellow,green,never}/   # Einzelbeeren-Crops (1448 PNG)
│       ├── masks/{red,yellow,green,never}/    # Zugehörige Masken (1398 PNG)
│       ├── instance_masks/fold_1_best/        # Segmentierungs-Instanzmasken pro Bild
│       ├── overlays/                          # Overlay-Visualisierungen
│       ├── metadata/
│       │   ├── crops.csv                      # Haupt-Metadaten (Klasse, Fold, Pfad)
│       │   ├── crops_rejected.csv             # Abgelehnte Crops
│       │   ├── never_union_crops.jsonl       # Synthetische Never-Crops
│       │   └── summary.json                   # Pipeline-Statistik
│       ├── splits/
│       │   ├── a2/{train,val,test}.txt        # Manuelle Splits A2
│       │   └── a3/{train,val,test}.txt        # Manuelle Splits A3
│       └── rejections/                        # Abgelehnte Instanzen (z. B. border_trim_empty)
│
├── src/
│   ├── config/                       # YAML/JSON-Config-Loader
│   │   ├── __init__.py
│   │   └── paths.py
│   ├── data/                         # Datensatzklassen & Parser
│   │   ├── classification_dataset.py # CSV-basierter Klassifikations-Datensatz
│   │   ├── coco_schema.py            # COCO-Annotations-Schema
│   │   ├── ids.py                    # Instanz-ID-Verwaltung
│   │   ├── matching.py               # Ground-Truth-Matching
│   │   ├── metadata.py               # Metadaten-Verwaltung
│   │   ├── rasterize.py              # Polygon → Maske
│   │   └── xml_parser.py             # CVAT-XML-Parser
│   ├── training/                     # Trainingslogik
│   │   ├── augment.py                # Datenaugmentierung
│   │   ├── calibration.py            # Temperatur-Kalibrierung
│   │   ├── classifier_models.py      # EfficientNet-Classifier-Builder
│   │   ├── losses.py                 # DiceLoss, BinaryFocalLoss, CombinedLoss
│   │   ├── metrics_classification.py # Klassifikationsmetriken
│   │   ├── models.py                 # DeepLabV3+ Implementierung
│   │   ├── train_backbone_a.py       # Trainingsschleife Klassifikation
│   │   └── train_segmentation.py     # Trainingsschleife Segmentierung
│   ├── evaluation/                   # Evaluation & Inference
│   │   ├── apply_model_to_images.py  # Batch-Segmentierung auf Bildern
│   │   ├── classical_yellow_green.py # Regelbasiertes Yellow/Green-System
│   │   ├── cleanup_overlays.py       # Overlay-Bereinigung
│   │   ├── evaluate_segmentation.py  # Segmentierungsevaluation
│   │   ├── metrics.py                # IoU, Count-Metriken
│   │   ├── postprocessing.py         # Threshold → Morphologie → Watershed
│   │   ├── run_a1_a2_pipeline.py     # End-to-End-Pipeline A1→A2→A3
│   │   └── test_backbone_a1.py       # A1-Testskript
│   ├── pipelines/                    # Daten-ETL
│   │   ├── crop_pipeline.py          # Instanz-Crop-Extraktion
│   │   ├── generate_notberry.py      # Synthetische Negativ-Generierung
│   │   └── prepare_dataset.py        # XML → COCO-Konvertierung
│   ├── tools/                        # Werkzeuge
│   │   ├── convert_checkpoint.py     # PT → ONNX/Safetensors-Konvertierung
│   │   ├── export_classifier_onnx.py # Classifier → ONNX-Export
│   │   └── relabel_from_suggestions.py  # Interaktives Relabeling
│   └── utils/                        # Hilfsfunktionen
│       ├── box_utils.py, color_norm.py, geometry_utils.py
│       ├── image_ops.py, image_utils.py, io_utils.py
│       ├── training_utils.py, vis_utils.py
│
├── tools/                            # Standalone-Tools
│   ├── build_thresholds.py           # Threshold-Builder
│   ├── fix_gemischte_platte_labels.py  # Label-Korrektur
│   ├── generate_a2_manual_split.py   # A2-Split-Generator
│   ├── generate_a3_manual_split.py   # A3-Split-Generator
│   ├── prepare_windows_app.py        # Windows-App-Bundle-Erstellung
│   ├── sync_unlabeled_crops.py       # Unlabeled-Crop-Sync
│   └── verify_windows_bundle.py      # Bundle-Verifikation
│
├── scripts/                          # Experiment-Skripte
│   ├── prepare_union_crops.py        # Union-Crop-Erstellung
│   ├── run_a2_experiments.sh         # A2-Experiment-Runner
│   ├── run_a3_experiments.sh         # A3-Experiment-Runner
│   ├── run_a3_reproduce_best.sh      # A3-Reproduzierbarkeit
│   └── visualize_union_boxes.py      # Union-Box-Visualisierung
│
├── tests/                            # Pytest-Testsuite
│   ├── conftest.py, test_crop_pipeline.py, test_dataset.py
│   ├── test_losses.py, test_metrics.py, test_postprocessing.py
│
├── inference_assets/                 # ONNX-Modelle für Inference
│   ├── manifest.json                 # Modell-Manifest
│   └── models/                       # 5 ONNX-Dateien + Meta-JSONs
│       ├── segmentation.onnx, a1_classifier.onnx
│       ├── a2_classifier.onnx, a3_classifier.onnx, a4_classifier.onnx
│       └── *_meta.json
│
├── outputs/                          # Trainingsergebnisse
│   ├── backbone_a/{a1,a2,a3,a4}/fold_XX/  # Checkpoints + Summaries
│   ├── classical/                    # Klassik-Evaluation (metrics.json)
│   ├── checkpoints/                  # Segmentierungs-Checkpoints (5 Folds)
│   ├── logs/metrics.json             # Segmentierungs-Ergebnisse (5 Folds)
│   ├── experiment_logs/              # Experiment-Logdateien
│   ├── overlays/                     # Segmentierungs-Overlays
│   ├── pipeline_a1_a2/              # End-to-End-Pipeline-Outputs
│   └── Test/                         # Inferenz-Testbild + Label
│
├── build/Heidelbeeren-Bewertung-App/ # Windows-App-Bundle
│   └── Thresholds.json               # Konfigurierbare Schwellenwerte
│
├── Kanditaten/                       # Checkpoint-Kandidaten + ONNX-Exporte
│
├── inference_single.py               # Standalone-ONNX-Inference (1245 Zeilen)
├── inference_gui.py                  # Tkinter-GUI (405 Zeilen)
├── run_windows.bat                   # Windows Auto-Installer + Launcher
├── run_windows_no_install.bat        # Launcher ohne Installation
├── Makefile                          # Build-Targets
├── requirements.txt                  # Python-Abhängigkeiten
├── README.md                         # Entwickler-Doku
├── windows_app_README.md             # Endnutzer-Doku (Deutsch)
├── CLAUDE.md                         # Codebase-Guide
└── Agents.md                         # Aufgabenspezifikation
```

## Tabelle der wichtigsten Entry-Point-Dateien

| Datei | Zweck | Quelle |
|-------|-------|--------|
| `inference_single.py` | Standalone-ONNX-Inference: Seg → A1 → A2 → A3 → A4 → Klassik | L1–L1245 |
| `inference_gui.py` | Tkinter-GUI für Einzel-/Batch-Bildbewertung | L1–L405 |
| `src/training/train_segmentation.py` | Trainingsschleife DeepLabV3+ Segmentierung | src/training/ |
| `src/training/train_backbone_a.py` | Trainingsschleife A1/A2/A3/A4 Klassifikation | src/training/ |
| `src/evaluation/run_a1_a2_pipeline.py` | End-to-End-Evaluations-Pipeline | src/evaluation/ |
| `src/evaluation/classical_yellow_green.py` | Regelbasiertes Yellow/Green-System | src/evaluation/ |
| `src/pipelines/prepare_dataset.py` | XML → COCO-Konvertierung | src/pipelines/ |
| `src/pipelines/crop_pipeline.py` | Instanz-Crop-Extraktion aus Segmentierung | src/pipelines/ |
| `src/pipelines/generate_notberry.py` | Synthetische Negativ-Crops | src/pipelines/ |
| `src/tools/export_classifier_onnx.py` | Classifier PT → ONNX-Export | src/tools/ |
| `tools/prepare_windows_app.py` | Windows-Bundle-Erstellung | tools/ |
| `tools/build_thresholds.py` | Threshold-Aggregation | tools/ |
| `run_windows.bat` | Auto-Installer + Launcher für Windows | L1–233 |

## Config/Env/Requirements

| Datei | Typ |
|-------|-----|
| `configs/train.yaml` | Segmentierungs-Hyperparameter |
| `configs/backbone_a.yaml` | A1/A2/A4-Klassifikations-Config |
| `configs/backbone_a3.yaml` | A3-Config (EfficientNet-B5) |
| `configs/thresholds_base.json` | Zentrale Entscheidungsschwellen |
| `requirements.txt` | Python-Abhängigkeiten (PyTorch, ONNX etc.) |
| `inference_assets/manifest.json` | Modell-Manifest für Inference |
| `build/.../Thresholds.json` | Schwellenwerte im Windows-Bundle |

## Windows-Inference-Skripte

| Datei | Funktion |
|-------|----------|
| `run_windows.bat` | Prüft/installiert Python 3.11.9, erstellt venv, installiert onnxruntime + opencv + Pillow + numpy, startet GUI via `pythonw.exe` (Quelle: `run_windows.bat:L1–L233`) |
| `run_windows_no_install.bat` | Startet GUI mit bestehendem venv, keine Installation (Quelle: `run_windows_no_install.bat`) |
| `tools/prepare_windows_app.py` | Erstellt Bundle unter `build/Heidelbeeren-Bewertung-App/` (Quelle: `Makefile:L67–L68`) |
| `tools/verify_windows_bundle.py` | Verifiziert Vollständigkeit des Bundles (Quelle: `Makefile:L69`) |
