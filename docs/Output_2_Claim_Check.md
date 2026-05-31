# Output 2: Claim-Check (Verifizierte Faktenliste)

| # | Claim | Quelle |
|---|-------|--------|
| 1 | Seed für alle Trainings: 1337 | `configs/train.yaml:L1`, `configs/backbone_a.yaml:L1`, `configs/backbone_a3.yaml:L1` |
| 2 | Segmentierungsmodell: DeepLabV3+ mit ResNet-50 Backbone | `configs/train.yaml:L20–L21`, `src/training/models.py:L67–L74` |
| 3 | Segmentierung: vortrainierte ImageNet1K_V2-Gewichte | `src/training/models.py:L75` |
| 4 | Segmentierungs-Eingabegröße: 1024 × 1024 | `configs/train.yaml:L7–L8`, `inference_assets/manifest.json:L7–L10` |
| 5 | Segmentierungs-Batch-Size: 10 | `configs/train.yaml:L10` |
| 6 | Segmentierungs-Optimizer: Adam, lr = 0.0003 | `configs/train.yaml:L36–L37` |
| 7 | Backbone-LR-Skalierung: 0.1× | `configs/train.yaml:L41–L42` |
| 8 | Segmentierungs-Scheduler: Poly, power = 0.9 | `configs/train.yaml:L48–L50` |
| 9 | Segmentierungs-Loss: dice_bce (Dice + BCE) | `configs/train.yaml:L30` |
| 10 | Segmentierungs-Epochen: max 120, Early Stopping patience = 20 | `configs/train.yaml:L53`, `configs/train.yaml:L79–L80` |
| 11 | 5-Fold-Kreuzvalidierung für Segmentierung | `configs/train.yaml:L111–L112` |
| 12 | Segmentierungs-Output-Stride: 8 | `configs/train.yaml:L23` |
| 13 | ASPP-Dilatationsraten: [6, 12, 18] | `configs/train.yaml:L26` |
| 14 | Postprocessing: Threshold = 0.5, Morphologie open = 3 / close = 5, min_area = 30, circularity_min = 0.25, Watershed enabled | `configs/train.yaml:L84–L100`, `inference_assets/manifest.json:L228–L243` |
| 15 | Beste Segmentierungs-IoU: 0.979 (Fold 0, Epoch 73) | `outputs/logs/metrics.json:L3–L6` |
| 16 | Schlechteste Segmentierungs-IoU: 0.973 (Fold 3) | `outputs/logs/metrics.json:L18–L20` |
| 17 | Alle Segmentierungs-Folds verwenden Threshold 0.3 | `outputs/logs/metrics.json:L6,L11,L15,L20,L25` |
| 18 | A1-Klassifikator: EfficientNet-B1, notberry vs. berry | `configs/backbone_a.yaml:L68`, `configs/thresholds_base.json:L19–L24` |
| 19 | A2-Klassifikator: EfficientNet-B1, never vs. ok | `configs/backbone_a.yaml:L68`, `configs/thresholds_base.json:L25–L30` |
| 20 | A3-Klassifikator: EfficientNet-B5, red vs. not-red | `configs/backbone_a3.yaml:L41`, `configs/thresholds_base.json:L31–L36` |
| 21 | A4-Klassifikator: EfficientNet-B1, green vs. yellow | `configs/backbone_a.yaml:L68`, `configs/thresholds_base.json:L37–L42` |
| 22 | Klassifikations-Eingabegröße: 384 × 384 (Training), 320 × 320 (Inference A1/A3/A4) | `configs/backbone_a.yaml:L42`, `inference_assets/manifest.json:L14–L18,L30–L34,L38–L42` |
| 23 | A1/A2-Batch-Size: 32, A3-Batch-Size: 16 | `configs/backbone_a.yaml:L43`, `configs/backbone_a3.yaml:L20` |
| 24 | Klassifikations-Optimizer: AdamW | `configs/backbone_a.yaml:L75`, `configs/backbone_a3.yaml:L46` |
| 25 | A1/A2-Learning-Rate: 0.0004, A3-Learning-Rate: 0.00035 | `configs/backbone_a.yaml:L76`, `configs/backbone_a3.yaml:L47` |
| 26 | Cosine-Scheduler, t_max = 60 | `configs/backbone_a.yaml:L80–L81`, `configs/backbone_a3.yaml:L50–L51` |
| 27 | Max 60 Epochen, Early Stopping (patience 20 für A1/A2, 15 für A3) | `configs/backbone_a.yaml:L90–L91`, `configs/backbone_a3.yaml:L59–L61` |
| 28 | BCELoss mit pos_weight = auto | `configs/backbone_a.yaml:L85–L86`, `configs/backbone_a3.yaml:L56–L57` |
| 29 | Temperatur-Kalibrierung aktiviert, Methode: temperature | `configs/backbone_a.yaml:L101–L103`, `configs/backbone_a3.yaml:L72–L74` |
| 30 | A3-Kalibrierungsfraktion: 0.4 | `configs/backbone_a3.yaml:L75` |
| 31 | Mask-Weighted-Pooling aktiviert für A1/A2 | `configs/backbone_a.yaml:L71–L72` |
| 32 | Include-Mask-Channel: true (4 Eingabekanäle) | `configs/backbone_a.yaml:L41`, `inference_assets/manifest.json:L14,L22,L30,L38` |
| 33 | A3 Color-Features: redness, darkness, hsv | `configs/backbone_a3.yaml:L37–L38`, `inference_assets/manifest.json:L107–L111` |
| 34 | Color-Normalisierung: gray_world für A3/A4 | `configs/backbone_a3.yaml:L37`, `inference_assets/manifest.json:L106,L130` |
| 35 | A1-Threshold: 0.04, Temperatur: 0.415 | `configs/thresholds_base.json:L21–L22` |
| 36 | A2-Threshold: 0.45, Temperatur: 3.317 | `configs/thresholds_base.json:L27–L28` |
| 37 | A3-Threshold: 0.2, Temperatur: 3.283 | `configs/thresholds_base.json:L33–L34` |
| 38 | A4-Threshold: 0.24, Temperatur: 5.0 | `configs/thresholds_base.json:L39–L40` |
| 39 | Labels: [unbekannt, Never, Red, Yellow, Green] | `configs/thresholds_base.json:L2–L8` |
| 40 | Entscheidungsfluss: Seg → A1 → A2 → A3 → A4 | `configs/thresholds_base.json:L9–L15` |
| 41 | Klassische Override-Regeln: deq ≥ 580 px → Yellow, circularity < 0.8 → Yellow | `configs/thresholds_base.json:L47–L51` |
| 42 | A1 Fold 0: F1 = 1.0, AUC = 1.0, TP = 12, FP = 0, FN = 0 | `outputs/backbone_a/a1/fold_00/summary.json:L26–L39` |
| 43 | A2 Fold 0: F1 = 0.944, AUC = 0.9997, Precision = 1.0, Recall = 0.895 | `outputs/backbone_a/a2/fold_00/summary.json:L12–L25` |
| 44 | A3 Fold 0 (kalibriert): F1 = 0.841, AUC = 0.977, Precision = 0.738, Recall = 0.978 | `outputs/backbone_a/a3/fold_00/summary.json:L27–L41` |
| 45 | A4 Fold 0 (kalibriert): F1 = 0.937, AUC = 0.996, Precision = 0.941, Recall = 0.933 | `outputs/backbone_a/a4/fold_00/summary.json:L27–L41` |
| 46 | Klassik-Metriken: Accuracy = 0.870, Precision_yellow = 0.890, Recall_yellow = 0.853 | `outputs/classical/metrics.json:L12–L14` |
| 47 | Klassik-Support: green = 375, yellow = 400 | `outputs/classical/metrics.json:L15–L18` |
| 48 | Instance-Crops (Ordner): red = 269, yellow = 397, green = 400, never = 382 (Gesamt: 1448) | Dateizählung `data/instance_crops/images/{red,yellow,green,never}/` |
| 49 | Instance-Masks (Ordner): red = 260, yellow = 380, green = 378, never = 380 (Gesamt: 1398) | Dateizählung `data/instance_crops/masks/{red,yellow,green,never}/` |
| 50 | 27 Roh-XML-Annotationen + 27 Roh-JPG-Bilder | Dateizählung `data/raw/annotations/`, `data/raw/images/` |
| 51 | Segmentierungs-Split: 18 Train / 4 Val / 5 Test (70/15/15 %) | `data/processed/{train,val,test}/annotations.json`, `src/pipelines/prepare_dataset.py:L304` |
| 52 | A2-Splits: train = 875, val = 341, test = 8 | Dateizählung `data/instance_crops/splits/a2/` |
| 53 | A3-Splits: train = 740, val = 320, test = 6 | Dateizählung `data/instance_crops/splits/a3/` |
| 54 | 5 ONNX-Modelle: segmentation, a1, a2, a3, a4 | `inference_assets/manifest.json:L2–L43` |
| 55 | ONNX-Inference: CPUExecutionProvider | `inference_single.py:L277` |
| 56 | Top-K = 50 größte Instanzen bei Inference | `inference_single.py:L620,L650` |
| 57 | Margin für Crop-Extraktion: 15 % | `inference_single.py:L619` |
| 58 | Gray-World: 80 % korrigiert + 20 % Original, Gains geclippt auf [1/1.8, 1.8] | `inference_single.py:L516–L518` |
| 59 | Mixed Precision (AMP) aktiviert | `configs/train.yaml:L55`, `configs/backbone_a.yaml:L95` |
| 60 | Gradient Clipping: norm = 1.0 | `configs/train.yaml:L56`, `configs/backbone_a.yaml:L98` |
| 61 | Segmentierungs-Freeze: stem | `configs/train.yaml:L25` |
| 62 | A3-Dropout: 0.4, A1/A2-Dropout: 0.3 | `configs/backbone_a3.yaml:L43`, `configs/backbone_a.yaml:L70` |
| 63 | PyTorch 2.7.1+cu128, CUDA 12.8 | `requirements.txt:L19–L20` |
| 64 | onnxruntime ≥ 1.20.1 | `requirements.txt:L22` |
| 65 | Crop-Pipeline: 180 Instanzen vorhergesagt, 179 akzeptiert, 1 abgelehnt | `data/instance_crops/metadata/summary.json` |
| 66 | Windows-App installiert Python 3.11.9 (AMD64/ARM64/x86 Support) | `run_windows.bat` |
| 67 | GUI-Titel: „Blueberry QA – Inferenz" | `inference_gui.py:L19` |
| 68 | Overlay-Modi in GUI: Seg, A1 (Beere), A2 (Never), A3 (Red), A4 (Green) | `inference_single.py:L1206–L1208` |
| 69 | 0 Instanzen → leeres Overlay ohne Label | `inference_single.py:L633–L637` |
| 70 | A2 Fold 0: Temperatur = 2.188, Threshold = 0.949 | `outputs/backbone_a/a2/fold_00/summary.json:L5,L8` |
| 71 | A3 Fold 0: Temperatur = 5.0, Threshold = 0.822 | `outputs/backbone_a/a3/fold_00/summary.json:L5,L8` |
| 72 | Klassenzuordnung erfolgt über Ordnernamen, nicht Dateinamen (`_infer_label_from_path` nutzt `path.parent.name`) | `src/data/classification_dataset.py:L94–L104` |
| 73 | Segmentierungs-COCO-JSON: Train 450 Annotationen, Val 100, Test 125 (Gesamt 675) | `data/processed/{train,val,test}/annotations.json` |
