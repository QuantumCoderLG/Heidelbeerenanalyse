# KI-gestützte Qualitätsbewertung von Heidelbeeren: Instanzsegmentierung und mehrstufige Klassifikation

Der zugehörige Quellcode und Projektdateien befindet sich im Repository unter dem Dummy-Link `https://github.com/example/berries2.0`.

## 1) Qualitätslogik: Mehrstufige binäre Entscheidungen und Klassenzuordnung

Das zentrale Konzept der Qualitätsbewertung beruht auf einer Kaskade aus vier binären Entscheidungen. Jede Stufe stellt eine Ja/Nein-Frage, und bei positivem Ausgang wird sofort ein Label vergeben. Nur bei negativem Ausgang wird die nächste Stufe befragt. Jede Stufe wird als eigenständiges binäres Problem trainiert.

Die fünf Qualitätsklassen sind physikalisch wie folgt definiert (Quelle: `windows_app_README.md`):

- **Green** („Gut"): Form, Größe, Farbe und Oberfläche sind in Ordnung. Keine sichtbaren Beschädigungen oder Schimmelbefall. Die Beere ist verkaufsfähig.
- **Yellow** („Akzeptabel"): Leicht unförmig oder zu groß, kleine weiße oder braune Stellen, dunkelblaue Druckstellen. Nicht perfekt, aber noch bewertbar.
- **Red** („Schlecht"): Nicht richtig blau (grünlich oder rötlich verfärbt), teilweiser Schimmelbefall, sichtbare braune Stellen.
- **Never** („Nicht bewertbar"): Extrem deformiert, mit Stiel, großflächige Beschädigungen oder braune Stellen, starker Schimmelbefall, oder kein Heidelbeer-Objekt. Auch unscharfe und verwackelte Aufnahmen fallen in diese Kategorie.
- **unbekannt**: Hintergrund, Hände, Beschriftungen oder andere Nicht-Beeren-Objekte. Diese Klasse wird nur intern verwendet und nicht als Qualitätsurteil ausgegeben.

Die Übergänge zwischen den Klassen – insbesondere zwischen Yellow und Green – sind in der Praxis fließend: Eine leicht unförmige Beere ohne sonstige Mängel kann je nach Beurteilung in beide Kategorien fallen.

Der Entscheidungsfluss ist im Quellcode dokumentiert (Quellcode: `configs/thresholds_base.json:L9–L15`):

1. **Segmentierung → Instanzen**: Das Bild wird segmentiert und in Einzelinstanzen aufgeteilt.
2. **A1: notberry?** → Wenn prob(notberry) ≥ 0,04: Label = „unbekannt", sonst weiter.
3. **A2: never?** → Wenn prob(never) ≥ 0,45: Label = „Never", sonst weiter.
4. **A3: red?** → Wenn prob(red) ≥ 0,2: Label = „Red", sonst weiter.
5. **A4: green?** → Wenn prob(green) ≥ 0,24: Label = „Green", sonst Label = „Yellow".

Jeder Klassifikator gibt einen Logit-Wert aus, der durch den jeweiligen Temperaturparameter dividiert und anschließend durch die Sigmoid-Funktion in eine Wahrscheinlichkeit überführt wird. Die Temperaturwerte sind: A1 = 0,415, A2 = 3,317, A3 = 3,283, A4 = 5,0 (Quellcode: `configs/thresholds_base.json:L22,L28,L34,L40`). Eine Temperatur größer als 1 „weicht" die Vorhersagen auf und macht das Modell weniger überkonfident; eine Temperatur kleiner als 1 „schärft" sie.

Nach der neuronalen Klassifikation durch A4 greifen zusätzlich klassische Regeln als Override-Mechanismus. Diese basieren auf geometrischen Merkmalen der Instanzmaske:

- **Äquivalenter Durchmesser (deq)**: Beträgt der deq mindestens 580,0 Pixel, wird die Beere unabhängig von der A4-Entscheidung als „Yellow" (zu groß) eingestuft (Quellcode: `configs/thresholds_base.json:L47`).
- **Zirkularität**: Liegt die Zirkularität unter 0,8, wird die Beere als „Yellow" (zu unregelmäßig) eingestuft (Quellcode: `configs/thresholds_base.json:L48`).

Diese Overrides korrigieren Fälle, in denen das neuronale Netz eine Beere als green bewertet, deren Geometrie aber außerhalb des für grüne Beeren typischen Bereichs liegt. Die klassischen Regeln wurden auf Basis der Statistiken grüner Beeren kalibriert: Der mittlere deq beträgt 504,2 Pixel bei einer Standardabweichung von 32,2, und die mittlere Zirkularität liegt bei 0,647 (Quellcode: `inference_assets/manifest.json:L176–L206`).

## 2) Ziel und Einordnung in die wissenschaftliche Arbeit

Das System bearbeitet hochauflösende Bilder (6000 × 4000 Pixel), auf denen typischerweise rund 25 Heidelbeeren auf einer Platte angeordnet sind (Quellcode: `configs/train.yaml:L116`). Jede erkannte Beere durchläuft eine Kette aus mehreren binären Entscheidungen, an deren Ende eine von fünf Qualitätskategorien steht: „unbekannt", „Never", „Red", „Yellow" oder „Green" (Quellcode: `configs/thresholds_base.json:L2–L8`). Das Repository enthält die vollständige Pipeline: Datenvorbereitung, Training, Evaluation und eine Windows-Anwendung für die Inference.

## 3) Datensatz und Annotationen

Der Datensatz besteht aus hochauflösenden JPG-Fotografien von Heidelbeeren, die auf Platten angeordnet und unter kontrollierten Bedingungen aufgenommen wurden. Die Rohbilder sind nach Qualitätsklassen in Unterordner gegliedert: Yellow, Green, Never und Red (Quellcode: `data/all_images/Ampel/`). Insgesamt umfasst das Bildarchiv 222 JPG- und CR3-Dateien mit verschiedenen Versuchsbedingungen, darunter frische, gefrorene und aufgetaute Proben (Quellcode: `data/all_images/Heidelbeeren2/`).

Für die Segmentierung wurden 27 Bilder mit Polygonannotationen im CVAT-XML-Format (Computer Vision Annotation Tool, ein webbasiertes Annotationswerkzeug) [1] versehen (Quellcode: `data/raw/annotations/`, 27 XML-Dateien). Jede Heidelbeere wurde als einzelnes Polygon markiert. Diese Annotationen werden über einen Parser in COCO-Format (ein verbreitetes Annotationsformat für Objekterkennung) [2] konvertiert, wobei die Polygone zu binären Instanzmasken rasterisiert werden (Quellcode: `src/data/xml_parser.py`, `src/data/rasterize.py`, `src/pipelines/prepare_dataset.py`). Die aufbereiteten Daten liegen aufgeteilt in Trainings-, Validierungs- und Testsets vor: 18 Bilder mit 450 Annotationen im Training, 4 Bilder mit 100 Annotationen in der Validierung und 5 Bilder mit 125 Annotationen im Test (Quellcode: `data/processed/{train,val,test}/annotations.json`; Split-Defaults 70/15/15 % in `src/pipelines/prepare_dataset.py:L304`).

Zusätzlich existieren 7 XML-Dateien mit Bounding-Box-Annotationen für die Klasse „Kaputte Stelle" (Quellcode: `data/BBoxes_annotation_data/`, 7 Dateien). Diese dienen der Identifikation beschädigter Stellen auf den Beeren.

Für die Klassifikation wurden aus den Segmentierungsergebnissen Einzelbeeren-Ausschnitte (Crops) extrahiert. Der Datensatz der Instanz-Crops umfasst 1448 Bilder, verteilt auf die Klassen: red = 269, yellow = 397, green = 400, never = 382 (Quellcode: Dateizählung `data/instance_crops/images/`). Zugehörige binäre Masken existieren in leicht abweichender Anzahl: red = 260, yellow = 380, green = 378, never = 380, insgesamt 1398 (Quellcode: Dateizählung `data/instance_crops/masks/`). Die Metadaten sind in einer CSV-Datei organisiert, die Klasse, Fold-Zugehörigkeit und Dateipfad jeder Instanz erfasst (Quellcode: `data/instance_crops/metadata/crops.csv`). Die Klassenzuordnung erfolgt ausschließlich über den Ordnernamen, nicht über den Dateinamen: Die Funktion `_infer_label_from_path` wertet `path.parent.name` aus (Quellcode: `src/data/classification_dataset.py:L94–L104`). Eine Datei mit dem Namen `Red_1_1_id022.png` im Ordner `yellow/` wird als „yellow" trainiert – sie wurde nach manueller Prüfung umsortiert.

Für die Aufgaben A2 (never vs. ok) und A3 (red vs. not-red) werden manuelle Splits verwendet: A2 umfasst 875 Trainings-, 341 Validierungs- und 8 Test-Einträge; A3 umfasst 740 Trainings-, 320 Validierungs- und 6 Test-Einträge (Quellcode: Dateizählung `data/instance_crops/splits/a2/`, `data/instance_crops/splits/a3/`). Ergänzend wurden synthetische „Never"-Crops durch Vereinigung benachbarter Instanzen erzeugt, deren Metadaten in einer JSONL-Datei dokumentiert sind (Quellcode: `data/instance_crops/metadata/never_union_crops.jsonl`, `configs/backbone_a.yaml:L18–L28`).

## 4) Pipeline-Überblick (End-to-End)

Die Gesamtpipeline verarbeitet ein Eingabebild in mehreren aufeinanderfolgenden Stufen. Zunächst wird das Bild durch ein Segmentierungsmodell in eine Wahrscheinlichkeitskarte (Probability Map) überführt, die für jedes Pixel die Wahrscheinlichkeit angibt, ob es zu einer Heidelbeere gehört. Nach Schwellenwertbildung, morphologischen Operationen und Watershed-Segmentierung (ein Verfahren zur Trennung zusammenhängender Objekte) [3] werden die einzelnen Instanzen identifiziert (Quellcode: `inference_single.py:L459–L494`).

Die bis zu 50 flächenmäßig größten Instanzen werden ausgewählt (Quellcode: `inference_single.py:L650`) und durchlaufen dann sequenziell vier binäre Klassifikationsstufen:

1. **A1 (notberry vs. berry)**: Erkennt, ob der Ausschnitt überhaupt eine Beere zeigt. Wenn die Wahrscheinlichkeit für „notberry" den Schwellenwert von 0,04 überschreitet, erhält die Instanz das Label „unbekannt" und wird nicht weiter klassifiziert (Quellcode: `configs/thresholds_base.json:L19–L24`, `inference_single.py:L757–L782`).

2. **A2 (never vs. ok)**: Prüft, ob die erkannte Beere der Klasse „Never" (nicht bewertbar) angehört. Ab einem Schwellenwert von 0,45 wird das Label „Never" vergeben (Quellcode: `configs/thresholds_base.json:L25–L30`, `inference_single.py:L784–L809`).

3. **A3 (red vs. not-red)**: Bestimmt, ob die Beere eine rote Qualitätseinstufung erhält. Der Schwellenwert liegt bei 0,2 (Quellcode: `configs/thresholds_base.json:L31–L36`, `inference_single.py:L811–L836`).

4. **A4 (green vs. yellow)**: Unterscheidet abschließend zwischen „Green" (beste Qualität) und „Yellow" (eingeschränkte Qualität). Der Schwellenwert beträgt 0,24 (Quellcode: `configs/thresholds_base.json:L37–L42`, `inference_single.py:L838–L963`).

Nach der A4-Entscheidung greifen zusätzlich klassische, geometriebasierte Override-Regeln: Beeren mit einem äquivalenten Durchmesser (deq, berechnet als 2 · √(Fläche / π)) von mindestens 580 Pixeln werden als „zu groß" auf „Yellow" überschrieben, und Beeren mit einer Zirkularität (das Verhältnis von Fläche zu Umfangsquadrat, normiert auf einen Kreis) unter 0,8 werden als „zu unregelmäßig" ebenfalls auf „Yellow" gesetzt (Quellcode: `configs/thresholds_base.json:L43–L52`, `inference_single.py:L882–L910`).

## 5) Modellarchitektur und zentrale Designentscheidungen

### Segmentierung: DeepLabV3+ mit ResNet-50

Für die semantische Segmentierung wurde eine DeepLabV3+-Architektur gewählt [4]. DeepLabV3+ ist ein Encoder-Decoder-Netzwerk, das ASPP (Atrous Spatial Pyramid Pooling) zur Erfassung von Kontextinformation auf verschiedenen Skalen einsetzt und einen Decoder-Pfad mit Low-Level-Features für feinere Kanten nutzt. Als Backbone dient ein ResNet-50 [5], vortrainiert auf ImageNet [6] (Gewichte: IMAGENET1K_V2) (Quellcode: `src/training/models.py:L67–L75`).

Der Output-Stride beträgt 8, was die räumliche Auflösung im Encoder auf ein Achtel der Eingabegröße reduziert. Die ASPP-Dilatationsraten sind [6, 12, 18] (Quellcode: `configs/train.yaml:L23,L26`). Ein Dropout von 0,1 wird im Klassifikator-Kopf eingesetzt (Quellcode: `configs/train.yaml:L27`). Die Architektur gibt eine einzelne Klasse aus (binär: Beere vs. Hintergrund) und erzeugt Logits, die mittels Sigmoid-Funktion in Wahrscheinlichkeiten umgerechnet werden (Quellcode: `inference_single.py:L471–L474`).

### Klassifikation: EfficientNet-Familie

Für die binären Klassifikationsstufen werden EfficientNet-Modelle eingesetzt [7]. Die Stufen A1, A2 und A4 verwenden ein EfficientNet-B1 (Quellcode: `configs/backbone_a.yaml:L68`), während A3 ein größeres EfficientNet-B5 verwendet, um die anspruchsvollere Farbunterscheidung besser zu erfassen (Quellcode: `configs/backbone_a3.yaml:L41`).

Alle Klassifikatoren erhalten einen erweiterten Eingangskanal: Statt der üblichen drei RGB-Kanäle erhalten die Modelle einen vierten Kanal mit der binären Instanzmaske. Dies wird durch Anpassung der ersten Faltungsschicht realisiert, wobei die vortrainierten Gewichte der RGB-Kanäle erhalten bleiben und der neue Kanal mit Nullen initialisiert wird (Quellcode: `src/training/classifier_models.py:L22–L41`, `configs/backbone_a.yaml:L41`). Dadurch kann das Netzwerk gezielt auf **einzelne** Bereiche der Beere fokussieren.

Zusätzlich wird für A1/A2 ein Mask-Weighted-Pooling-Wrapper eingesetzt: Statt des standardmäßigen Global Average Pooling wird ein maskengewichteter Durchschnitt berechnet, der den Hintergrund ausschließt (Quellcode: `configs/backbone_a.yaml:L71–L72`, `src/training/classifier_models.py`).

Für den A3-Klassifikator werden ergänzende Farbmerkmale berechnet: Redness (R − max(G, B)), Darkness (1 − V aus dem HSV-Farbraum) und die drei HSV-Kanäle (Hue, Saturation, Value). Diese werden als zusätzliche Eingabekanäle dem Netzwerk übergeben (Quellcode: `configs/backbone_a3.yaml:L37–L38`, `inference_single.py:L528–L563`). Vor der Berechnung wird eine Gray-World-Farbnormalisierung durchgeführt, die Farbverschiebungen durch unterschiedliche Beleuchtung kompensiert. Dabei wird jeder Farbkanal so skaliert, dass der Mittelwert aller Kanäle gleich ist, mit einer Blendung von 80 % korrigierten und 20 % Originalwerten und einer Gain-Begrenzung auf den Faktor 1,8 (Quellcode: `inference_single.py:L510–L521`).

## 6) Trainingsverfahren und Experiment-Setup

### Segmentierung

Das Segmentierungsmodell wird mit einer 5-Fold-Kreuzvalidierung trainiert (Quellcode: `configs/train.yaml:L111–L112`). Die Eingabebilder werden auf 1024 × 1024 Pixel skaliert unter Beibehaltung des Seitenverhältnisses (Letterboxing) (Quellcode: `configs/train.yaml:L7–L9`). Als Verlustfunktion dient eine Kombination aus Dice-Loss und Binary Cross-Entropy (dice_bce) mit gleicher Gewichtung (Quellcode: `configs/train.yaml:L29–L33`). Der Optimizer ist Adam mit einer Lernrate von 0,0003, wobei das Backbone eine um den Faktor 10 reduzierte Lernrate erhält (0,00003) (Quellcode: `configs/train.yaml:L36–L46`). Ein Poly-Scheduler mit power = 0,9 reduziert die Lernrate über die Trainingszeit (Quellcode: `configs/train.yaml:L48–L50`).

Das Training läuft maximal 120 Epochen mit Early Stopping (Patience 20, min_delta = 0,0005) (Quellcode: `configs/train.yaml:L53,L78–L81`). Mixed Precision (AMP, Automatic Mixed Precision) und Gradient Clipping (Norm 1,0) sind aktiviert (Quellcode: `configs/train.yaml:L55–L56`). Der Stem-Block des Backbones wird eingefroren, um die vortrainierten niedrigen Features zu erhalten (Quellcode: `configs/train.yaml:L25`).

### Klassifikation (A1–A4)

Die Klassifikatoren verwenden AdamW als Optimizer mit Weight Decay von 0,02 (A1/A2) bzw. 0,01 (A3) (Quellcode: `configs/backbone_a.yaml:L75–L77`, `configs/backbone_a3.yaml:L46–L48`). Die Lernraten betragen 0,0004 für A1/A2 und 0,00035 für A3 (Quellcode: `configs/backbone_a.yaml:L76`, `configs/backbone_a3.yaml:L47`). Ein Cosine-Scheduler reduziert die Lernrate über 60 Epochen auf ein Minimum von 10⁻⁶ (Quellcode: `configs/backbone_a.yaml:L79–L82`).

Die Verlustfunktion ist Binary Cross-Entropy mit automatisch berechneter Klassengewichtung (pos_weight = auto) (Quellcode: `configs/backbone_a.yaml:L85–L86`). Die Batch-Size beträgt 32 für A1/A2/A4 und 16 für A3 (wegen des größeren EfficientNet-B5) (Quellcode: `configs/backbone_a.yaml:L43`, `configs/backbone_a3.yaml:L20`). Mixed Precision und Gradient Clipping (1,0) sind ebenfalls aktiv (Quellcode: `configs/backbone_a.yaml:L95–L98`).

Die Augmentierung unterscheidet sich je nach Aufgabe: A1/A2 verwenden einen texturorientierten Modus mit Helligkeits-/Kontrastveränderungen, CLAHE (Contrast Limited Adaptive Histogram Equalization), Unsharp-Masking und Rauschen (Quellcode: `configs/backbone_a.yaml:L49–L62`). A3 verwendet einen farborientierten Modus mit gezielten Hue- (0,02), Sättigungs- (0,15) und Gamma-Variationen (Quellcode: `configs/backbone_a3.yaml:L26–L35`).

Nach dem Training wird eine Temperatur-Kalibrierung durchgeführt. Dabei wird ein einzelner Temperaturparameter auf einem Kalibrierungs-Subset optimiert, um die Modellkonfidenz an die tatsächliche Trefferquote anzupassen (Quellcode: `configs/backbone_a.yaml:L101–L107`). Für A3 wird ein größerer Kalibrierungsanteil von 40 % verwendet (Quellcode: `configs/backbone_a3.yaml:L75`). Die optimalen Schwellenwerte werden über eine Rastersuche bestimmt, die eine kostenbasierte Metrik minimiert (Falsch-Negativ-Kosten = 10, Falsch-Positiv-Kosten = 1) bei einem Mindest-Recall von 0,85 (Quellcode: `configs/backbone_a.yaml:L109–L121`).

### Hardware und Frameworks

Das Training erfordert PyTorch 2.7.1 mit CUDA 12.8 [8] (Quellcode: `requirements.txt:L19–L20`). Weitere zentrale Bibliotheken sind torchvision 0.22.1, albumentations ≥ 1.4.0, OpenCV, Pillow, NumPy und Pandas (Quellcode: `requirements.txt:L1–L22`). Für die Inference wird ONNX Runtime ≥ 1.20.1 eingesetzt [9], das ausschließlich auf der CPU läuft (CPUExecutionProvider) (Quellcode: `requirements.txt:L22`, `inference_single.py:L277`).

## 7) Grenzen, typische Fehlerbilder und Edge-Cases

### Keine Beeren im Bild (0 Instanzen)

Wenn die Segmentierung keine Instanzen erkennt, gibt die Pipeline ein leeres Overlay ohne Label zurück und meldet 0 Instanzen in den Metadaten (Quellcode: `inference_single.py:L633–L637`). Es erfolgt keine Fehlermeldung; das Originalbild wird unverändert angezeigt.

### Mehr oder weniger als 25 Beeren

Die Pipeline verarbeitet beliebig viele Instanzen, begrenzt jedoch die Klassifikation auf die 50 flächenmäßig größten (top_k = 50) (Quellcode: `inference_single.py:L620,L650`).

### Filterung kleiner und unförmiger Segmente

Segmente mit einer Fläche unter 30 Pixeln werden als Rauschen verworfen (Quellcode: `configs/train.yaml:L95`, `inference_assets/manifest.json:L234`). Segmente mit einer Zirkularität unter 0,25 werden ebenfalls entfernt, da sie wahrscheinlich keine Heidelbeeren darstellen (Quellcode: `configs/train.yaml:L97–L98`, `inference_assets/manifest.json:L235–L237`).

### Unsicherheitsbehandlung

Die Stufe A1 fungiert als expliziter Unsicherheitsfilter: Bereits bei einer sehr niedrigen Wahrscheinlichkeit von 0,04 wird eine Instanz als „unbekannt" markiert (Quellcode: `configs/thresholds_base.json:L21`). Die Threshold-Suche der Klassifikatoren verwendet eine asymmetrische Kostenmatrix (Falsch-Negativ-Kosten = 10, Falsch-Positiv-Kosten = 1), d. h. das Übersehen eines Qualitätsmangels wird zehnfach stärker bestraft als eine Falschklassifikation (Quellcode: `configs/backbone_a.yaml:L117–L121`). Ein Abstain-Margin von 0,02 ist für A3 konfiguriert, der eine Enthaltungszone um den Schwellenwert definiert (Quellcode: `configs/backbone_a3.yaml:L89`).

### Bekannte Schwachstellen und überlappende Trainingsdaten

Die Trainingsdaten weisen zwischen den Qualitätsklassen unvermeidliche Überlappungen auf. Die physikalischen Unterschiede zwischen einer gelben und einer grünen Heidelbeere – etwa eine leichte Unförmigkeit oder ein kaum sichtbarer Druckfleck – sind oft so gering, dass selbst menschliche Bewerter nicht einheitlich urteilen. Ähnliches gilt für die Grenze zwischen Red und Yellow: Eine Beere mit wenigen kleinen braunen Stellen kann je nach Schwere beider Klassen zugeordnet werden. Diese inhärente Unschärfe der Klassengrenzen begrenzt die maximal erreichbare Klassifikationsgenauigkeit.

Das A3-Modell (red vs. not-red) zeigt die niedrigste Klassifikationsleistung mit einem kalibrierten F1-Score von 0,841 und einer Precision von nur 0,738, was bedeutet, dass 26,2 % der als „red" klassifizierten Beeren tatsächlich nicht zur Klasse „red" gehören (Quellcode: `outputs/backbone_a/a3/fold_00/summary.json:L28–L31`). Ohne Temperatur-Kalibrierung liegt der F1-Score bei 0,581; die Kalibrierung hebt ihn auf 0,841 (Quellcode: `outputs/backbone_a/a3/fold_00/summary.json:L13–L18`). Neben der Klassenüberlappung trägt auch die geringe Anzahl an Red-Crops (269 von 1448 Instanzen) zu dieser niedrigeren Leistung bei.

## 8) Reproduzierbarkeit und Wartbarkeit (Versionen, Configs, Seeds, Logging)

Alle Trainingsläufe verwenden den festen Seed 1337 (Quellcode: `configs/train.yaml:L1`, `configs/backbone_a.yaml:L1`, `configs/backbone_a3.yaml:L1`). Die exakten Framework-Versionen sind in `requirements.txt` festgeschrieben, einschließlich PyTorch 2.7.1+cu128 und ONNX Runtime ≥ 1.20.1 (Quellcode: `requirements.txt:L19–L22`).

Alle Hyperparameter liegen in YAML-Konfigurationsdateien und lassen sich über CLI-Overrides anpassen (Quellcode: `configs/train.yaml`, `configs/backbone_a.yaml`, `configs/backbone_a3.yaml`). Die Trainingsläufe protokollieren Metriken in CSV-Dateien und JSON-Summaries (Quellcode: `configs/backbone_a.yaml:L123–L128`, `configs/train.yaml:L120–L124`). Für jede Fold- und Aufgabenkombination werden detaillierte Summary-Dateien mit Schwellenwerten, Temperaturen, Konfusionsmatrizen und Slice-Metriken pro Quellbild gespeichert (Quellcode: `outputs/backbone_a/*/fold_*/summary.json`).

Die Experiment-Skripte für systematische A2- und A3-Läufe sind im Repository archiviert, zusammen mit den resultierenden Logdateien (Quellcode: `scripts/run_a2_experiments.sh`, `scripts/run_a3_experiments.sh`, `outputs/experiment_logs/`). Eine Testsuite mit Pytest deckt die Verlustfunktionen, Metriken, Postprocessing und Crop-Pipeline ab (Quellcode: `tests/test_losses.py`, `tests/test_metrics.py`, `tests/test_postprocessing.py`, `tests/test_crop_pipeline.py`).

Das Windows-Bundle wird automatisiert über `make windows-dist` erstellt und durch `tools/verify_windows_bundle.py` auf Vollständigkeit geprüft (Quellcode: `Makefile:L66–L69`). Die Entscheidungsschwellen werden zentral in `configs/thresholds_base.json` gepflegt und beim Build in das Bundle kopiert (Quellcode: `configs/thresholds_base.json`).

## 9) Kurzes Fazit und naheliegende Verbesserungen

Die Segmentierung erreicht einen IoU von 0,979 (Quellcode: `outputs/logs/metrics.json:L4`). Die Klassifikationsstufen liefern folgende F1-Scores: A1 = 1,0 (Quellcode: `outputs/backbone_a/a1/fold_00/summary.json:L30`), A2 = 0,944 (Quellcode: `outputs/backbone_a/a2/fold_00/summary.json:L17`), A4 = 0,937 (Quellcode: `outputs/backbone_a/a4/fold_00/summary.json:L34`).

Die A3-Stufe (red vs. not-red) erreicht einen kalibrierten F1-Score von 0,841 bei einer Precision von 0,738 – 26,2 % der als „red" klassifizierten Beeren sind also Fehlklassifikationen (Quellcode: `outputs/backbone_a/a3/fold_00/summary.json:L30–L31`). Das rein klassische Regelwerk für die Yellow/Green-Unterscheidung erreicht eine Accuracy von 0,870: 42 von 375 grünen Beeren werden fälschlich als yellow klassifiziert, 59 von 400 gelben als green (Quellcode: `outputs/classical/metrics.json:L2–L14`). Im finalen System ergänzt der neuronale A4-Klassifikator die klassischen Override-Regeln.

Diese Fehlerraten spiegeln die fließenden Übergänge zwischen den Qualitätsklassen wider – die Grenzen zwischen Yellow und Green oder zwischen Red und Yellow sind auch für menschliche Bewerter nicht immer eindeutig. In praktischen Echt-Welt-Tests hat das System jedoch gezeigt, dass es zuverlässig zwischen guten (Green) und schlechten (Red, Never) Heidelbeeren unterscheiden kann. Die Fehlklassifikationen konzentrieren sich auf die benachbarten Klassen (Yellow ↔ Green, Red ↔ Yellow), während grobe Verwechslungen zwischen den Extremen (z. B. Green ↔ Never) praktisch nicht auftreten.

Für A2 und A3 wurden systematische Hyperparameter-Suchen durchgeführt; die zugehörigen Shell-Skripte und Logdateien liegen unter `scripts/run_a2_experiments.sh`, `scripts/run_a3_experiments.sh` und `outputs/experiment_logs/`.

## 10) Externe Quellen (Web)

[1] CVAT (Computer Vision Annotation Tool): https://www.cvat.ai/ (letzter Zugriff: 09.02.2026)

[2] COCO (Common Objects in Context): https://cocodataset.org/ (letzter Zugriff: 09.02.2026)

[3] OpenCV-Dokumentation, Watershed-Algorithmus: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html (letzter Zugriff: 09.02.2026)

[4] DeepLabV3+ (arXiv:1802.02611): https://arxiv.org/abs/1802.02611 (letzter Zugriff: 09.02.2026)

[5] ResNet (arXiv:1512.03385): https://arxiv.org/abs/1512.03385 (letzter Zugriff: 09.02.2026)

[6] ImageNet: https://www.image-net.org/ (letzter Zugriff: 09.02.2026)

[7] EfficientNet (arXiv:1905.11946): https://arxiv.org/abs/1905.11946 (letzter Zugriff: 09.02.2026)

[8] PyTorch: https://pytorch.org/ (letzter Zugriff: 09.02.2026)

[9] ONNX Runtime: https://onnxruntime.ai/docs/ (letzter Zugriff: 09.02.2026)
