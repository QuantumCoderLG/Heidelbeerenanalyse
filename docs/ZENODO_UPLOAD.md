# Zenodo-Upload vorbereiten

Dieses Repository trennt Quellcode, Forschungsdaten, Modellartefakte und
generierte Dateien. Bildmetadaten einschließlich EXIF bleiben unverändert.

## Vor der Veröffentlichung ausfüllen

1. Prüfen, ob MIT-Lizenz für den Quellcode rechtlich möglich ist.
2. Separate Lizenz für Bilddaten im Zenodo-Datensatz festlegen. Code-Lizenz
   gilt nicht automatisch für Bilder. Platzhalter in `docs/DATASET_README.md`
   ersetzen.
3. Bildrechte und Veröffentlichung sämtlicher EXIF-Daten bestätigen.

## Archive erzeugen

```bash
bash tools/prepare_publication_archives.sh
```

Ausgabe unter `zenodo_upload/v1.0.0/`:

| Datei | Ziel |
| --- | --- |
| `blueberry-source-images-v1.0.0.zip` | Zenodo-Dataset-Record |
| `blueberry-curated-crops-v1.0.0.zip` | Zenodo-Dataset-Record |
| `blueberry-models-v1.0.0.zip` | Zenodo-Model- oder Software-Record |
| `blueberry-research-results-v1.0.0.zip` | Zenodo-Software-Record, falls `outputs/` vorhanden |
| `Heidelbeeren-Bewertung-App-v1.0.0.zip` | GitHub Release und optional Zenodo-Software-Record |
| `SHA256SUMS-dataset.txt` | Mit Dataset-Record veröffentlichen |
| `SHA256SUMS-software.txt` | Mit Software-Record und GitHub Release veröffentlichen |
| `SHA256SUMS.txt` | Vollständige lokale Prüfliste |

`blueberry-source-images-v1.0.0.zip` enthält Originalbilder inklusive EXIF.
`data/all_images/_split/` fehlt bewusst, weil Dateien Kopien darstellen.
`blueberry-curated-crops-v1.0.0.zip` enthält kuratierte Trainings-Crops,
Masken, Metadaten und manuelle Splits, aber keine generierten Overlays.
Wenn `outputs/` bereits gelöscht wurde, bewahrt das Skript ein vorhandenes
Resultat-ZIP beim erneuten Lauf.

## Zenodo-Records

Empfohlene Trennung:

1. Dataset-Record: Source-Images, kuratierte Crops, `SHA256SUMS.txt`.
2. Software- oder Model-Record: Modelle, Forschungsresultate, Windows-App,
   `SHA256SUMS.txt`.
3. GitHub-Repository in Zenodo aktivieren. GitHub-Release `v1.0.0` nach
   öffentlichem Push erzeugen.
4. DOI-Badges und Record-Links anschließend in `README.md` ergänzen.

Vor Upload `SHA256SUMS.txt` prüfen:

```bash
cd zenodo_upload/v1.0.0
sha256sum --check SHA256SUMS.txt
```

Ausführliche Veröffentlichungsschritte stehen in
[`PUBLISHING_GUIDE.md`](PUBLISHING_GUIDE.md).
