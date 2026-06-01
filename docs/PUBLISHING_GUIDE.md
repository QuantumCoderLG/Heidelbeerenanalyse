# Veröffentlichung: GitHub und Zenodo

Stand: 2026-05-31

## Ausgangslage

- GitHub-Repository: `https://github.com/QuantumCoderLG/Heidelbeerenanalyse`
- Lokales schlankes Git-Repository: `github_upload/berries2.0/`
- Zenodo-Dateien: `zenodo_upload/v1.0.0/`
- Quellcode-Lizenz: MIT
- Autor: Lando Maximilian Garbe
- EXIF-Metadaten bleiben absichtlich erhalten.

## 1. Datenlizenz entscheiden

Vor Zenodo-Upload Datenlizenz festlegen. Empfehlung für offene
Forschungsdaten: `CC BY 4.0`. Diese erlaubt Nachnutzung bei Namensnennung.
Falls kommerzielle Nutzung ausgeschlossen werden soll: `CC BY-NC 4.0`.

Danach Platzhalter ersetzen:

```bash
cd /home/lando/Dokumente/berries2.0
sed -i 's/TODO: ADD DATASET LICENSE/CC BY 4.0/' docs/DATASET_README.md
```

Falls `CC BY-NC 4.0` gewählt wird, Befehl entsprechend ändern.

## 2. GitHub-Code veröffentlichen

GitHub-Repository ist aktuell privat und leer. Lokales Staging-Repository
enthält nur kleine Quellcode-Dateien.

Prüfen:

```bash
cd /home/lando/Dokumente/berries2.0/github_upload/berries2.0
git status --short
git remote -v
git log --oneline --decorate -n 3
```

Auf diesem Rechner existiert bereits `~/.ssh/id_ed25519.pub`, aber GitHub kennt
den Schlüssel noch nicht. Einmalig eintragen:

```bash
cat ~/.ssh/id_ed25519.pub
```

Dann:

1. `https://github.com/settings/ssh/new` öffnen.
2. Titel eintragen, zum Beispiel `lando-workstation`.
3. Unter `Key type` den Wert `Authentication Key` belassen.
4. Vollständige Ausgabe des `cat`-Befehls in `Key` einfügen.
5. `Add SSH key` anklicken.
6. Falls GitHub Passwort oder Zwei-Faktor-Code verlangt, bestätigen.

Verbindung testen:

```bash
ssh -T git@github.com
```

Erwartete Meldung enthält:

```text
Hi QuantumCoderLG! You've successfully authenticated
```

Push ausführen:

```bash
git push -u origin main
```

Alternative ohne SSH: Remote auf HTTPS setzen und beim Push GitHub-Benutzername
plus Personal Access Token statt GitHub-Passwort eingeben:

```bash
git remote set-url origin https://github.com/QuantumCoderLG/Heidelbeerenanalyse.git
git push -u origin main
```

Repository öffentlich stellen:

1. `https://github.com/QuantumCoderLG/Heidelbeerenanalyse` öffnen.
2. `Settings` anklicken.
3. Unter `General` bis `Danger Zone` scrollen.
4. Bei `Change repository visibility` auf `Change visibility` klicken.
5. `Public` wählen und Bestätigung abschließen.
6. Ausgeloggt oder in privatem Browserfenster prüfen, ob README sichtbar ist.

Alternative per Konsole: Fine-grained GitHub-Token für
`QuantumCoderLG/Heidelbeerenanalyse` erzeugen. Repository-Berechtigungen:
`Administration: Read and write`, `Contents: Read and write`.

```bash
mkdir -p ~/.config/github
chmod 700 ~/.config/github
printf '%s' 'PASTE_TOKEN_HERE' > ~/.config/github/token
chmod 600 ~/.config/github/token

cd /home/lando/Dokumente/berries2.0
bash tools/github_publish.sh status
bash tools/github_publish.sh public
```

## 3. Zenodo-DOIs reservieren

Dataset-Draft anlegen:

1. `https://zenodo.org` öffnen und anmelden.
2. Oben rechts `+` anklicken, dann `New upload`.
3. Unter DOI-Frage `No` wählen.
4. `Get a DOI now!` anklicken.
5. Reservierten Dataset-DOI notieren. Draft noch nicht veröffentlichen.

Software-/Model-Draft separat anlegen:

1. Oben rechts `+`, dann `New upload`.
2. DOI-Frage `No`, dann `Get a DOI now!`.
3. Reservierten Software-DOI notieren. Draft noch nicht veröffentlichen.

## 4. DOIs einsetzen und Archive finalisieren

`XXXXXXXX` und `YYYYYYYY` durch echte Zenodo-Nummern ersetzen:

```bash
cd /home/lando/Dokumente/berries2.0
sed -i 's/TODO: ADD DATASET DOI AFTER DEPOSIT/10.5281\\/zenodo.XXXXXXXX/g' README.md docs/DATASET_README.md
sed -i 's/TODO: ADD SOFTWARE DOI AFTER DEPOSIT/10.5281\\/zenodo.YYYYYYYY/g' README.md
bash tools/prepare_publication_archives.sh
cd zenodo_upload/v1.0.0
sha256sum --check SHA256SUMS.txt
```

GitHub-Staging synchronisieren und DOI-Commit pushen:

```bash
cd /home/lando/Dokumente/berries2.0
bash tools/sync_github_staging.sh
cd github_upload/berries2.0
git add .
git commit -m "Add Zenodo DOI links"
git push
```

## 5. Zenodo-Dataset-Record veröffentlichen

Dataset-Draft erneut öffnen und Dateien aus `zenodo_upload/v1.0.0/` hochladen:

```text
blueberry-source-images-v1.0.0.zip
blueberry-curated-crops-v1.0.0.zip
SHA256SUMS-dataset.txt
DATASET_README.md
```

Metadaten:

```text
Resource type: Dataset
Title: Blueberry Quality Dataset for Instance Segmentation and Multi-Stage Classification
Creators: Garbe, Lando Maximilian
Version: 1.0.0
Language: English
Access right: Open access
License: gewählte Datenlizenz
Keywords: blueberry, computer vision, instance segmentation, classification, food quality
```

Beschreibung aus `docs/DATASET_README.md` übernehmen. Dann `Save draft`,
`Preview`, prüfen, `Publish` anklicken und bestätigen.

Alternative per Konsole: Token mit Scopes `deposit:write` und
`deposit:actions` erzeugen und geschützt speichern:

```bash
mkdir -p ~/.config/zenodo
chmod 700 ~/.config/zenodo
printf '%s' 'PASTE_TOKEN_HERE' > ~/.config/zenodo/token
chmod 600 ~/.config/zenodo/token

cd /home/lando/Dokumente/berries2.0
bash tools/zenodo_drafts.sh status
bash tools/zenodo_drafts.sh prepare
bash tools/zenodo_drafts.sh status
```

`prepare` aktualisiert beide Drafts und lädt Dateien hoch. Veröffentlichung
bleibt separater, bestätigungspflichtiger Schritt:

```bash
bash tools/zenodo_drafts.sh publish
```

## 6. Zenodo-Software-/Model-Record veröffentlichen

Software-/Model-Draft erneut öffnen und Dateien hochladen:

```text
blueberry-models-v1.0.0.zip
blueberry-research-results-v1.0.0.zip
Heidelbeeren-Bewertung-App-v1.0.0.zip
SHA256SUMS-software.txt
UPLOAD_INSTRUCTIONS.md
```

Metadaten:

```text
Resource type: Software
Title: Heidelbeerenanalyse: Trained Models and Windows Inference Application
Creators: Garbe, Lando Maximilian
Version: 1.0.0
Language: German
Access right: Open access
License: MIT
Keywords: blueberry, computer vision, ONNX, instance segmentation, classification
Related identifier: https://github.com/QuantumCoderLG/Heidelbeerenanalyse
Relation: Is supplement to
```

Dann `Save draft`, `Preview`, prüfen, `Publish` anklicken und bestätigen.

## 7. Zenodo-GitHub-Integration aktivieren

Vor GitHub-Release aktivieren:

1. Auf Zenodo anmelden.
2. Profilmenü öffnen, `GitHub` anklicken.
3. GitHub-Konto verbinden, falls noch nicht verbunden.
4. `Sync now` anklicken.
5. `QuantumCoderLG/Heidelbeerenanalyse` suchen.
6. Schalter aktivieren.
7. Seite aktualisieren und aktiven Zustand prüfen.

Neue GitHub-Releases werden dann automatisch als Software-Archiv in Zenodo
aufgenommen. Dies archiviert Quellcode-Releases zusätzlich zum manuellen
Model-/App-Record.

## 8. GitHub-Release erstellen

1. `https://github.com/QuantumCoderLG/Heidelbeerenanalyse/releases/new`
   öffnen.
2. `Choose a tag` anklicken, neuen Tag `v1.0.0` erstellen.
3. Release-Titel: `Heidelbeerenanalyse v1.0.0`
4. Beschreibung eintragen:

```text
Initial public research release.

Includes source code, documentation and Windows inference application.
Large research datasets and model artifacts are published on Zenodo.
```

5. Diese Dateien aus `zenodo_upload/v1.0.0/` anhängen:

```text
Heidelbeeren-Bewertung-App-v1.0.0.zip
SHA256SUMS-software.txt
```

6. `Publish release` anklicken.
7. Zenodo-GitHub-Seite prüfen. Automatisch erzeugten Software-Record öffnen.

Alternative per Konsole nach aktivierter Zenodo-GitHub-Verknüpfung:

```bash
cd /home/lando/Dokumente/berries2.0
bash tools/github_publish.sh release v1.0.0
```

## 9. Abschlussprüfung

```bash
cd /home/lando/Dokumente/berries2.0/zenodo_upload/v1.0.0
sha256sum --check SHA256SUMS.txt

cd /home/lando/Dokumente/berries2.0/github_upload/berries2.0
git status --short
git remote -v
git log --oneline --decorate -n 3
```

Dann GitHub-Seite in privatem Browserfenster öffnen und prüfen:

- README sichtbar
- `LICENSE` sichtbar
- `CITATION.cff` sichtbar
- Release sichtbar
- Zenodo-DOIs im README sichtbar
- Dataset- und Software-Records öffentlich erreichbar
