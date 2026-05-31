# Blueberry QA – Kurzanleitung (Windows)

Diese Datei beschreibt kurz, wie Sie die Anwendung unter Windows starten und wie die Kategorien (Labels) der KI zu verstehen sind.

---

## 1. Start unter Windows – welche `.bat` Datei?

In diesem Ordner liegen zwei Startdateien:

- `run_windows.bat`
- `run_windows_no_install.bat`

**`run_windows.bat` – nur beim allerersten Mal benutzen**

- Prüft, ob Python (mindestens Version 3.11) installiert ist.
- Lädt bei Bedarf Python aus dem Internet herunter und installiert es.
- Legt eine lokale Python-Umgebung an und installiert alle benötigten Pakete.
- Startet danach die GUI.

Verwendung:
- Beim **allerersten Start** auf einem neuen Rechner.
- Falls später einmal etwas „kaputt“ ist (z. B. die Umgebung gelöscht wurde), können Sie diese Datei erneut verwenden.

**`run_windows_no_install.bat` – für den normalen täglichen Gebrauch**

- Führt **keine** Installationen aus.
- Nutzt die bereits vorhandene, beim ersten Mal eingerichtete Umgebung.
- Startet nur noch die GUI.

Verwendung:
- Für den **normalen Start im Alltag** (schneller, kein Admin-Recht nötig).
- Voraussetzung: `run_windows.bat` wurde mindestens einmal auf dem Gerät erfolgreich ausgeführt.

---

## 2. Decision Flow – wie entscheidet die KI?

Die Datei `Thresholds.json` in diesem Ordner enthält den sogenannten **Decision Flow** und die **Labels**. In Kurzform:

1. **Segmentierung → einzelne Beeren**
   - Das Bild wird in einzelne Objekte (vermutete Heidelbeeren) aufgeteilt.

2. **A1: „notberry?“ – Ist das überhaupt eine Heidelbeere?**
   - Wenn **nein** (also eher Verpackung, Blatt, Hintergrund, Hand oder anderes Objekt):  
     → Label = `unbekannt`  
   - Wenn **ja**: weiter zu A2.

3. **A2: „never?“ – Absolut nicht verkaufsfähig?**
   - Wenn **ja**: → Label = `Never`  
   - Wenn **nein**: weiter zu A3.

4. **A3: „red?“ – Deutlich unreif oder problematisch?**
   - Wenn **ja**: → Label = `Red`  
   - Wenn **nein**: weiter zu A4.

5. **A4: „green?“ – Sehr gute Qualität?**
   - Wenn **ja**: → Label = `Green`  
   - Wenn **nein**: → Label = `Yellow`

Bevor eine Heidelbeere wirklich als Green eingestuft wird, kommt zusätzlich noch eine **klassische Entscheidungslogik** zum Einsatz (siehe Abschnitt 4).

---

## 3. Labels (Kategorien) und ihre Bedeutung

Die wichtigsten Labels in `Thresholds.json` sind:

- `unbekannt` – interne Kategorie, wird in der Oberfläche nicht angezeigt
- `Never` (schwarz)
- `Red` (Red)
- `Yellow` (Yellow)
- `Green` (Green)

Im Detail:

**`Never`**  
Diese Beeren sollen **nie** in den Verkauf:
- Extrem verformte Beeren.
- Beeren mit Stiel.
- Beeren mit **großen**, kaputten oder braunen Stellen.
- Starke Schimmelbildung.
- Alles, was **überhaupt keine Heidelbeere** ist (z. B. andere Früchte, Fremdkörper).

Hinweis:  
Auch **verschwommene oder verwackelte Heidelbeeren** landen oft in `Never`, weil die KI nicht mehr richtig erkennen kann, was zu sehen ist.

**`Red`**  
Beeren, die deutlich problematisch oder unreif sind:
- Farblich **nicht richtig blau**, sondern eher grünlich oder rötlich.
- Beeren mit **etwas Schimmel**.
- Beeren mit **normal großen** braunen Stellen (deutlich sichtbar, aber nicht komplett zerstört).

**`Yellow`**  
Beeren, die gerade noch in Ordnung sind, aber nicht „Top-Qualität“:
- Leicht unförmig oder deutlich zu groß.
- Wirken insgesamt nicht perfekt.
- Mögliche Ursachen:
  - kleine weiße oder braune Stellen,
  - dunkelblaue Druckstellen.

**`Green`**  
Die gewünschten, guten Beeren:
- Form, Größe, Farbe und Oberfläche passen.
- Keine auffälligen Schäden, kein sichtbarer Schimmel.
- Insgesamt „verkaufsfähig“ bzw. Ziel-Kategorie.

**`unbekannt`**  
Objekte, bei denen die KI nicht sicher ist, ob es überhaupt eine Heidelbeere ist:
- Hintergrund, Schale, Hände, Etiketten usw.
- Andere Objekte, die wie Beeren aussehen könnten, aber keine sind.
- Auch extrem schlechte, stark verwackelte oder überbelichtete Bilder können hier landen.  
Diese Kategorie ist nur **intern** relevant und wird der Nutzerin/dem Nutzer in der GUI nicht als eigene Klasse angezeigt.

---

## 4. Klassische Entscheidung (Fallback)

Zusätzlich zu den KI-Modellen gibt es eine **klassische Regel-Entscheidung**.

Vereinfacht gesagt:
- Für jede, als **Green**, erkannte Beere werden zwei **Merkmale** berechnet.
  - Form (wie rund ist die Beere?) und
  - Größe
- Für jedes dieser Merkmale gibt es **Grenzwerte** (Schwellen).
- Beispielablauf
  1. eine Beere geht durch den gesamten Decision Flow (siehe oben) und wird zuletzt als Green eingestuft. 
  2. diese Heidelbeere ist sehr groß und übersteigt den Grenzwert für die Größe in Pixeln.
  3. da nur eine der beiden Merkmale erfüllt sein muss, wird die Heidelbeere nicht als Green sondern als Yellow in der GUI gekennzeichnet.
  4. oben links steht jetzt nicht mehr die Sicherheit des KI Modells in Prozent sondern die Größe der Heidelbeere, zusammen mit dem Grund (z.B. too_large).

---

## 5. Wichtige Hinweise zur Bildqualität

- Bitte möglichst **scharfe, nicht verschwommene** Fotos machen.
- Kamera ruhig halten, Beeren gut ausleuchten, aber nicht überbelichten.
- Verschwommene oder verwackelte Bilder führen mit hoher Wahrscheinlichkeit dazu, dass viele Beeren in der Kategorie `Never` oder `unbekannt` landen, obwohl sie in Wirklichkeit gut sind.

Je besser die Bildqualität, desto zuverlässiger arbeitet die KI!

---

## 6. Thresholds (Schwellenwerte) anpassen

Die Datei `Thresholds.json` in diesem Ordner speichert:
- die **Labels**,
- den **Decision Flow**,
- und verschiedene **Schwellenwerte** (z. B. `threshold` bei A1, A2, A3, A4).

Wenn Sie hier Werte anpassen (z. B. strenger oder großzügiger bewerten möchten), beachten Sie bitte:

1. Datei `Thresholds.json` bearbeiten und speichern.
2. Die GUI vollständig schließen.
3. Die Anwendung **neu starten** (z. B. mit `run_windows_no_install.bat`).

Änderungen an den Thresholds werden erst nach einem **Neustart der GUI** übernommen.

---

## 7. Kurzzusammenfassung

- **Erster Start auf einem neuen PC:**  
  `run_windows.bat` (installiert Python + Umgebung, startet GUI).

- **Alle weiteren Starts:**  
  `run_windows_no_install.bat` (nutzt bestehende Umgebung, startet nur die GUI).

- **Keine verschwommenen Fotos:**  
  Sonst landen viele Beeren fälschlich in `Never` oder `unbekannt`.

- **Thresholds geändert?**  
  `Thresholds.json` speichern → GUI schließen → mit `.bat` neu starten, damit die Änderungen wirken.

