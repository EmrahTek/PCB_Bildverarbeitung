OpenCV – Template Matching (matchTemplate):
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

OpenCV – ORB Features:
https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

OpenCV – Feature Matching (BFMatcher/FLANN):
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

OpenCV – Geometrische Transformationen (warpPerspective):
https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html

Python – venv (Tutorial):
https://docs.python.org/3/tutorial/venv.html

Python Packaging Guide – pip + venv + requirements:
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

Python – Logging (HOWTO) + RotatingFileHandler:
https://docs.python.org/3/howto/logging.html
https://docs.python.org/3/library/logging.handlers.html

Git – Pro Git Book (Branching):
https://git-scm.com/book/en/v2
https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging

GitHub Flow:
https://docs.github.com/en/get-started/using-github/github-flow

pytest – Getting started:
https://docs.pytest.org/en/stable/getting-started.html

Qt for Python (offizielles OpenCV Beispiel; Konzept übertragbar auf PyQt6):
https://doc.qt.io/qtforpython-6.5/examples/example_external_opencv.html

Picamera2 offizielles OpenCV Beispiel (für spätere Raspberry Pi Migration):
https://github.com/raspberrypi/picamera2/blob/main/examples/opencv_face_detect.py



# PCB Component Detection – Developer Guide & Implementation Manual (Bildverarbeitung 1)

**Projekt:** Echtzeit-Erkennung von PCB-Komponenten (ESP32 FireBeetle, Reset-Button, JST, USB)  
**Stack:** Python, OpenCV, Git/GitHub, venv, YAML-Konfig, pytest  
**Ziel:** Live-Feed mit Bounding Boxes + Labels + Counts, stabil & robust, sauber geloggt, testbar  
**Zeitrahmen:** 45 Tage, ca. 4h/Woche pro Person (2 Studierende)

---

## 0. Arbeitsprinzipien (damit ihr wirklich “stabil + nachhaltig” baut)

1. **Stabilität zuerst:** Logging, deterministische Tests (Video/Images), klarer Pipeline-Fluss.
2. **Board-Normalisierung als Fundament:** Erst PCB finden + perspektivisch “warp”, dann Bauteile detektieren.
3. **Konfiguration statt Magic Numbers:** Alle Thresholds/Parameter in `config/default.yaml`.
4. **Keine Webcam-Abhängigkeit in Tests:** Unit-Tests laufen auf `assets/test_images/` und/oder Video-Datei.

---

# 1. The Execution Masterplan (Roadmap to Success)

## 1.1 Wochenplan (Woche 1–3) – exakt welche Dateien wer codet

### Woche 1 – “Skeleton + Stabilität + Logging + CLI”
**Ziel:** App startet zuverlässig, liest Konfig, loggt sauber, kann Webcam ODER Video-File verwenden.  
**Deliverable:** `python main.py --help` + `python main.py --camera 0` zeigt Live-Fenster (oder headless mit Logs).

**Person A (Ubuntu) – Pipeline/Infra Owner**
- `src/logging/setup.py`
- `src/app/pipeline.py`
- `src/app/cli.py`
- `src/camera_input/base.py`
- `src/camera_input/video_file.py` (für deterministische Tests!)
- `src/utils/io.py` (YAML laden, Pfade, Debug-Output)
- `src/utils/types.py` (Dataclasses)

**Person B (Windows) – Rendering/UX + Tests/Assets Owner**
- `src/render/overlay.py` (nur Basis: “draw box + label + count panel”)
- `src/render/fps.py`
- `tests/test_pipeline_smoke.py` (smoke test: “läuft 10 Frames” über Testbilder/Video)
- `assets/test_images/` aufbauen (mind. 20 Bilder in 3 Lichtvarianten)
- Ordnerstruktur `assets/templates/` vorbereiten (noch ohne Logik)

**Woche-1 Test-Checkliste (vor Merge)**
- `pytest -q` läuft grün (mind. 1 Smoke Test).
- `python main.py --video path/to/video.mp4 --headless` läuft ohne Crash und loggt Frames.
- Logging schreibt in `logs/app.log` (Rotation aktiv oder vorbereitet).

---

### Woche 2 – “Board Find + Perspective Warp + Preprocessing”
**Ziel:** PCB wird im Frame erkannt und perspektivisch auf Standardansicht normalisiert.  
**Deliverable:** Debug-Overlay zeigt Board-Ecken, und es gibt “warped frames” zum Kontrollieren.

**Person A (Ubuntu)**
- `src/preprocessing/geometry.py` (Board finden, Eckpunkte, Homographie, Warp)
- `tests/test_geometry.py` (Unit Tests mit festen Testbildern: “board found / homography ok”)

**Person B (Windows)**
- `src/preprocessing/color.py` (Gray/HSV; einfache Normalisierung)
- `src/preprocessing/filters.py` (Blur/CLAHE/Canny als Bausteine)
- `assets/test_images/` erweitern (mind. 50 Bilder; inkl. leicht schräg, leicht unscharf)
- `tests/test_template_matching.py` vorbereiten (noch “skip”/placeholder erlaubt, aber Struktur anlegen)

**Woche-2 Test-Checkliste**
- `pytest -q` grün.
- `python main.py --video ... --debug` speichert `warped_*.png` (oder ähnliches) und Logs sagen klar: “board found: True/False”.

---

### Woche 3 – “ESP32 Baseline: Template Matching + NMS + Overlay”
**Ziel:** ESP32 FireBeetle zuverlässig detektieren (Baseline), Counts im Overlay.  
**Deliverable:** Live- oder Video-Demo mit Box+Label+Count für ESP32.

**Person A (Ubuntu)**
- `src/detection_logic/base.py` (Detector Interface)
- `src/detection_logic/postprocess.py` (NMS, score filtering, counting helpers)
- Pipeline anpassen: Detection-Stage integriert (via Interface)

**Person B (Windows)**
- `src/detection_logic/template_match.py` (Multi-Scale Template Matching + threshold)
- `assets/templates/esp32_module/` erstellen (mind. 10 Templates: verschiedene Licht/Schärfe)
- `tests/test_template_matching.py` (Detektion auf Testbildern: “mind. X/ Y Treffer”)

**Woche-3 Test-Checkliste**
- `pytest -q` grün (inkl. template matching test).
- “Regression test” (manuell): 10 bekannte Frames → ähnliche Detection Count wie vorher.
- PR enthält: kurze Notes “wie Templates erstellt wurden” (README oder `assets/templates/README.md`).

---

## 1.2 Wie ihr testet, bevor ihr merged (Pflicht-Prozess)

### A) Lokale Pre-Merge Checks (jeder PR)
1. **Unit/Smoke Tests**
   - `pytest -q`
2. **Minimaler Lauf (ohne GUI)**
   - `python main.py --video assets/test_images_or_video --headless --max-frames 50`
3. **Style/Static (empfohlen, wenn ihr es installiert)**
   - `python -m ruff check .`
   - `python -m black --check .`
   - `python -m isort --check-only .`

### B) PR-Review Checkliste (Reviewer muss abhaken)
- [ ] Keine Änderung an “fremden Ownership-Dateien” ohne Absprache.
- [ ] Konfig-Änderungen dokumentiert (Parameter/Thresholds begründet).
- [ ] Logs sind sinnvoll (keine Spams; Warnungen bei Fehlerfällen).
- [ ] Tests decken neues Verhalten ab (mind. 1 Test oder Smoke).
- [ ] Kein Hardcoding von Pfaden; `utils/io.py` genutzt.

---

# 2. Deep Dive: File-by-File Blueprint & Resources

> **Hinweis:** Alle “Links” sind als Codeblock geschrieben, damit ihr sie 1:1 kopieren könnt.

---

## 2.1 `src/utils/types.py`
### Zweck
Zentrale Datentypen als `dataclasses` für saubere Schnittstellen (BBox, Detection, FrameMeta, optional Config).

### Zu implementierende Funktionen / Klassen
- `@dataclass class BBox: x, y, w, h`
- `@dataclass class Detection: label, score, bbox`
- `@dataclass class FrameMeta: frame_id, timestamp, source`
- (Optional) `@dataclass class TemplateSpec` (label, path, method, thresholds)

### Lernressourcen & Links
```text
Python dataclasses:
https://docs.python.org/3/library/dataclasses.html

Python typing (List, Optional, Protocol):
https://docs.python.org/3/library/typing.html
