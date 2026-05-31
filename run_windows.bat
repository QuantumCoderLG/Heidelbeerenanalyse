@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Blueberry QA - Start per Doppelklick (Windows, mit Python)
REM Zweck: Automatisch Python (>=3.11) sicherstellen, venv anlegen, Abhaengigkeiten installieren, GUI starten

REM ------------------------------------------------------------
REM 0) Adminrechte sicherstellen (UAC-Elevation) fuer AllUsers-Installation
REM ------------------------------------------------------------
if /I not "%~1"=="ELEVATED" (
  fltmc >nul 2>&1
  if errorlevel 1 (
    echo [0/3] Fordere Administratorrechte an
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -ArgumentList 'ELEVATED' -Verb RunAs" 2>nul
    if errorlevel 1 (
      echo Konnte keine Administratorrechte erhalten. Bitte Rechtsklick ^> Als Administrator ausfuehren.
      pause
    )
    exit /b
  )
)

REM In Projektverzeichnis wechseln (auch wenn von UNC/Netzpfad gestartet)
pushd "%~dp0"
if errorlevel 1 (
  echo Konnte nicht ins Projektverzeichnis wechseln: %~dp0
  pause
  exit /b 1
)

set "APP_DIR=nicht_anfassen"
if exist "%APP_DIR%\inference_gui.py" (
  rem Neue Struktur: Python-Dateien liegen in nicht_anfassen
) else (
  set "APP_DIR=."
)
set "GUI_SCRIPT=%APP_DIR%\inference_gui.py"
if /I "%APP_DIR%"=="." (
  set "VENV_DIR=.venv"
) else (
  set "VENV_DIR=%APP_DIR%\.venv"
)
if not exist "%GUI_SCRIPT%" (
  echo Fehler: %GUI_SCRIPT% wurde nicht gefunden.
  echo Bitte pruefe, ob das Paket vollstaendig entpackt wurde.
  pause
  popd
  exit /b 1
)
set "PY="
set "PY_VER_INSTALL=3.11.9"
set "DEPS_STAMP=%VENV_DIR%\_deps_installed.stamp"

REM ------------------------------------------------------------
REM 0.5) Bereits installierte Standardpfade fuer Python 3.11 pruefen
REM      (verhindert erneuten Download, falls PATH noch nicht aktualisiert ist)
REM ------------------------------------------------------------
set "_PY311_SYS=%ProgramFiles%\Python311\python.exe"
set "_PY311_USER=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%_PY311_SYS%" set PY="%_PY311_SYS%"
if not defined PY if exist "%_PY311_USER%" set PY="%_PY311_USER%"

REM Wenn wir hier schon PY haben, ueberspringen wir die PATH-Suche
if defined PY goto :have_python

REM ------------------------------------------------------------
REM 1) Geeigneten Python-Interpreter (>=3.11) ermitteln
REM    - bevorzuge den neuesten 3.x via py -3
REM    - sonst pruefe python im PATH
REM ------------------------------------------------------------
for /f "tokens=*" %%# in ("1") do (
  REM Versuche py -3
  py -3 -V >nul 2>&1
  if not errorlevel 1 (
    for /f "tokens=2 delims= " %%v in ('py -3 -V 2^>^&1') do set "_ver=%%v"
    for /f "tokens=1-3 delims=." %%a in ("!_ver!") do (
      set "_major=%%a" & set "_minor=%%b" & set "_patch=%%c"
    )
    if defined _major if defined _minor (
      if !_major! gtr 3 ( set "PY=py -3" ) else (
        if !_major! equ 3 if !_minor! geq 11 set "PY=py -3"
      )
    )
  )

  REM Falls noch nicht gefunden: pruefe python direkt
  if not defined PY (
    python -V >nul 2>&1
    if not errorlevel 1 (
      for /f "tokens=2 delims= " %%v in ('python -V 2^>^&1') do set "_ver=%%v"
      for /f "tokens=1-3 delims=." %%a in ("!_ver!") do (
        set "_major=%%a" & set "_minor=%%b" & set "_patch=%%c"
      )
      if defined _major if defined _minor (
        if !_major! gtr 3 ( set "PY=python" ) else (
          if !_major! equ 3 if !_minor! geq 11 set "PY=python"
        )
      )
    )
  )
)

:have_python

REM ------------------------------------------------------------
REM 2) Falls kein Python >=3.11 gefunden: Python 3.11.9 herunterladen + installieren
REM    - Architektur erkennen (AMD64/x86/ARM64)
REM    - per-user Silent-Install, PATH wird gesetzt (wir nutzen aber Direktpfad)
REM ------------------------------------------------------------
if not defined PY (
  echo [0/3] Python %PY_VER_INSTALL% wird installiert (kein Python >=3.11 gefunden)

  set "ARCH=%PROCESSOR_ARCHITECTURE%"
  if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"

  set "PY_URL="
  if /I "%ARCH%"=="AMD64" set "PY_URL=https://www.python.org/ftp/python/%PY_VER_INSTALL%/python-%PY_VER_INSTALL%-amd64.exe"
  if /I "%ARCH%"=="ARM64" set "PY_URL=https://www.python.org/ftp/python/%PY_VER_INSTALL%/python-%PY_VER_INSTALL%-arm64.exe"
  if /I "%ARCH%"=="x86"   set "PY_URL=https://www.python.org/ftp/python/%PY_VER_INSTALL%/python-%PY_VER_INSTALL%.exe"

  if not defined PY_URL (
    echo Unerwartete Architektur "%ARCH%". Versuche AMD64-Installer.
    set "PY_URL=https://www.python.org/ftp/python/%PY_VER_INSTALL%/python-%PY_VER_INSTALL%-amd64.exe"
  )

  set "PY_DL=%TEMP%\python-%PY_VER_INSTALL%.exe"
  echo Lade Installer herunter: %PY_URL%
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_DL%' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
  if errorlevel 1 (
    echo PowerShell-Download fehlgeschlagen, versuche CertUtil
    certutil -urlcache -split -f "%PY_URL%" "%PY_DL%" >nul 2>&1
  )
  if not exist "%PY_DL%" (
    echo Fehler beim Herunterladen des Python-Installers.
    echo Bitte pruefe die Internetverbindung und versuche es erneut.
    pause
    exit /b 1
  )

  echo Fuehre Silent-Installation (AllUsers) durch
  "%PY_DL%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 SimpleInstall=1 Include_launcher=1 InstallLauncherAllUsers=1
  set "_install_ec=%ERRORLEVEL%"
  if not "%_install_ec%"=="0" (
    echo Python-Installation meldete Fehlercode %_install_ec%.
    echo Eventuell sind Administratorrechte erforderlich oder Sicherheitssoftware blockiert die Installation.
    pause
    exit /b 1
  )

  REM Nach Installation: Direktpfad suchen (AllUsers bevorzugt; PATH in diesem Prozess nicht neu geladen)
  set "PY_HOME_SYS=%ProgramFiles%\Python311"
  if exist "%PY_HOME_SYS%\python.exe" (
    set PY="%PY_HOME_SYS%\python.exe"
  ) else (
    set "PY_HOME_USER=%LOCALAPPDATA%\Programs\Python\Python311"
    if exist "%PY_HOME_USER%\python.exe" set PY="%PY_HOME_USER%\python.exe"
  )
  if not defined PY (
    REM Als Fallback versuchen wir den Launcher
    py -3.11 -V >nul 2>&1 && set "PY=py -3.11"
  )

  if not defined PY (
    echo Konnte den installierten Python nicht finden.
    echo Bitte starte dieses Skript neu oder fuege Python manuell dem PATH hinzu.
    pause
    exit /b 1
  )
)

echo [1/3] Erzeuge/aktualisiere virtuelle Umgebung

REM Falls eine defekte venv existiert (ohne Scripts\python.exe), entferne sie
if exist "%VENV_DIR%" if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Vorhandene virtuelle Umgebung scheint unvollstaendig. Entferne .venv
  rmdir /s /q "%VENV_DIR%"
)

echo Verwende Interpreter: %PY%
%PY% -m venv "%VENV_DIR%"
if not exist "%VENV_DIR%\Scripts\python.exe" goto :venv_error

set "VPY=%VENV_DIR%\Scripts\python.exe"
set "VPYW=%VENV_DIR%\Scripts\pythonw.exe"

REM Pruefen, ob Abhaengigkeiten bereits installiert sind (Import-Test + Markerdatei)
set "NEED_INSTALL=1"
set "IMPORTS_OK="
"%VPY%" -c "import importlib; [importlib.import_module(m) for m in ['onnxruntime','cv2','PIL','numpy']]" >nul 2>&1
if not errorlevel 1 set "IMPORTS_OK=1"
if defined IMPORTS_OK if exist "%DEPS_STAMP%" set "NEED_INSTALL="

if defined NEED_INSTALL (
  echo [2/3] Installiere Laufzeit-Abhaengigkeiten (einmalig, Internet erforderlich)
  call "%VPY%" -m ensurepip --upgrade
  call "%VPY%" -m pip install --upgrade pip
  REM Minimal: ONNXRuntime (CPU), OpenCV, Pillow, NumPy (kompatibel zu OpenCV)
  call "%VPY%" -m pip install "onnxruntime>=1.20.1" opencv-python Pillow "numpy>=2,<2.3.0"
  if errorlevel 1 (
    echo Fehler beim Installieren der Abhaengigkeiten.
    echo Pruefe Internetzugang oder versuche es erneut.
    pause
    exit /b 1
  )
  >"%DEPS_STAMP%" echo ok
) else (
  echo [2/3] Abhaengigkeiten vorhanden — Installation uebersprungen
)

echo [3/3] Starte Anwendung
if exist "%VPYW%" (
  start "" "%VPYW%" "%GUI_SCRIPT%"
) else (
  start "" "%VPY%" "%GUI_SCRIPT%"
)

echo Die Anwendung wurde gestartet. Dieses Fenster kann geschlossen werden.
popd
exit /b 0

:venv_error
echo Fehler: Virtuelle Umgebung konnte nicht erstellt werden.
echo Moegliche Ursachen:
echo   - Python >=3.11 nicht installiert oder nicht im PATH
echo   - Ausfuehrung von einem Netzwerk/UNC-Pfad; bitte Projekt nach C:\Users\%USERNAME%\Downloads oder Dokumente kopieren
echo   - Antiviren-Software blockiert das Anlegen von Skripten
echo Diagnose:
%PY% -V 2>&1
echo Arbeitsverzeichnis: %CD%
dir /b "%VENV_DIR%" 2>nul
pause
exit /b 1
