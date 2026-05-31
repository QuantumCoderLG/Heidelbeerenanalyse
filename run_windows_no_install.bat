@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Blueberry QA - Start ohne Installationsschritte (setzt bestehende .venv voraus)

REM In Projektverzeichnis wechseln (Skript liegt dort)
pushd "%~dp0" 2>nul
if errorlevel 1 (
  echo Konnte nicht ins Projektverzeichnis wechseln: %~dp0
  pause
  exit /b 1
)

set "APP_DIR=nicht_anfassen"
if exist "%APP_DIR%\inference_gui.py" (
  rem Neue Struktur mit ausgelagertem Python-Code
) else (
  set "APP_DIR=."
)
set "GUI_SCRIPT=%APP_DIR%\inference_gui.py"
if /I "%APP_DIR%"=="." (
  set "VENV_DIR=.venv"
) else (
  set "VENV_DIR=%APP_DIR%\.venv"
)
set "VPYW=%VENV_DIR%\Scripts\pythonw.exe"
set "VPY=%VENV_DIR%\Scripts\python.exe"

if not exist "%GUI_SCRIPT%" (
  echo Fehler: %GUI_SCRIPT% wurde nicht gefunden.
  echo Bitte pruefe, ob das Paket vollstaendig entpackt wurde.
  pause
  popd
  exit /b 1
)

echo Starte Anwendung (bestehende Umgebung wird verwendet)

if exist "%VPYW%" (
  start "" "%VPYW%" "%GUI_SCRIPT%"
  goto :started
)

if exist "%VPY%" (
  start "" "%VPY%" "%GUI_SCRIPT%"
  goto :started
)

echo Fehler: Erwartete virtuelle Umgebung %VENV_DIR% wurde nicht gefunden.
echo Bitte fuehre einmal run_windows.bat aus.
pause
popd
exit /b 1

:started
echo Die Anwendung wurde gestartet. Dieses Fenster kann geschlossen werden.
popd
exit /b 0
