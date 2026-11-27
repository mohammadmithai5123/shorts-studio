@echo off
rem One-time setup for Shorts Studio on Windows.
rem Creates the "venv" folder and installs what the tool needs (about 200 MB, a few minutes the first time).
rem   setup.bat            normal install
rem   setup.bat whisper    also installs speech recognition (only for videos that have no captions at all)
setlocal
cd /d "%~dp0"

set "PY="
for %%C in (py python) do (
  if not defined PY (
    %%C -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=%%C"
  )
)
if not defined PY (
  echo Python 3.10 or newer was not found.
  echo Install it from https://www.python.org/downloads/  and tick "Add python.exe to PATH", then run this again.
  pause
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo Creating the virtual environment...
  %PY% -m venv venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)

echo Installing the requirements...
"venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-local.txt
if errorlevel 1 (
  echo.
  echo The install failed. Check your internet connection and run setup.bat again - it continues where it stopped.
  pause
  exit /b 1
)
if /i "%~1"=="whisper" (
  echo Installing speech recognition...
  "venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-whisper.txt
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo ffmpeg was not found. Install it with:   winget install Gyan.FFmpeg
  echo then close this window and open a NEW one, so the PATH is refreshed.
)

echo.
set PYTHONIOENCODING=utf-8
"venv\Scripts\python.exe" main.py --doctor
echo.
echo Setup finished. Double-click run_ui.bat to start Shorts Studio.
if not defined SHORTS_NO_PAUSE pause
endlocal
