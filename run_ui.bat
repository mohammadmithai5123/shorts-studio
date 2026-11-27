@echo off
rem Double-click to start Shorts Studio (opens http://127.0.0.1:8765 in your browser).
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo First start: running the one-time setup...
  call "%~dp0setup.bat"
  if not exist "venv\Scripts\python.exe" exit /b 1
)
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"venv\Scripts\python.exe" app.py
pause
