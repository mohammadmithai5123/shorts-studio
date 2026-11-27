@echo off
rem Double-click once: desktop + Start menu icon for Shorts Studio, and start it with Windows.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" call "%~dp0setup.bat"
if not exist "venv\Scripts\python.exe" exit /b 1
"venv\Scripts\python.exe" launcher.pyw --install
start "" "venv\Scripts\pythonw.exe" launcher.pyw
pause
