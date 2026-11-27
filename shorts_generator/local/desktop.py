"""Desktop icon, Start menu entry and "start with Windows" for Shorts Studio (Windows only).

Everything is per user and easy to undo: two shortcut files and one value under
HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from ..config import ROOT

NAME = "Shorts Studio"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "ShortsStudio"
ICON = ROOT / "assets" / "shortsstudio.ico"


def supported() -> bool:
    return sys.platform == "win32"


def _pythonw() -> str:
    local = ROOT / "venv" / "Scripts" / "pythonw.exe"
    return str(local if local.exists() else Path(sys.executable).with_name("pythonw.exe"))


def start_command() -> str:
    return f'"{_pythonw()}" "{ROOT / "launcher.pyw"}" --background'


def autostart_enabled() -> bool:
    if not supported():
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, RUN_VALUE)
        return bool(value)
    except OSError:
        return False


def set_autostart(on: bool) -> None:
    if not supported():
        raise RuntimeError("Starting with the computer is only set up for Windows.")
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
        if on:
            winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, start_command())
        else:
            try:
                winreg.DeleteValue(key, RUN_VALUE)
            except FileNotFoundError:
                pass


def _shortcut_folders() -> List[Path]:
    """Desktop (asked from Windows: it is often inside OneDrive) and the Start menu."""
    proc = subprocess.run(["powershell", "-NoProfile", "-Command",
                           "[Environment]::GetFolderPath('Desktop'); [Environment]::GetFolderPath('Programs')"],
                          capture_output=True, text=True, timeout=60)
    return [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def _quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _make_shortcut(path: Path) -> bool:
    script = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({_quote(path)}); "
              f"$s.TargetPath = {_quote(_pythonw())}; $s.Arguments = {_quote(chr(34) + str(ROOT / 'launcher.pyw') + chr(34))}; "
              f"$s.WorkingDirectory = {_quote(ROOT)}; $s.IconLocation = {_quote(ICON)}; "
              f"$s.Description = 'Make and post YouTube Shorts'; $s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, timeout=60)
    return path.exists()


def install(folders: List[Path] = None) -> List[str]:
    if not supported():
        return ["Only Windows is set up for a desktop icon; start the app with: python app.py"]
    done = []
    for folder in folders if folders is not None else _shortcut_folders():
        path = Path(folder) / f"{NAME}.lnk"
        done.append(f"icon created: {path}" if _make_shortcut(path) else f"could not create {path}")
    set_autostart(True)
    done.append("Shorts Studio now starts by itself when you sign in to Windows (Settings -> App to switch off).")
    return done


def uninstall(folders: List[Path] = None) -> List[str]:
    if not supported():
        return []
    done = []
    for folder in folders if folders is not None else _shortcut_folders():
        path = Path(folder) / f"{NAME}.lnk"
        if path.exists():
            os.remove(path)
            done.append(f"removed: {path}")
    set_autostart(False)
    done.append("Shorts Studio no longer starts with Windows.")
    return done
