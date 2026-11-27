"""Shorts Studio as an app: no console window to keep open.

    pythonw launcher.pyw                 open the app (starting it first when needed)      <- the desktop icon
    pythonw launcher.pyw --background    only keep it running, restart it if it crashes  <- Windows sign-in
    python  launcher.pyw --install       desktop + Start menu icon, start with Windows
    python  launcher.pyw --uninstall     remove them again

Only the standard library here: this file must work even when something else is broken.
"""
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("SHORTS_STUDIO_PORT", "8765"))
URL = f"http://127.0.0.1:{PORT}"
QUIT_FLAG = ROOT / "output" / "state" / "quit.flag"           # written by the app's Quit button
SERVER_LOG = ROOT / "output" / "logs" / "server.log"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def running(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(URL + "/api/config", timeout=timeout) as r:
            return "version" in json.load(r)
    except Exception:  # noqa: BLE001
        return False


def python_exe(windowless: bool = False) -> str:
    name = "pythonw.exe" if windowless else "python.exe"
    local = ROOT / "venv" / "Scripts" / name
    return str(local if local.exists() else Path(sys.executable).with_name(name) if os.name == "nt" else sys.executable)


def start_server() -> subprocess.Popen:
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    if SERVER_LOG.exists() and SERVER_LOG.stat().st_size > 3_000_000:
        SERVER_LOG.replace(SERVER_LOG.with_name("server.old.log"))
    with open(SERVER_LOG, "ab") as log:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
        # python.exe with no window (not pythonw): ffmpeg started by the app then stays invisible too
        return subprocess.Popen([python_exe(), str(ROOT / "app.py"), "--no-browser", "--port", str(PORT)], cwd=str(ROOT),
                                stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=env, creationflags=NO_WINDOW)


def _only_keeper() -> bool:
    """One keeper per computer user (a second sign-in script or double start just leaves)."""
    if os.name != "nt":
        return True
    ctypes.windll.kernel32.CreateMutexW(None, False, f"ShortsStudioKeeper{PORT}")
    return ctypes.windll.kernel32.GetLastError() != 183               # ERROR_ALREADY_EXISTS


def keep_running() -> None:
    """Start the app and bring it back if it dies; stop when the person pressed Quit."""
    if not _only_keeper():
        return
    QUIT_FLAG.unlink(missing_ok=True)
    crashes: list = []
    while not QUIT_FLAG.exists():
        if running():                                          # started some other way (run_ui.bat): just watch
            time.sleep(20)
            continue
        child = start_server()
        child.wait()
        if QUIT_FLAG.exists():
            break
        crashes = [t for t in crashes if time.time() - t < 600] + [time.time()]
        if len(crashes) >= 5:                                  # something is broken: do not loop forever (see server.log)
            break
        time.sleep(5)


def window_command():
    """Edge or Chrome in "app" mode: a window of its own, no tabs or address bar."""
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        for relative in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"):
            path = Path(base or "/nowhere") / relative
            if path.exists():
                return [str(path), f"--app={URL}"]
    return None


def open_app() -> None:
    if not running():
        QUIT_FLAG.unlink(missing_ok=True)
        subprocess.Popen([python_exe(windowless=True), str(ROOT / "launcher.pyw"), "--background"], cwd=str(ROOT), creationflags=NO_WINDOW)
        for _ in range(90):
            if running(1.0):
                break
            time.sleep(0.7)
        else:
            message = f"Shorts Studio did not start. Details are in:\n{SERVER_LOG}"
            if os.name == "nt":
                ctypes.windll.user32.MessageBoxW(None, message, "Shorts Studio", 0x10)
            return
    command = window_command()
    if command:
        subprocess.Popen(command)
    else:
        webbrowser.open(URL)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    os.chdir(ROOT)
    if "--background" in args:
        keep_running()
        return 0
    if "--install" in args or "--uninstall" in args:
        sys.path.insert(0, str(ROOT))
        from shorts_generator.local import desktop

        if "--install" in args:
            for line in desktop.install():
                print(line)
        else:
            for line in desktop.uninstall():
                print(line)
        return 0
    open_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
