"""Sound cues and spoken results, for a subject who cannot see the screen. Played on their own thread so the
timing of a find is untouched."""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
import time

try:
    import winsound
except ImportError:     # not on Windows: terminal bell, enough for dummy runs
    winsound = None

PATTERNS = {                # (Hz, ms); frequency 0 = silence
    "count": [(800, 120)],
    "go": [(1200, 300)],
    "miss": [(300, 80)],
    "success": [(800, 150), (1200, 150), (1600, 250)],
    "partial": [(800, 200), (500, 300)],
    "failed": [(300, 500), (0, 150), (300, 500), (0, 150), (300, 500)],
    "stopped": [(500, 600)],
}

_queue: queue.Queue = queue.Queue()
_started = False


def speech_command(text: str) -> list[str] | None:
    """Text-to-speech with what the OS already has; None if it has nothing. Text comes from this program."""
    clean = "".join(c for c in text if c.isalnum() or c in " ,.-")
    if sys.platform == "win32":
        return ["powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean}')"]
    if shutil.which("say"):
        return ["say", clean]
    return None


def _player() -> None:
    while True:
        item = _queue.get()
        if isinstance(item, tuple):
            cmd = speech_command(item[1])
            try:
                if cmd is not None:
                    subprocess.run(cmd, timeout=30, check=False, capture_output=True)
            except Exception as e:
                print(f"[cues] speech failed: {e}", flush=True)
            continue
        for hz, ms in PATTERNS[item]:
            if hz and winsound is not None:
                winsound.Beep(hz, ms)
            else:
                if hz:
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                time.sleep(ms / 1000.0)


def _put(item) -> None:
    global _started
    if not _started:
        _started = True
        threading.Thread(target=_player, name="SoundCues", daemon=True).start()
    _queue.put(item)


def play(name: str) -> None:
    if name not in PATTERNS:
        raise ValueError(f"Unknown cue {name}")
    _put(name)


def say(text: str) -> None:
    _put(("say", text))
