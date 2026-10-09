"""
Entry point for the Homodyne Plane Test GUI.

Set the flags below (same style as the other GUIs), then run this file
(e.g. from PyCharm), or pass --dummy on the command line.

The eye-lens Zaber is attached WITHOUT homing or moving it, so start this
with the lens already at the reflection plane. Nothing moves until you
press "Back off + slew".

Wiring (USB-6008 analog terminal block, RSE):
    PD H (PBS output 1): signal -> pin 2 (AI0), shield -> pin 1 (GND)
    PD V (PBS output 2): signal -> pin 5 (AI1), shield -> pin 4 (GND)
"""

from __future__ import annotations

import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from brillouin_system.guis.homodyne_test.homodyne_test_gui import HomodyneTestWindow
from brillouin_system.logging_utils.logging_setup import install_crash_hooks, start_logging

use_dummy = False       # simulated eye lens + simulated 2-channel homodyne DAQ
ZABER_PORT = "COM5"     # eye-lens Zaber (same default as ZaberEyeLens)
NI_DEVICE = "Dev1"      # USB-6008


def main():
    try:
        if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy"):
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
            )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyleSheet("* { font-size: 8pt; }")
    window = HomodyneTestWindow(
        use_dummy=use_dummy or "--dummy" in sys.argv,
        zaber_port=ZABER_PORT,
        ni_device=NI_DEVICE,
    )
    window.resize(1400, 900)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # IMPORTANT: On Windows, spawn the logging writer process ONLY here.
    start_logging()
    install_crash_hooks()

    main()
