"""
Entry point for the Homodyne Plane Test GUI.

Set the flags below (same style as the other GUIs), then run this file
(e.g. from PyCharm), or pass --dummy on the command line.

The eye-lens Zaber and the rig stage are attached WITHOUT homing or moving
them, so start this with the lens already at the reflection plane. Nothing
moves until you press a move button or "Back off + slew".

Eye tracking (stereo cameras) runs as in the main GUI: Move XY moves the rig
stage so the laser lands at (R, phi) relative to the pupil; Move Z moves the
eye lens to the target delta_c (this leaves the reflection plane; use a slew
and "Move lens to last slew peak" to get back onto the surface).

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

use_dummy = False               # simulated eye lens, rig stage, shutters and 2-channel DAQ
include_eye_tracking = True
use_eye_tracker_dummy = False   # dummy stereo cameras (independent of use_dummy)
ZABER_PORT = "COM5"             # eye-lens Zaber (same default as ZaberEyeLens)
STAGE_PORT = "COM6"             # rig XYZ stage (same default as ZaberHumanInterface)
NI_DEVICE = "Dev1"              # USB-6008


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
    dummy = use_dummy or "--dummy" in sys.argv
    window = HomodyneTestWindow(
        use_dummy=dummy,
        zaber_port=ZABER_PORT,
        ni_device=NI_DEVICE,
        stage_port=STAGE_PORT,
        include_eye_tracking=include_eye_tracking,
        use_eye_tracker_dummy=use_eye_tracker_dummy or dummy,
    )
    window.resize(1700, 950)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # IMPORTANT: On Windows, spawn the logging writer process ONLY here.
    start_logging()
    install_crash_hooks()

    main()
