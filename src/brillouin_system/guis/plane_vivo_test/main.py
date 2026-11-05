"""
Entry point for in vivo reflection plane finder testing.

Close the human interface GUI first: the Andor, the Zabers, the DAQ and the eye
cameras serve one program at a time. Options: --dummy, --no-eye.
"""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from brillouin_system.guis.plane_vivo_test.plane_vivo_gui import PlaneVivoWindow
from brillouin_system.logging_utils.logging_setup import install_crash_hooks, start_logging

LENS_PORT = "COM5"
STAGE_PORT = "COM6"
NI_DEVICE = "Dev1"


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("* { font-size: 8pt; }")
    window = PlaneVivoWindow(use_dummy="--dummy" in sys.argv, use_eye="--no-eye" not in sys.argv, lens_port=LENS_PORT,
                             stage_port=STAGE_PORT, ni_device=NI_DEVICE)
    window.resize(1500, 950)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start_logging()
    install_crash_hooks()
    main()
