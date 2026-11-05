"""
Entry point for the plane precision + accuracy run (phantom only).

Close the human interface GUI first: the Andor, the Zabers and the DAQ serve one
program at a time. Set the flags below, then run this file, or pass --dummy.
"""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from brillouin_system.guis.plane_accuracy_test.plane_accuracy_gui import PlaneAccuracyWindow
from brillouin_system.logging_utils.logging_setup import install_crash_hooks, start_logging

use_dummy = False       # simulated lens, stage, shutters, DAQ and Andor
LENS_PORT = "COM5"
STAGE_PORT = "COM6"
NI_DEVICE = "Dev1"


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("* { font-size: 8pt; }")
    window = PlaneAccuracyWindow(use_dummy=use_dummy or "--dummy" in sys.argv, lens_port=LENS_PORT,
                                 stage_port=STAGE_PORT, ni_device=NI_DEVICE)
    window.resize(1500, 950)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start_logging()
    install_crash_hooks()
    main()
