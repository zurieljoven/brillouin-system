"""
Eye-tracking view for the homodyne test GUI.

Reuses the main human-interface pieces unchanged: EyeTrackerController
(stereo cameras + pupil fit in a subprocess) and get_eye_tracker_results
(laser position in pupil coordinates and delta_c from the laser focus).
The laser focus is computed exactly like hi_frontend: eye-lens z plus the
calibrated laser offset from offset.toml.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget, TextItem
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import (
    LaserOffset, load_laser_coord_system_from_toml,
)
from brillouin_system.eye_tracker.eye_position.coordinates import RigCoord
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults, get_eye_tracker_results
from brillouin_system.guis.human_interface.eye_tracker_controller import EyeTrackerController
from brillouin_system.logging_utils.logging_setup import get_logger

log = get_logger(__name__)


class EyeTrackingPanel(QWidget):
    """Camera views + laser position map. Owns the eye-tracker thread."""

    result_updated = pyqtSignal(object)        # EyeTrackerResults
    set_et_config = pyqtSignal(object)         # EyeTrackerConfig -> controller
    set_et_allied_configs = pyqtSignal(object, object)
    request_eye_shutdown = pyqtSignal()

    def __init__(self, *, use_dummy: bool, parent=None):
        super().__init__(parent)
        self.use_dummy = use_dummy
        self.latest: Optional[EyeTrackerResults] = None
        self._latest_monotonic = 0.0
        self.laser_focus_position: Optional[RigCoord] = None
        try:
            self._laser_offset = load_laser_coord_system_from_toml()
            self.offset_error: Optional[str] = None
        except Exception as e:
            self._laser_offset = LaserOffset(0, 0, 0)
            self.offset_error = f"Laser offset not loaded ({e}); using 0,0,0."
            log.warning(self.offset_error)

        self._build()
        self.eye_thread: Optional[QThread] = None
        self.eye_ctrl: Optional[EyeTrackerController] = None

    # ------------------------------------------------------------------ UI

    def _build(self):
        self.glw = GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.img_vb, self.img = [], []
        for col in range(2):
            vb = pg.ViewBox(lockAspect=True, enableMenu=False)
            vb.invertY(True)
            img = pg.ImageItem(autoDownsample=True)
            vb.addItem(img)
            vb.setBorder((80, 80, 80))
            self.glw.ci.addItem(vb, row=0, col=col)
            self.img_vb.append(vb)
            self.img.append(img)

        # laser map (same layout as hi_frontend._init_laser_position_map)
        vb = pg.ViewBox(lockAspect=True, enableMenu=False)
        vb.setBorder((80, 80, 80))
        vb.setRange(xRange=(-4, 4), yRange=(-4, 4))
        ang = np.linspace(0, 2 * np.pi, 720)
        for r in range(1, 5):
            vb.addItem(pg.PlotDataItem(r * np.cos(ang), r * np.sin(ang), pen=pg.mkPen(150, 150, 150)))
        for a in range(0, 360, 45):
            t = np.deg2rad(a)
            vb.addItem(pg.PlotDataItem([np.cos(t), 4 * np.cos(t)], [np.sin(t), 4 * np.sin(t)],
                                       pen=pg.mkPen(150, 150, 150, style=QtCore.Qt.DashLine)))
        self.target_point = pg.ScatterPlotItem([0.0], [0.0], size=14, symbol="+",
                                               brush=pg.mkBrush(0, 120, 255), pen=pg.mkPen(0, 120, 255, width=2))
        self.laser_point = pg.ScatterPlotItem([0.0], [0.0], size=12, brush=pg.mkBrush("r"), pen=pg.mkPen("r"))
        vb.addItem(self.target_point)
        vb.addItem(self.laser_point)
        self.pos_text = TextItem(text="x=\ny=\nz=\nΔc=", color=(0, 0, 0), anchor=(1, 0))
        vb.addItem(self.pos_text)
        self.pos_text.setPos(4, 4)
        self.glw.ci.addItem(vb, row=1, col=0)

        # delta_c bar: laser at 0, cornea band at [dc, dc + 0.5 mm]
        p = self.glw.ci.addPlot(row=1, col=1)
        p.setLabel("bottom", "Δc: laser (red) to cornea (grey) [mm]")
        p.hideAxis("left")
        p.setYRange(-1, 1, padding=0.0)
        p.setXRange(-3.0, 10.0, padding=0.0)
        p.setMenuEnabled(False)
        p.addItem(pg.ScatterPlotItem([0.0], [0.0], size=10, brush=pg.mkBrush("r"), pen=pg.mkPen("r")))
        self.cornea_band = pg.LinearRegionItem(values=(2.0, 2.5), movable=False,
                                               brush=pg.mkBrush(150, 150, 150, 80),
                                               pen=pg.mkPen(150, 150, 150, 200))
        p.addItem(self.cornea_band)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw)

    # ------------------------------------------------------------------ eye tracker lifecycle

    def start(self):
        self.eye_thread = QThread(self)
        self.eye_ctrl = EyeTrackerController(use_dummy=self.use_dummy)
        self.eye_ctrl.moveToThread(self.eye_thread)
        self.eye_thread.started.connect(self.eye_ctrl.start)
        self.eye_ctrl.frames_ready.connect(self._on_frames)
        self.set_et_config.connect(self.eye_ctrl.send_config)
        self.set_et_allied_configs.connect(self.eye_ctrl.proxy.set_allied_configs)
        self.request_eye_shutdown.connect(self.eye_ctrl.shutdown)
        self.eye_thread.start()

    def shutdown(self, *, settle_s: float = 3.0):
        if self.eye_thread is None:
            return
        self.request_eye_shutdown.emit()
        deadline = time.monotonic() + settle_s     # let the worker process exit
        while time.monotonic() < deadline:
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.02)
        self.eye_thread.quit()
        if not self.eye_thread.wait(10000):
            log.error("Eye tracker thread did not stop within 10 s; continuing anyway.")
        for sig in (self.set_et_config, self.set_et_allied_configs, self.request_eye_shutdown):
            try:
                sig.disconnect()
            except TypeError:
                pass
        self.eye_thread = None
        self.eye_ctrl = None

    def restart(self):
        self.shutdown()
        self.start()

    # ------------------------------------------------------------------ data

    def set_lens_position(self, lens_um: float):
        """Laser focus in rig coordinates [mm], as in hi_frontend.update_zaber_lens_position."""
        o = self._laser_offset
        self.laser_focus_position = RigCoord(x=o.dx / 1000, y=o.dy / 1000, z=lens_um / 1000 + o.dz / 1000)

    def set_target(self, r_mm: float, phi_deg: float):
        t = np.deg2rad(phi_deg)
        self.target_point.setData([r_mm * np.cos(t)], [r_mm * np.sin(t)])

    def latest_result(self, max_age_s: float = 0.3) -> Optional[EyeTrackerResults]:
        """Most recent result with a laser position, if no older than max_age_s."""
        r = self.latest
        if r is None or r.laser_position is None:
            return None
        if time.monotonic() - self._latest_monotonic > max_age_s:
            return None
        return r

    @QtCore.pyqtSlot(object, object, dict)
    def _on_frames(self, left, right, meta):
        self.img[0].setImage(left, autoLevels=True)
        self.img[1].setImage(right, autoLevels=True)
        if self.laser_focus_position is None:
            return  # lens position not read yet
        res = get_eye_tracker_results(left=left, right=right, meta=meta,
                                      laser_focus_position=self.laser_focus_position)
        self.latest = res
        self._latest_monotonic = time.monotonic()

        lp = res.laser_position
        dc = res.delta_laser_corner
        if lp is not None:
            self.laser_point.setData([float(lp[0])], [float(lp[1])])
            self.laser_point.setVisible(True)
        else:
            self.laser_point.setVisible(False)
        fmt = lambda v: f"{v:6.3f}" if v is not None else " " * 6
        self.pos_text.setText(
            f"x = {fmt(lp[0] if lp else None)}\ny = {fmt(lp[1] if lp else None)}\n"
            f"z = {fmt(lp[2] if lp else None)}\nΔc = {fmt(dc)}")
        if lp is not None and dc is not None:
            self.cornea_band.setRegion((dc, dc + 0.5))
            self.cornea_band.setVisible(True)
        else:
            self.cornea_band.setVisible(False)
        self.result_updated.emit(res)
