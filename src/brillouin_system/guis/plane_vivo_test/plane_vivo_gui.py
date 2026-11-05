"""
Reflection plane finder testing on a person (standalone; close the human interface first).

Shows every slew as it happens: S(z) with the threshold and background window,
plane against time, and detection counts. Attaches to the Zabers WITHOUT homing or
moving them, and the lens refuses to move until a limit is set.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from brillouin_system.eye_tracker.eye_position.coordinates import RigCoord
from brillouin_system.eye_tracker.eye_tracker_results import get_eye_tracker_results
from brillouin_system.guis.plane_accuracy_test.plane_accuracy_gui import _dspin, connect_devices
from brillouin_system.guis.plane_vivo_test.bounded_lens import BoundedLens
from brillouin_system.guis.plane_vivo_test.session import Session, Settings, Slew, xy_move_for_target
from brillouin_system.scan_managers.homodyne_plane_finder import PRE_ROLL_S, HomodyneFinderParams, zero_phase_s

DEFAULT_SAVE_DIR = Path.home() / "plane_vivo_runs"
EYE_MAX_AGE_S = 0.3
EYE_STALE_S = 30.0      # generous: the first frame comes after the tracker process has started
BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#c0392b", "#8a8a85"


class Worker(QObject):
    """Owns the devices; everything that blocks runs here."""

    status = pyqtSignal(str)
    positions = pyqtSignal(float, float, float, float)
    connected = pyqtSignal(str)
    done = pyqtSignal()
    slew = pyqtSignal(object)
    frame = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(self, use_dummy: bool, lens_port: str, stage_port: str, ni_device: str, run_dir: Path, subject: str):
        super().__init__()
        self._args = (use_dummy, lens_port, stage_port, ni_device)
        self.run_dir, self.subject = run_dir, subject
        self.dev = None
        self.session: Optional[Session] = None
        self.stop = threading.Event()
        self.eye_state: dict = {}    # replaced as a whole from the GUI thread; eye_t is its time.monotonic
        self.closed = threading.Event()

    def _eye_snapshot(self) -> dict:
        state = dict(self.eye_state)
        t = state.pop("eye_t", None)
        state["eye_age_s"] = None if t is None else time.monotonic() - t
        return state

    def _emit_positions(self):
        x, y, z = self.dev.stage.get_position()
        self.positions.emit(float(x), float(y), float(z), float(self.dev.lens.get_position()))

    @pyqtSlot()
    def connect_all(self):
        try:
            self.status.emit("Connecting (the camera may take a while to cool) ...")
            self.dev = connect_devices(*self._args)
            self.dev.lens = BoundedLens(self.dev.lens)
            self.dev.daq.configure(sample_rate_hz=5000.0)
            self.session = Session(self.dev, self.run_dir, self.subject, eye_state=self._eye_snapshot,
                                   on_status=self.status.emit, on_slew=self.slew.emit,
                                   on_frame=lambda f, z: self.frame.emit(f, z))
            self._emit_positions()
            self.connected.emit("")
        except Exception as e:
            self.connected.emit(f"{type(e).__name__}: {e}")

    def _guarded(self, fn):
        try:
            fn()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        try:
            self._emit_positions()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        self.done.emit()

    @pyqtSlot(float)
    def set_ceiling(self, um: float):
        self._guarded(lambda: self.dev.lens.set_ceiling(um))

    @pyqtSlot(float, float)
    def jog_stage(self, dx: float, dy: float):
        def go():
            self.session.plane_um = None    # another spot on the cornea: the old plane must not aim a round trip
            self.dev.stage.move_rel(dx=dx, dy=dy)
        self._guarded(go)

    @pyqtSlot(float)
    def jog_lens(self, dz: float):
        self._guarded(lambda: self.dev.lens.move_rel(dz))

    @pyqtSlot(float)
    def lens_to(self, z: float):
        self._guarded(lambda: self.dev.lens.move_abs(z))

    @pyqtSlot(bool)
    def set_shutter(self, is_open: bool):
        self._guarded(self.session.open_shutters if is_open else self.session.close_shutters)

    @pyqtSlot(object, float)
    def single_find(self, st: Settings, direction: float):
        self.stop.clear()
        self._guarded(lambda: self.session.single_find(st, direction, self.stop))

    @pyqtSlot(object)
    def round_trips(self, st: Settings):
        def go():
            self.stop.clear()
            c = self.session.round_trips(st, self.stop)
            self.status.emit(f"Round trips ended ({c['ended']}): {c['cycles']} cycles, {c['found']} found, "
                             f"{c['rejected']} rejected, {c['missed']} missed. Lens retracted, shutters closed.")
        self._guarded(go)

    @pyqtSlot()
    def shutdown(self):
        try:
            if self.dev is None:
                return
            try:
                self.dev.shutters.close_all()
            except Exception:
                pass
            for device in (self.dev.camera, self.dev.lens, self.dev.stage):
                try:
                    device.close()
                except Exception:
                    pass
        finally:
            self.closed.set()


class PlaneVivoWindow(QWidget):
    _req_connect = pyqtSignal()
    _req_ceiling = pyqtSignal(float)
    _req_stage = pyqtSignal(float, float)
    _req_lens = pyqtSignal(float)
    _req_lens_to = pyqtSignal(float)
    _req_shutter = pyqtSignal(bool)
    _req_single = pyqtSignal(object, float)
    _req_trips = pyqtSignal(object)
    _req_shutdown = pyqtSignal()
    _req_eye_shutdown = pyqtSignal()

    def __init__(self, *, use_dummy: bool, use_eye: bool = True, use_eye_dummy: Optional[bool] = None,
                 lens_port: str = "COM5", stage_port: str = "COM6", ni_device: str = "Dev1",
                 save_dir: Optional[Path] = None, subject: str = ""):
        super().__init__()
        self.setWindowTitle("Reflection plane finder, in vivo" + ("  [DUMMY]" if use_dummy else ""))
        self._lens: Optional[float] = None
        self._ceiling_set = False
        self._busy = True
        self._rows: list[dict] = []
        self._t0: Optional[float] = None
        self._eye = None
        self._eye_t = 0.0
        self._save_root = save_dir or DEFAULT_SAVE_DIR
        self._run_dir = self._save_root / f"vivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._build_ui(subject)

        self._laser_offset = self._load_offset()

        self._thread = QThread(self)
        self._worker = Worker(use_dummy, lens_port, stage_port, ni_device, self._run_dir, subject)
        self._worker.moveToThread(self._thread)
        self._worker.status.connect(self._log)
        self._worker.positions.connect(self._on_positions)
        self._worker.connected.connect(self._on_connected)
        self._worker.done.connect(lambda: self._set_busy(False))
        self._worker.slew.connect(self._on_slew)
        self._worker.frame.connect(self._on_frame)
        self._worker.failed.connect(self._on_failed)
        self._req_connect.connect(self._worker.connect_all)
        self._req_ceiling.connect(self._worker.set_ceiling)
        self._req_stage.connect(self._worker.jog_stage)
        self._req_lens.connect(self._worker.jog_lens)
        self._req_lens_to.connect(self._worker.lens_to)
        self._req_shutter.connect(self._worker.set_shutter)
        self._req_single.connect(self._worker.single_find)
        self._req_trips.connect(self._worker.round_trips)
        self._req_shutdown.connect(self._worker.shutdown)
        self._thread.start()
        self._set_busy(True)
        self._req_connect.emit()

        self._eye_thread = None
        if use_eye:
            self._start_eye(use_dummy if use_eye_dummy is None else use_eye_dummy)

    # ------------------------------------------------------------------ eye tracker
    @staticmethod
    def _load_offset():
        from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import (
            LaserOffset, load_laser_coord_system_from_toml,
        )
        try:
            return load_laser_coord_system_from_toml()
        except Exception:
            return LaserOffset(0, 0, 0)

    def _start_eye(self, use_dummy: bool):
        from brillouin_system.guis.human_interface.eye_tracker_controller import EyeTrackerController
        self._eye_thread = QThread(self)
        self._eye_ctrl = EyeTrackerController(use_dummy=use_dummy)
        self._eye_ctrl.moveToThread(self._eye_thread)
        self._eye_thread.started.connect(self._eye_ctrl.start)
        self._eye_ctrl.frames_ready.connect(self._on_eye_frames)
        self._req_eye_shutdown.connect(self._eye_ctrl.shutdown)
        self._eye_thread.start()
        self._eye_frame_t = time.monotonic()
        self._eye_watch = QTimer(self)
        self._eye_watch.timeout.connect(self._check_eye_frames)
        self._eye_watch.start(1000)

    def _check_eye_frames(self):
        """The tracker only logs a crash or a failed start; a frozen view must not pass for a working one."""
        if time.monotonic() - self._eye_frame_t > EYE_STALE_S:
            self._eye = None
            self.eye_label.setText("EYE TRACKER: NO FRAMES. See the console, then restart this program.")
            self.eye_label.setStyleSheet("font-family: monospace; color: #c0392b; font-weight: bold;")

    def _on_eye_frames(self, left, right, meta):
        self._eye_frame_t = time.monotonic()
        self.eye_label.setStyleSheet("font-family: monospace;")
        self.eye_left.setImage(left, autoLevels=True)
        self.eye_right.setImage(right, autoLevels=True)
        if self._lens is None:
            return
        o = self._laser_offset
        focus = RigCoord(x=o.dx / 1000, y=o.dy / 1000, z=self._lens / 1000 + o.dz / 1000)
        res = get_eye_tracker_results(left=left, right=right, meta=meta, laser_focus_position=focus)
        now = time.monotonic()
        if res.laser_position is None:
            self._eye = None
            self._worker.eye_state = dict(eye_x_mm=None, eye_y_mm=None, eye_dc_mm=None, eye_t=now)
            self.eye_label.setText("Pupil not found")
            return
        self._eye, self._eye_t = res, now
        x, y, _ = res.laser_position
        dc = res.delta_laser_corner
        self._worker.eye_state = dict(eye_x_mm=float(x), eye_y_mm=float(y), eye_dc_mm=None if dc is None else float(dc),
                                      eye_t=now)
        dc_text = "—" if dc is None else f"{dc:+.3f}"
        self.eye_label.setText(f"Laser at x {x:+.3f}, y {y:+.3f} mm from pupil center "
                               f"(R {np.hypot(x, y):.2f} mm)   Δc {dc_text} mm")

    def _move_xy(self):
        if self._eye is None or time.monotonic() - self._eye_t > EYE_MAX_AGE_S:
            QMessageBox.warning(self, "Move XY", "No recent eye tracker reading.")
            return
        x, y, _ = self._eye.laser_position
        dx, dy, clamped = xy_move_for_target(self.target_r.value(), self.target_phi.value(), (x, y))
        if clamped:
            self._log("Move XY clamped to 3.5 mm.")
        self._set_busy(True)
        self._req_stage.emit(dx, dy)

    # ------------------------------------------------------------------ UI
    def _build_ui(self, subject: str):
        root = QHBoxLayout(self)
        panel = QWidget()
        cl = QVBoxLayout(panel)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left = QVBoxLayout()
        root.addLayout(left)

        self.stop_btn = QPushButton("STOP: close shutters, retract")
        self.stop_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self._stop)
        left.addWidget(self.stop_btn)       # outside the scroll area: always in view
        left.addWidget(scroll, 1)

        g = QGroupBox("Lens safety (+z moves toward the eye)")
        f = QFormLayout(g)
        self.ceiling = _dspin(0.0, 60000.0, 0.0, 50.0, 1, " um")
        self.ceiling.setToolTip("No lens move or slew goes past this. Set it from your rig's clearance.")
        self.retract = _dspin(0.0, 60000.0, 0.0, 50.0, 1, " um")
        self.retract.setToolTip("Where the lens goes at STOP and after round trips.")
        self.ceiling_btn = QPushButton("Apply limit")
        self.ceiling_btn.clicked.connect(self._apply_ceiling)
        self.ceiling_label = QLabel("LIMIT NOT SET: the lens will not move")
        self.ceiling_label.setStyleSheet("color: #c0392b; font-weight: bold;")
        self.retract_here = QPushButton("Retract position = here")
        self.retract_here.clicked.connect(lambda: self._lens is not None and self.retract.setValue(self._lens))
        f.addRow("Lens limit:", self.ceiling)
        f.addRow(self.ceiling_btn)
        f.addRow(self.ceiling_label)
        f.addRow("Retract to:", self.retract)
        f.addRow(self.retract_here)
        cl.addWidget(g)

        g = QGroupBox("Position")
        f = QFormLayout(g)
        self.pos_label = QLabel("—")
        self.pos_label.setStyleSheet("font-family: monospace;")
        f.addRow(self.pos_label)
        self.stage_step = _dspin(1.0, 1000.0, 100.0, 50.0, 0, " um")
        f.addRow("Stage step:", self.stage_step)
        row = QHBoxLayout()
        for text, sx, sy in (("X-", -1, 0), ("X+", 1, 0), ("Y-", 0, -1), ("Y+", 0, 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _, a=sx, c=sy: self._jog_stage(a, c))
            row.addWidget(b)
        f.addRow(row)
        self.lens_step = _dspin(1.0, 500.0, 50.0, 10.0, 0, " um")
        f.addRow("Lens step:", self.lens_step)
        row = QHBoxLayout()
        for text, sign in (("Lens - (away)", -1), ("Lens + (toward eye)", 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _, s=sign: self._jog_lens(s))
            row.addWidget(b)
        f.addRow(row)
        self.target_r = _dspin(0.0, 5.0, 0.0, 0.25, 2, " mm")
        self.target_phi = _dspin(-360.0, 360.0, 0.0, 15.0, 0, " deg")
        f.addRow("Target R:", self.target_r)
        f.addRow("Target phi:", self.target_phi)
        self.xy_btn = QPushButton("Move XY (eye tracker)")
        self.xy_btn.clicked.connect(self._move_xy)
        f.addRow(self.xy_btn)
        row = QHBoxLayout()
        self.open_btn = QPushButton("Open sample shutters")
        self.open_btn.clicked.connect(lambda: self._shutter(True))
        self.close_btn = QPushButton("Close shutters")
        self.close_btn.clicked.connect(lambda: self._shutter(False))
        row.addWidget(self.open_btn)
        row.addWidget(self.close_btn)
        f.addRow(row)
        cl.addWidget(g)

        d = HomodyneFinderParams()
        g = QGroupBox("Finder")
        f = QFormLayout(g)
        self.speed = _dspin(50.0, 2000.0, d.speed_um_s, 50.0, 0, " um/s")
        self.speed.setToolTip("Only 400 um/s has been tested.")
        self.n_sigma = _dspin(1.0, 1000.0, d.n_sigma, 1.0, 0, " σ")
        self.stop_after = _dspin(30.0, 1000.0, d.stop_after_um, 10.0, 0, " um")
        self.search = _dspin(100.0, 3000.0, 1500.0, 100.0, 0, " um")
        self.search.setToolTip("Longest slew of a single find. Round trips use 2 x approach.")
        for label, w in (("Speed:", self.speed), ("Threshold:", self.n_sigma), ("Stop after peak:", self.stop_after),
                         ("Single find travel:", self.search)):
            f.addRow(label, w)
        row = QHBoxLayout()
        self.find_fwd = QPushButton("Find forward")
        self.find_fwd.clicked.connect(lambda: self._single(1.0))
        self.find_bwd = QPushButton("Find backward")
        self.find_bwd.clicked.connect(lambda: self._single(-1.0))
        row.addWidget(self.find_fwd)
        row.addWidget(self.find_bwd)
        f.addRow(row)
        cl.addWidget(g)

        g = QGroupBox("Round trips")
        f = QFormLayout(g)
        self.approach = _dspin(50.0, 1500.0, 300.0, 50.0, 0, " um")
        self.gate = _dspin(10.0, 2000.0, 200.0, 10.0, 0, " um")
        self.gate.setToolTip("A crossing further than this from the estimate is saved and marked rejected, "
                             "but does not update the estimate.")
        self.cycles = QSpinBox()
        self.cycles.setRange(1, 1000)
        self.cycles.setValue(10)
        self.max_misses = QSpinBox()
        self.max_misses.setRange(1, 100)
        self.max_misses.setValue(3)
        self.depth = _dspin(0.0, 1000.0, 0.0, 50.0, 0, " um")
        self.depth.setToolTip("Andor frame at plane + this lens travel in each cycle. 0 = no frames.")
        self.exposure = _dspin(0.001, 60.0, 0.2, 0.05, 3, " s")
        self.gain = QSpinBox()
        self.gain.setRange(0, 1000)
        for label, w in (("Approach:", self.approach), ("Plane gate:", self.gate), ("Round trips:", self.cycles),
                         ("End after misses:", self.max_misses), ("Andor frame depth:", self.depth),
                         ("Andor exposure:", self.exposure), ("Andor gain:", self.gain)):
            f.addRow(label, w)
        self.plane_label = QLabel("Plane estimate: none, run a single find first")
        f.addRow(self.plane_label)
        self.trips_btn = QPushButton("Run round trips")
        self.trips_btn.clicked.connect(self._trips)
        f.addRow(self.trips_btn)
        self.subject_edit = QLineEdit(subject)
        self.subject_edit.setPlaceholderText("Code, not a name")
        f.addRow("Subject code:", self.subject_edit)
        save_label = QLabel(f"Saving to {self._run_dir}")
        save_label.setWordWrap(True)
        f.addRow(save_label)
        cl.addWidget(g)
        cl.addStretch()
        scroll.setMinimumWidth(panel.sizeHint().width() + 30)

        right = QVBoxLayout()
        pg.setConfigOptions(useOpenGL=False, antialias=True, imageAxisOrder="row-major", background="w", foreground="k")
        self.glw = pg.GraphicsLayoutWidget()
        self.eye_left, self.eye_right, self.andor_img = pg.ImageItem(), pg.ImageItem(), pg.ImageItem()
        for item in (self.eye_left, self.eye_right, self.andor_img):
            vb = self.glw.addViewBox(lockAspect=True, enableMenu=False)
            vb.invertY(True)
            vb.addItem(item)
        right.addWidget(self.glw, 2)
        self.eye_label = QLabel("Eye tracker: no reading")
        self.eye_label.setStyleSheet("font-family: monospace;")
        right.addWidget(self.eye_label)
        self.stats_label = QLabel("No slews yet")
        self.stats_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        right.addWidget(self.stats_label)
        self.fig = Figure(figsize=(9, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax_s = self.fig.add_subplot(121)
        self.ax_z = self.fig.add_subplot(122)
        right.addWidget(self.canvas, 3)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        right.addWidget(self.log_box, 1)
        root.addLayout(right, 1)
        self._controls = [w for w in panel.findChildren(QPushButton) if w is not self.stop_btn]

    # ------------------------------------------------------------------ helpers
    def _log(self, text: str):
        self.log_box.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')}  {text}")

    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in self._controls:
            b.setEnabled(not busy)
        if not busy:
            session = self._worker.session
            has_plane = session is not None and session.plane_um is not None
            self.plane_label.setText(f"Plane estimate: {session.plane_um:.1f} um" if has_plane
                                     else "Plane estimate: none, run a single find first")
            self.trips_btn.setEnabled(has_plane and self._ceiling_set)
            for b in (self.find_fwd, self.find_bwd):
                b.setEnabled(self._ceiling_set)

    def _on_connected(self, err: str):
        if err:
            self._log(f"NOT CONNECTED: {err}")
            QMessageBox.critical(self, "Connection", err)
            return
        self._log("Connected. Nothing has moved. Set the lens limit before anything else.")
        self._set_busy(False)

    def _on_positions(self, x: float, y: float, z: float, lens: float):
        first = self._lens is None
        self._lens = lens
        self.pos_label.setText(f"Stage {x:.1f} / {y:.1f} / {z:.1f} um\nLens {lens:.1f} um")
        if first:
            self.retract.setValue(lens)

    def _on_failed(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.warning(self, "Error", msg)

    def _apply_ceiling(self):
        um = self.ceiling.value()
        if self._lens is not None and um <= self._lens:
            r = QMessageBox.question(self, "Lens limit", f"The limit {um:.1f} um is not above the lens ({self._lens:.1f} um). "
                                     "The lens could then only move away from the eye. Apply?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self._ceiling_set = True
        self.ceiling_label.setText(f"Limit {um:.1f} um")
        self.ceiling_label.setStyleSheet("")
        self._set_busy(True)
        self._req_ceiling.emit(um)

    def _jog_stage(self, sx: int, sy: int):
        self._set_busy(True)
        self._req_stage.emit(sx * self.stage_step.value(), sy * self.stage_step.value())

    def _jog_lens(self, sign: int):
        self._set_busy(True)
        self._req_lens.emit(sign * self.lens_step.value())

    def _shutter(self, is_open: bool):
        self._set_busy(True)
        self._req_shutter.emit(is_open)

    def _settings(self) -> Settings:
        return Settings(ceiling_um=self.ceiling.value(), retract_um=self.retract.value(), speed_um_s=self.speed.value(),
                        n_sigma=self.n_sigma.value(), stop_after_um=self.stop_after.value(),
                        search_um=self.search.value(), approach_um=self.approach.value(), gate_um=self.gate.value(),
                        n_cycles=self.cycles.value(), max_misses=self.max_misses.value(),
                        frame_depth_um=self.depth.value(), exposure_s=self.exposure.value(), gain=self.gain.value(),
                        target_r_mm=self.target_r.value(), target_phi_deg=self.target_phi.value())

    def _single(self, direction: float):
        self._worker.session.subject = self.subject_edit.text().strip()
        self._set_busy(True)
        self._req_single.emit(self._settings(), direction)

    def _trips(self):
        st = self._settings()
        try:
            st.validate()
        except ValueError as e:
            QMessageBox.warning(self, "Round trips", str(e))
            return
        self._worker.session.subject = self.subject_edit.text().strip()
        self._set_busy(True)
        self._req_trips.emit(st)

    def _stop(self):
        """Shutters first and directly, without waiting for the worker. The worker retracts when it ends its find."""
        self._worker.stop.set()
        dev = self._worker.dev
        if dev is not None and dev.shutters is not None:
            threading.Thread(target=dev.shutters.close_all, daemon=True).start()
        self._log("STOP: shutters closing.")
        if self._busy:
            return
        if self._ceiling_set:
            self._set_busy(True)
            self._req_lens_to.emit(self.retract.value())

    # ------------------------------------------------------------------ results
    def _on_frame(self, frame, z: float):
        f = np.asarray(frame, dtype=np.float32)
        self.andor_img.setImage(f, autoLevels=True)
        self._log(f"Andor frame at z = {z:.1f} um, sum above minimum {float(np.sum(f - f.min())):.0f}")

    def _on_slew(self, s: Slew):
        r = s.row
        self._rows.append(r)
        if self._t0 is None:
            self._t0 = r["t_perf"]
        snr = r.get("peak_snr")
        kind = "fwd" if r["direction"] > 0 else "bwd"
        text = f"{kind} {s.status}"
        if s.z_um is not None:
            text += f" z = {s.z_um:.1f} um"
        if snr is not None:
            text += f", peak SNR {snr:.0f} at {r['z_at_max_um']:.1f} um"
        if r.get("truncated"):
            text += ", shortened by the lens limit"
        self._log(text)
        self._plot_s(s)
        self._plot_z()
        self._update_stats()
        self.canvas.draw_idle()

    def _plot_s(self, s: Slew):
        a = self.ax_s
        a.clear()
        rec = s.rec
        z = None if rec is None else rec.z_um()
        if z is None or rec.n_samples < 300:
            a.set(title=f"Last slew: {s.status}, too short to show")
            return
        p = HomodyneFinderParams(speed_um_s=abs(rec.meta["speed_um_s"]))
        fs = rec.sample_rate_hz
        v = zero_phase_s(rec.values, fs, p)
        col = BLUE if s.row["direction"] > 0 else ORANGE
        a.plot(z, v, lw=0.6, color=col)
        i0 = int(round((PRE_ROLL_S + p.bg_start_s) * fs))
        i1 = min(i0 + int(round(p.bg_duration_s * fs)), z.size - 1)
        if i1 > i0:
            a.axvspan(z[i0], z[i1], color=GRAY, alpha=0.2, label="background")
        if s.row.get("threshold") is not None:
            a.axhline(s.row["threshold"], color=GRAY, lw=0.8, ls=":", label="threshold")
        if s.z_um is not None:
            a.axvline(s.z_um, color=RED, lw=0.9, ls="--", label="plane")
        a.set(title=f"Last slew: {s.status}", xlabel="lens z (um)", ylabel="S")
        a.legend(fontsize=7, frameon=False, loc="upper right")

    def _plot_z(self):
        a = self.ax_z
        a.clear()
        for sign, col, lab in ((1.0, BLUE, "forward"), (-1.0, ORANGE, "backward")):
            rows = [r for r in self._rows if r["direction"] == sign]
            for status, marker, face in (("found", "o", col), ("rejected", "o", "none")):
                pts = [(r["t_perf"] - self._t0, r["z_um"]) for r in rows if r["status"] == status]
                if pts:
                    a.plot(*zip(*pts), marker, ms=3.5, mfc=face, color=col, ls="none",
                           label=lab if status == "found" else f"{lab}, rejected")
            pts = [(r["t_perf"] - self._t0, r["plane_before_um"]) for r in rows
                   if r["status"] in ("missed", "no data") and r.get("plane_before_um") is not None]
            if pts:
                a.plot(*zip(*pts), "x", ms=5, color=col, ls="none", label=f"{lab}, missed")
        a.set(title="Planes this session", xlabel="time (s)", ylabel="lens z (um)")
        a.legend(fontsize=7, frameon=False, loc="best")

    def _update_stats(self):
        parts = []
        for sign, name in ((1.0, "fwd"), (-1.0, "bwd")):
            rows = [r for r in self._rows if r["direction"] == sign and r["status"] != "stopped"]
            found = sum(r["status"] == "found" for r in rows)
            rej = sum(r["status"] == "rejected" for r in rows)
            parts.append(f"{name} {found}/{len(rows)} found, {rej} rejected")
        cycles: dict = {}
        for r in self._rows:
            if r["status"] == "found" and r.get("cycle") is not None:
                cycles.setdefault((r["run"], r["cycle"]), {})[r["direction"]] = r["z_um"]
        d = np.array([c[1.0] - c[-1.0] for c in cycles.values() if len(c) == 2])
        if d.size:
            sd = f"{np.std(d, ddof=1) / np.sqrt(2):.1f}" if d.size > 1 else "—"
            parts.append(f"in − out {np.mean(d):+.1f} µm, short-term σ {sd} µm (n {d.size})")
        snr = [r["peak_snr"] for r in self._rows if r.get("peak_snr") is not None]
        if snr:
            parts.append(f"median peak SNR {np.median(snr):.0f}")
        lo = self._rows[-1].get("lo_v")
        if lo is not None:
            parts.append(f"mean PD {lo:.2f} V")
        self.stats_label.setText("   ".join(parts))

    def closeEvent(self, event):
        if self._busy and self._worker.dev is not None:
            QMessageBox.information(self, "Busy", "Press STOP and wait for it to finish first.")
            event.ignore()
            return
        self._req_shutdown.emit()
        self._worker.closed.wait(10.0)      # quit() could otherwise drop the queued shutdown
        self._thread.quit()
        self._thread.wait(5000)
        if self._eye_thread is not None:
            self._req_eye_shutdown.emit()
            time.sleep(5.0)                 # as the human interface: lets the eye tracker process finish
            self._eye_thread.quit()
            self._eye_thread.wait(10000)
        event.accept()
