"""
Plane precision + accuracy run on a phantom (standalone, in vitro only).

Set the center and the edge once, then the run alternates live plane finds and
Andor axial scans in randomized blocks, with the phantom untouched throughout.
Precision: spread of the finds. Accuracy: finds against the surface in the axial
scans taken around them. Attaches to the Zabers WITHOUT homing or moving them.
"""

from __future__ import annotations

import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from brillouin_system.guis.homodyne_test.experiment_plan import (
    StageRef, stage_ref_distance_mm, stage_ref_sphere_radius_mm,
)
from brillouin_system.guis.plane_accuracy_test.runner import (
    Cancelled, Devices, RunConfig, move_lens_from_below, open_sample_path, run_axial, run_experiment, run_find,
)
from brillouin_system.scan_managers.homodyne_plane_finder import HomodyneFinderParams
from brillouin_system.scan_managers.homodyne_processing import process_channels
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

DEFAULT_SAVE_DIR = Path.home() / "plane_accuracy_runs"
SATURATION_COUNTS = 60000.0     # the iXon digitizes to 16 bits
MAX_SUM_POINTS = 2000


def connect_devices(use_dummy: bool, lens_port: str, stage_port: str, ni_device: str) -> Devices:
    if use_dummy:
        from brillouin_system.devices.shutter_device import ShutterManagerDummy
        from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import \
            ZaberHumanInterfaceDummy
        from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
        from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
        from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
        lens = SimZaberLens(start_um=8000.0)
        stage = ZaberHumanInterfaceDummy(home_on_connect=False)
        stage.move_abs(12e3, 10e3, 12e3)
        daq = SimHomodyneDAQ(lens, plane_um=8000.0, peak_amplitude_v=0.2, noise_std_v=0.006)
        return Devices(daq, lens, stage, SimAndor(lens, plane_um=8000.0), ShutterManagerDummy("human_interface"))

    from brillouin_system.devices.cameras.andor.andor_frame.andor_config import andor_frame_config
    from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
    from brillouin_system.devices.ni.ni6008_multi import NI6008Multi
    from brillouin_system.devices.shutter_device import ShutterManager
    from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
    from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface
    lens = ZaberEyeLens(port=lens_port, home_on_connect=False)
    stage = ZaberHumanInterface(port=stage_port, home_on_connect=False)
    cfg = andor_frame_config.get()
    camera = IxonUltra(index=0, temperature="off", fan_mode="full", x_start=cfg.x_start, x_end=cfg.x_end,
                       y_start=cfg.y_start, y_end=cfg.y_end, vbin=cfg.vbin, hbin=cfg.hbin,
                       advanced_gain_option=cfg.advanced_gain_option, verbose=False)
    camera.set_from_config_file(cfg)    # same camera settings as the human interface; cools to its setpoint
    return Devices(NI6008Multi(device=ni_device), lens, stage, camera, ShutterManager("human_interface"))


class Worker(QObject):
    """Owns the devices; everything that blocks runs here."""

    status = pyqtSignal(str)
    positions = pyqtSignal(float, float, float, float)    # stage x, y, z, lens z [um]
    connected = pyqtSignal(str)                           # error text, empty when fine
    done = pyqtSignal()
    trace = pyqtSignal(object, object)                    # row or None, recording
    frame = pyqtSignal(object, float, bool)               # Andor frame, lens z [um], live view
    run_finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, use_dummy: bool, lens_port: str, stage_port: str, ni_device: str):
        super().__init__()
        self._args = (use_dummy, lens_port, stage_port, ni_device)
        self.dev: Optional[Devices] = None
        self.stop = threading.Event()

    def _emit_positions(self) -> None:
        x, y, z = self.dev.stage.get_position()
        self.positions.emit(float(x), float(y), float(z), float(self.dev.lens.get_position()))

    @pyqtSlot()
    def connect_all(self):
        try:
            self.status.emit("Connecting (the camera may take a while to cool) ...")
            self.dev = connect_devices(*self._args)
            self._emit_positions()
            self.connected.emit("")
        except Exception as e:
            self.connected.emit(f"{type(e).__name__}: {e}")

    def _guarded(self, fn):
        try:
            fn()
            self._emit_positions()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        self.done.emit()

    @pyqtSlot(float, float)
    def jog_stage(self, dx: float, dy: float):
        self._guarded(lambda: self.dev.stage.move_rel(dx=dx, dy=dy))

    @pyqtSlot(float)
    def jog_lens(self, dz: float):
        self._guarded(lambda: self.dev.lens.move_rel(dz))

    @pyqtSlot(bool)
    def set_shutter(self, is_open: bool):
        self._guarded(lambda: open_sample_path(self.dev.shutters) if is_open else self.dev.shutters.close_all())

    @pyqtSlot(float, int, bool)
    def snap(self, exposure_s: float, gain: int, live: bool):
        try:
            self.dev.camera.set_exposure_time(exposure_s)
            self.dev.camera.set_emccd_gain(gain)
            frame = np.asarray(self.dev.camera.snap())
            self.frame.emit(frame, float(self.dev.lens.get_position()), live)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.frame.emit(None, 0.0, False)

    @pyqtSlot(float, int, float, float, float)
    def scan_here(self, exposure_s: float, gain: int, before_um: float, after_um: float, step_um: float):
        """Andor frames stepping through the current z, then back to it. Saves nothing."""
        def go():
            self.stop.clear()
            z0 = float(self.dev.lens.get_position())
            self.dev.camera.set_exposure_time(exposure_s)
            self.dev.camera.set_emccd_gain(gain)
            try:
                run_axial(self.dev, before_um, after_um, step_um, z0, {}, self.stop,
                          lambda f, z: self.frame.emit(f, z, False))
            except Cancelled:
                pass
            move_lens_from_below(self.dev.lens, z0)
        self._guarded(go)

    @pyqtSlot(float)
    def move_lens_to(self, z_um: float):
        self._guarded(lambda: move_lens_from_below(self.dev.lens, z_um))

    @pyqtSlot(float, object, float)
    def find_manual(self, direction: float, params: HomodyneFinderParams, half_um: float):
        """Search around the current z and leave the lens on the plane (back at the start on a miss)."""
        def go():
            self.stop.clear()
            z0 = float(self.dev.lens.get_position())
            self.dev.daq.configure(sample_rate_hz=5000.0)
            rec = run_find(self.dev, params, z0, half_um, direction, {"kind": "manual"}, self.stop)
            found = rec is not None and rec.meta["finder_found"]
            self.dev.lens.move_abs(rec.meta["finder_z_um"] if found else z0)
            self.status.emit(f"Plane at z = {rec.meta['finder_z_um']:.1f} um, lens moved there." if found
                             else "Plane not found; lens back at the start.")
            if rec is not None:
                self.trace.emit(None, rec)
        self._guarded(go)

    @pyqtSlot(object, object)
    def run(self, cfg: RunConfig, run_dir: Path):
        self.stop.clear()
        try:
            counts = run_experiment(self.dev, cfg, run_dir, self.stop, on_status=self.status.emit,
                                    on_trial=lambda row, rec: self.trace.emit(row, rec),
                                    on_frame=lambda f, z: self.frame.emit(f, z, False))
            word = "stopped" if self.stop.is_set() else "finished"
            self.run_finished.emit(f"Run {word}: {counts['n_done']} trials, {counts['n_found']} found, "
                                   f"{counts['n_missed']} missed, {counts['n_axial']} axial scans. Folder: {run_dir}")
            self._emit_positions()
        except Exception as e:
            self.run_finished.emit(f"RUN ENDED BY AN ERROR ({type(e).__name__}: {e}). Saved so far: {run_dir}")

    @pyqtSlot()
    def shutdown(self):
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


def _dspin(lo, hi, val, step, decimals=1, suffix="") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
    w.setSuffix(suffix)
    return w


class PlaneAccuracyWindow(QWidget):
    _req_connect = pyqtSignal()
    _req_stage = pyqtSignal(float, float)
    _req_lens = pyqtSignal(float)
    _req_shutter = pyqtSignal(bool)
    _req_snap = pyqtSignal(float, int, bool)
    _req_scan = pyqtSignal(float, int, float, float, float)
    _req_lens_to = pyqtSignal(float)
    _req_find = pyqtSignal(float, object, float)
    _req_run = pyqtSignal(object, object)
    _req_shutdown = pyqtSignal()

    def __init__(self, *, use_dummy: bool, lens_port: str = "COM5", stage_port: str = "COM6", ni_device: str = "Dev1",
                 save_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Plane precision + accuracy, phantom only" + ("  [DUMMY]" if use_dummy else ""))
        self._xyz: Optional[tuple[float, float, float]] = None
        self._lens: Optional[float] = None
        self._center: Optional[StageRef] = None
        self._edge: Optional[StageRef] = None
        self._busy = True
        self._running = False
        self._build_ui(save_dir or DEFAULT_SAVE_DIR)

        self._thread = QThread(self)
        self._worker = Worker(use_dummy, lens_port, stage_port, ni_device)
        self._worker.moveToThread(self._thread)
        self._worker.status.connect(self._log)
        self._worker.positions.connect(self._on_positions)
        self._worker.connected.connect(self._on_connected)
        self._worker.done.connect(lambda: self._set_busy(False))
        self._worker.trace.connect(self._on_trace)
        self._worker.frame.connect(self._on_frame)
        self._worker.run_finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_failed)
        self._req_connect.connect(self._worker.connect_all)
        self._req_stage.connect(self._worker.jog_stage)
        self._req_lens.connect(self._worker.jog_lens)
        self._req_shutter.connect(self._worker.set_shutter)
        self._req_snap.connect(self._worker.snap)
        self._req_scan.connect(self._worker.scan_here)
        self._req_lens_to.connect(self._worker.move_lens_to)
        self._req_find.connect(self._worker.find_manual)
        self._req_run.connect(self._worker.run)
        self._req_shutdown.connect(self._worker.shutdown)
        self._thread.start()
        self._set_busy(True)
        self._req_connect.emit()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, save_dir: Path):
        root = QHBoxLayout(self)
        panel = QWidget()
        cl = QVBoxLayout(panel)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(520)
        root.addWidget(scroll)

        g = QGroupBox("Position")
        f = QFormLayout(g)
        self.pos_label = QLabel("—")
        self.pos_label.setStyleSheet("font-family: monospace;")
        f.addRow(self.pos_label)
        self.stage_step = _dspin(1.0, 2000.0, 100.0, 50.0, 0, " um")
        self.lens_step = _dspin(1.0, 2000.0, 50.0, 10.0, 0, " um")
        row = QHBoxLayout()
        for text, dx, dy in (("X-", -1, 0), ("X+", 1, 0), ("Y-", 0, -1), ("Y+", 0, 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _, a=dx, c=dy: self._jog_stage(a, c))
            row.addWidget(b)
        f.addRow("Stage step:", self.stage_step)
        f.addRow(row)
        row = QHBoxLayout()
        for text, sign in (("Lens -", -1), ("Lens +", 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _, s=sign: self._jog_lens(s))
            row.addWidget(b)
        f.addRow("Lens step:", self.lens_step)
        f.addRow(row)
        row = QHBoxLayout()
        self.open_btn = QPushButton("Open sample shutters")
        self.open_btn.clicked.connect(lambda: self._shutter(True))
        self.close_btn = QPushButton("Close all shutters")
        self.close_btn.clicked.connect(lambda: self._shutter(False))
        row.addWidget(self.open_btn)
        row.addWidget(self.close_btn)
        f.addRow(row)
        row = QHBoxLayout()
        for text, d in (("Find plane forward", 1.0), ("Find plane backward", -1.0)):
            b = QPushButton(text)
            b.clicked.connect(lambda _, x=d: self._find(x))
            row.addWidget(b)
        f.addRow(row)
        self.exposure = _dspin(0.001, 60.0, 0.2, 0.05, 3, " s")
        self.exposure.setToolTip("Used by Snap, Live (from the next frame), Andor scan and the run.")
        self.gain = QSpinBox()
        self.gain.setRange(0, 1000)
        self.exposure.valueChanged.connect(self._update_info)
        self.gain.valueChanged.connect(self._update_info)
        f.addRow("Andor exposure:", self.exposure)
        f.addRow("Andor gain:", self.gain)
        row = QHBoxLayout()
        self.snap_btn = QPushButton("Snap Andor")
        self.snap_btn.clicked.connect(self._snap)
        self.live_btn = QPushButton("Live Andor")
        self.live_btn.setCheckable(True)
        self.live_btn.toggled.connect(self._toggle_live)
        self.scan_btn = QPushButton("Andor scan through here")
        self.scan_btn.setToolTip("Frames from the axial before/after/step settings below, then back to this z.")
        self.scan_btn.clicked.connect(self._scan_here)
        row.addWidget(self.snap_btn)
        row.addWidget(self.live_btn)
        row.addWidget(self.scan_btn)
        f.addRow(row)
        row = QHBoxLayout()
        self.brightest_btn = QPushButton("Move lens to brightest")
        self.brightest_btn.clicked.connect(self._move_to_brightest)
        self.clear_sum_btn = QPushButton("Clear summed plot")
        self.clear_sum_btn.clicked.connect(self._clear_sum)
        row.addWidget(self.brightest_btn)
        row.addWidget(self.clear_sum_btn)
        f.addRow(row)
        cl.addWidget(g)

        g = QGroupBox("Reference positions")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.center_btn = QPushButton("Set center")
        self.center_btn.clicked.connect(lambda: self._set_ref("center"))
        self.edge_btn = QPushButton("Set edge")
        self.edge_btn.clicked.connect(lambda: self._set_ref("edge"))
        row.addWidget(self.center_btn)
        row.addWidget(self.edge_btn)
        f.addRow(row)
        self.center_label = QLabel("Center: —")
        self.edge_label = QLabel("Edge: —")
        self.geom_label = QLabel("")
        for w in (self.center_label, self.edge_label, self.geom_label):
            w.setStyleSheet("font-family: monospace;")
            f.addRow(w)
        cl.addWidget(g)

        d = HomodyneFinderParams()
        g = QGroupBox("Run")
        f = QFormLayout(g)
        self.r_step = _dspin(0.01, 10.0, 0.25, 0.05, 2, " mm")
        self.blocks = QSpinBox()
        self.blocks.setRange(1, 200)
        self.blocks.setValue(5)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(random.randrange(1000000))
        self.half = _dspin(10.0, 1500.0, 400.0, 50.0, 0, " um")
        self.speed = _dspin(50.0, 2000.0, d.speed_um_s, 50.0, 0, " um/s")
        self.n_sigma = _dspin(1.0, 1000.0, d.n_sigma, 1.0, 0, " σ")
        self.stop_after = _dspin(30.0, 1000.0, d.stop_after_um, 10.0, 0, " um")
        self.ax_before = _dspin(10.0, 1000.0, 150.0, 10.0, 0, " um")
        self.ax_after = _dspin(10.0, 1000.0, 200.0, 10.0, 0, " um")
        self.ax_step = _dspin(1.0, 200.0, 10.0, 5.0, 1, " um")
        self.close_end = QCheckBox("Close shutters at the end")
        self.close_end.setChecked(True)
        self.save_dir = QLineEdit(str(save_dir))
        self.info = QLabel()
        self.info.setWordWrap(True)
        for label, w in (("R step:", self.r_step), ("Blocks:", self.blocks), ("Order seed:", self.seed),
                         ("Find from plane ±:", self.half), ("Finder speed:", self.speed),
                         ("Finder threshold:", self.n_sigma), ("Finder stop after:", self.stop_after),
                         ("Axial before plane:", self.ax_before), ("Axial after plane:", self.ax_after),
                         ("Axial step:", self.ax_step)):
            f.addRow(label, w)
            w.valueChanged.connect(self._update_info)
        f.addRow(self.close_end)
        f.addRow("Save to:", self.save_dir)
        f.addRow(self.info)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start run")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        f.addRow(row)
        cl.addWidget(g)
        cl.addStretch()

        right = QVBoxLayout()
        right.addWidget(self._build_camera_view(), 5)
        self.sum_label = QLabel("Summed intensity: —")
        self.sum_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        right.addWidget(self.sum_label)
        row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._show_axial_frame)
        self.frame_slider_label = QLabel("Last axial scan: none yet")
        row.addWidget(self.frame_slider_label)
        row.addWidget(self.frame_slider, 1)
        right.addLayout(row)
        self.fig = Figure(figsize=(9, 2.6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax_s = self.fig.add_subplot(121)
        self.ax_err = self.fig.add_subplot(122)
        right.addWidget(self.canvas, 2)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        right.addWidget(self.log_box, 1)
        root.addLayout(right, 1)
        self._controls = [w for w in panel.findChildren(QPushButton) if w is not self.stop_btn]
        self._err = {"find_fwd": ([], []), "find_bwd": ([], []), "axial": ([], [])}
        self._last_axial = None
        self._fitter = SpectrumFitter()     # the human interface's spectrum: same row band and pixel offsets
        self._sum_z: list[float] = []
        self._sum_v: list[float] = []
        self._snap_pending = False
        self._current_sum: Optional[tuple[float, float]] = None
        self._update_info()

    def _build_camera_view(self) -> pg.GraphicsLayoutWidget:
        """Image and spectrum as in the human interface, plus summed intensity against lens z."""
        pg.setConfigOptions(useOpenGL=False, antialias=True, imageAxisOrder="row-major", background="w", foreground="k")
        self.glw = pg.GraphicsLayoutWidget()
        ci = self.glw.ci
        ci.setSpacing(12)
        ci.setContentsMargins(8, 8, 8, 8)

        self.vb_img = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.vb_img.invertY(True)
        self.img_item = pg.ImageItem(autoDownsample=True)
        self.vb_img.addItem(self.img_item)
        self.vb_img.setBorder((170, 170, 170))

        self.glw.nextRow()
        self.spec_plot = self.glw.addPlot(labels={"left": "Intensity", "bottom": "Pixel (X)"})
        self.spec_plot.setMenuEnabled(False)
        self.spec_plot.setTitle("Spectrum", color=(40, 40, 40))
        self.spec_plot.getViewBox().setBorder((170, 170, 170))
        self.spec_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4, symbolPen=pg.mkPen("k"), symbolBrush="k")
        self.spec_plot.addItem(self.spec_points)

        self.glw.nextRow()
        self.sum_plot = self.glw.addPlot(labels={"left": "Summed intensity", "bottom": "Lens z (um)"})
        self.sum_plot.setMenuEnabled(False)
        self.sum_plot.setTitle("Summed intensity vs lens z", color=(40, 40, 40))
        self.sum_plot.getViewBox().setBorder((170, 170, 170))
        self.sum_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5, symbolPen=pg.mkPen("k"), symbolBrush="k")
        self.sum_best = pg.PlotDataItem(pen=None, symbol="o", symbolSize=10, symbolPen=pg.mkPen("r"), symbolBrush="r")
        self.sum_plot.addItem(self.sum_points)
        self.sum_plot.addItem(self.sum_best)
        return self.glw

    # ------------------------------------------------------------------ helpers
    def _log(self, text: str):
        self.log_box.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')}  {text}")

    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in self._controls:
            b.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        if not busy:
            self.start_btn.setEnabled(self._center is not None and self._edge is not None and not self._no_devices())

    def _no_devices(self) -> bool:
        return self._worker.dev is None

    def _on_connected(self, err: str):
        if err:
            self._log(f"NOT CONNECTED: {err}")
            QMessageBox.critical(self, "Connection", err)
            self.stop_btn.setEnabled(False)
            return
        self._log("Connected. Nothing has moved.")
        self._set_busy(False)

    def _on_positions(self, x: float, y: float, z: float, lens: float):
        self._xyz, self._lens = (x, y, z), lens
        self.pos_label.setText(f"Stage {x:.1f} / {y:.1f} / {z:.1f} um\nLens {lens:.1f} um")

    def _on_failed(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.warning(self, "Error", msg)

    def _jog_stage(self, sx: int, sy: int):
        self._clear_sum()       # another spot on the surface
        self._set_busy(True)
        s = self.stage_step.value()
        self._req_stage.emit(sx * s, sy * s)

    def _jog_lens(self, sign: int):
        self._set_busy(True)
        self._req_lens.emit(sign * self.lens_step.value())

    def _shutter(self, is_open: bool):
        self._set_busy(True)
        self._req_shutter.emit(is_open)

    def _find_params(self) -> HomodyneFinderParams:
        half = self.half.value()
        stop = self.stop_after.value()
        return HomodyneFinderParams(speed_um_s=self.speed.value(), n_sigma=self.n_sigma.value(), stop_after_um=stop,
                                    max_distance_um=max(2.0 * half, half + stop + 50.0))

    def _snap(self):
        if not self._snap_pending:
            self._snap_pending = True
            self._req_snap.emit(self.exposure.value(), self.gain.value(), False)

    def _toggle_live(self, on: bool):
        if on and not self._snap_pending:
            self._snap_pending = True
            self._req_snap.emit(self.exposure.value(), self.gain.value(), True)

    def _on_frame(self, frame, z: float, live: bool):
        if live or not self._running:
            self._snap_pending = False
        if frame is None:
            self.live_btn.setChecked(False)
            return
        if self._running and self._sum_z and z < self._sum_z[-1]:
            self._clear_sum()       # a new axial scan started
        total = self._display_frame(frame, z)
        self._add_sum(z, total)
        if live and self.live_btn.isChecked() and not self._running:
            self._snap_pending = True
            self._req_snap.emit(self.exposure.value(), self.gain.value(), True)

    def _display_frame(self, frame: np.ndarray, z: float) -> float:
        """Min-max per frame, as in the human interface. Returns the summed intensity."""
        frame = np.ascontiguousarray(frame).astype(np.float32, copy=False)
        fmin, fmax = float(np.nanmin(frame)), float(np.nanmax(frame))
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            fmin, fmax = 0.0, 1.0
        self.img_item.setImage(frame, autoLevels=False, levels=(fmin, fmax))
        px, sline = self._fitter.get_px_sline_from_image(frame)
        self.spec_points.setData(px, sline)
        total = self._fitter.get_total_sline_value(sline)
        flag = "   NEAR SATURATION" if fmax >= SATURATION_COUNTS else ""
        self.spec_plot.setTitle(f"Spectrum at lens z = {z:.1f} um, frame max {fmax:.0f} counts{flag}",
                                color=(192, 57, 43) if flag else (40, 40, 40))
        self._current_sum = (z, total)
        self._update_sum_label()
        return total

    def _add_sum(self, z: float, total: float):
        self._sum_z.append(z)
        self._sum_v.append(total)
        del self._sum_z[:-MAX_SUM_POINTS], self._sum_v[:-MAX_SUM_POINTS]
        self._update_sum_plot()

    def _update_sum_plot(self):
        self.sum_points.setData(self._sum_z, self._sum_v)
        if self._sum_v:
            i = int(np.argmax(self._sum_v))
            self.sum_best.setData([self._sum_z[i]], [self._sum_v[i]])
        else:
            self.sum_best.setData([], [])
        self._update_sum_label()

    def _update_sum_label(self):
        z, total = self._current_sum if self._current_sum is not None else (None, None)
        text = "Summed intensity: —" if z is None else f"Summed intensity {total:.0f} at z = {z:.1f} um"
        if self._sum_v:
            i = int(np.argmax(self._sum_v))
            text += f"     brightest {self._sum_v[i]:.0f} at z = {self._sum_z[i]:.1f} um"
        self.sum_label.setText(text)

    def _clear_sum(self):
        self._sum_z, self._sum_v = [], []
        self._fitter = SpectrumFitter()     # its automatic row band may have frozen on a dark frame
        self._update_sum_plot()

    def _scan_here(self):
        self.live_btn.setChecked(False)
        self._clear_sum()
        self._set_busy(True)
        self._req_scan.emit(self.exposure.value(), self.gain.value(), self.ax_before.value(), self.ax_after.value(),
                            self.ax_step.value())

    def _move_to_brightest(self):
        if self._sum_v:
            self._set_busy(True)
            self._req_lens_to.emit(self._sum_z[int(np.argmax(self._sum_v))])

    def _show_axial_frame(self, i: int):
        rec = self._last_axial
        if rec is None or not 0 <= i < rec.frames.shape[0]:
            return
        self.frame_slider_label.setText(f"Last axial scan, frame {i + 1}/{rec.frames.shape[0]}, z = {rec.z_um[i]:.1f} um")
        self._display_frame(rec.frames[i], float(rec.z_um[i]))

    def _find(self, direction: float):
        self.live_btn.setChecked(False)
        self._set_busy(True)
        self._req_find.emit(direction, self._find_params(), self.half.value())

    def _set_ref(self, which: str):
        if self._xyz is None or self._lens is None:
            return
        ref = StageRef(*self._xyz, lens_um=self._lens)
        text = (f"{which.capitalize()}: stage {ref.stage_x_um:.1f} / {ref.stage_y_um:.1f} / {ref.stage_z_um:.1f} um, "
                f"lens {ref.lens_um:.1f} um")
        if which == "center":
            self._center = ref
            self.center_label.setText(text)
        else:
            self._edge = ref
            self.edge_label.setText(text)
        self._update_info()
        self._set_busy(False)

    def _config(self) -> RunConfig:
        return RunConfig(
            center=self._center, edge=self._edge, r_step_mm=self.r_step.value(), n_blocks=self.blocks.value(),
            seed=self.seed.value(), half_window_um=self.half.value(), speed_um_s=self.speed.value(),
            n_sigma=self.n_sigma.value(), stop_after_um=self.stop_after.value(),
            axial_before_um=self.ax_before.value(), axial_after_um=self.ax_after.value(),
            axial_step_um=self.ax_step.value(), exposure_s=self.exposure.value(), gain=self.gain.value(),
            close_shutters_at_end=self.close_end.isChecked())

    def _update_info(self, *_):
        c, e = self._center, self._edge
        if c is None or e is None:
            self.geom_label.setText("")
            self.info.setText("Put the lens on the plane at the apex: Set center. Then at the outermost point: Set edge.")
            return
        try:
            cfg = self._config()
            r = stage_ref_distance_mm(c, e)
            R = stage_ref_sphere_radius_mm(c, e)
            lines = [f"edge R {r:.3f} mm, sag {e.lens_um - c.lens_um:+.1f} um",
                     f"sphere R {R:.2f} mm" if R is not None else "flat (sag < 1 um)"]
            if R is not None and r <= R:
                lines.append(f"edge tilt {np.degrees(np.arcsin(r / R)):.1f} deg")
            if abs(e.stage_z_um - c.stage_z_um) > 1.0:
                lines.append(f"WARNING stage Z differs by {e.stage_z_um - c.stage_z_um:+.1f} um (only X/Y move)")
            if e.lens_um < c.lens_um - 1.0:
                lines.append("WARNING the edge plane is at a smaller lens z than the center: check center and edge")
            self.geom_label.setText("\n".join(lines))
            n_r = len(cfg.radii())
            self.info.setText(f"{len(cfg.trials())} trials: {n_r} radii x 3 kinds (find fwd, find bwd, axial) x "
                              f"{cfg.n_blocks} blocks, random order. Roughly {cfg.estimated_s() / 60:.0f} min.")
        except ValueError as ex:
            self.geom_label.setText("")
            self.info.setText(str(ex))

    def _start(self):
        try:
            cfg = self._config()
            cfg.validate()
            cfg.trials()
        except (ValueError, TypeError) as ex:
            QMessageBox.warning(self, "Run", str(ex))
            return
        run_dir = Path(self.save_dir.text()) / f"plane_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        r = QMessageBox.question(
            self, "Start run",
            f"PHANTOM ONLY: this run is not for use on a person.\n\n{self.info.text()}\n\n"
            f"The rig stage (X/Y only) and the eye lens will move and the sample shutters will open.\n"
            f"Saving to\n{run_dir}\n\nStart?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        self._err = {k: ([], []) for k in self._err}
        self.live_btn.setChecked(False)
        self._running = True
        self._set_busy(True)
        self._req_run.emit(cfg, run_dir)

    def _stop(self):
        self._worker.stop.set()
        self._log("Stop requested.")

    def _on_run_finished(self, text: str):
        self._running = False
        self._log(text)
        self._set_busy(False)

    # ------------------------------------------------------------------ plots
    def _on_trace(self, row, rec):
        kind = "manual" if row is None else row["kind"]
        if kind == "axial":
            self._plot_axial(row, rec)
        else:
            self._plot_find(row, rec)
        if row is not None:
            z = row.get("finder_z_um") if kind != "axial" else row.get("axial_peak_z_um")
            if z is not None:
                xs, ys = self._err[kind]
                xs.append(row["r_mm"])
                ys.append(z - row["lens_pred_um"])
            self._plot_errors()
        self.canvas.draw_idle()

    def _plot_find(self, row, rec):
        a = self.ax_s
        a.clear()
        v = np.atleast_2d(rec.values)
        p = HomodyneFinderParams()
        if v.shape[1] > 300 and rec.z_um() is not None:
            s = process_channels(v, rec.sample_rate_hz, p.band_hz).s
            a.plot(rec.z_um(), s, lw=0.6, color="#2a78d6")
            if rec.meta.get("finder_found"):
                a.axvline(rec.meta["finder_z_um"], color="#c0392b", lw=0.8, ls="--")
        a.set(xlabel="lens z (um)", ylabel="S", title=f"Last find: {rec.meta.get('kind')}")

    def _plot_axial(self, row, rec):
        self._last_axial = rec
        self.frame_slider.setEnabled(True)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, rec.frames.shape[0] - 1)
        self.frame_slider.setValue(int(np.argmax(self._sum_v)) if len(self._sum_v) == rec.frames.shape[0] else 0)
        self.frame_slider.blockSignals(False)
        self._show_axial_frame(self.frame_slider.value())

    def _plot_errors(self):
        a = self.ax_err
        a.clear()
        for kind, col, label in (("find_fwd", "#2a78d6", "find fwd"), ("find_bwd", "#eb6834", "find bwd"),
                                 ("axial", "#1f1f1e", "axial brightest frame")):
            xs, ys = self._err[kind]
            if xs:
                a.plot(xs, ys, "o", ms=3, color=col, label=label)
        a.axhline(0, color="#8a8a85", lw=0.6)
        a.set(xlabel="R (mm)", ylabel="z - predicted (um)", title="Run so far")
        a.legend(fontsize=7, loc="best", frameon=False)

    def closeEvent(self, event):
        if self._running:
            QMessageBox.information(self, "Busy", "Stop the run first.")
            event.ignore()
            return
        self.live_btn.setChecked(False)
        self._req_shutdown.emit()
        self._thread.quit()
        self._thread.wait(5000)
        event.accept()
