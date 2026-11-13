"""
Lens-Z vs stage-Z carrier test (standalone; borrows the DAQ, Zaber, Andor and shutter modules).

Getting onto the plane: Live Andor while jogging the lens (image, the human interface's spectrum, and summed
intensity against lens z), or "Andor sweep here" then "Move lens to brightest", or the homodyne "Park lens on
plane". Then Run protocol: Andor fine sweeps along lens and stage Z, blocks of a stationary record plus lens
and stage-Z slews in random order, then the Andor sweeps again. Everything is centred on where the lens and
stage were when Run was pressed, and both go back there after every step.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from brillouin_system.guis.plane_accuracy_test.runner import open_sample_path
from brillouin_system.guis.z_carrier_test.motion import SimAxisMover, ZaberAxisMover
from brillouin_system.guis.z_carrier_test.protocol import PRE_ROLL_S, PRELOAD_UM, ProtocolConfig
from brillouin_system.guis.z_carrier_test.runner import (
    Cancelled, Rig, andor_sweep, approach, park_lens_on_plane, run_protocol,
)
from brillouin_system.scan_managers.homodyne_plane_finder import HomodyneFinderParams, zero_phase_s
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

PARK_HALF_UM = 150.0
PARK_SPEED_UM_S = 400.0
SATURATION_COUNTS = 60000.0     # the iXon digitizes to 16 bits
MAX_SUM_POINTS = 2000
MAX_TO_BRIGHTEST_UM = 200.0


def connect_rig(use_dummy: bool, lens_port: str, stage_port: str, ni_device: str) -> Rig:
    if use_dummy:
        from brillouin_system.devices.shutter_device import ShutterManagerDummy
        from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
        from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
        from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
        lens, stage_z = SimZaberLens(start_um=8030.0), SimZaberLens(start_um=12000.0, move_speed_um_s=3000.0)
        daq = SimHomodyneDAQ(lens, plane_um=8000.0, peak_amplitude_v=0.2, noise_std_v=0.006)
        return Rig(daq, SimAxisMover("lens", lens), SimAxisMover("stage", stage_z), SimAndor(lens, plane_um=8000.0),
                   ShutterManagerDummy("human_interface"), jog_lens=lens.move_rel, jog_stage=stage_z.move_rel)

    from brillouin_system.devices.cameras.andor.andor_frame.andor_config import andor_frame_config
    from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
    from brillouin_system.devices.ni.ni6008_multi import NI6008Multi
    from brillouin_system.devices.shutter_device import ShutterManager
    from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
    from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface
    lens = ZaberEyeLens(port=lens_port, home_on_connect=False)        # attach without homing or moving
    stage = ZaberHumanInterface(port=stage_port, home_on_connect=False)
    cfg = andor_frame_config.get()
    camera = IxonUltra(index=0, temperature="off", fan_mode="full", x_start=cfg.x_start, x_end=cfg.x_end,
                       y_start=cfg.y_start, y_end=cfg.y_end, vbin=cfg.vbin, hbin=cfg.hbin,
                       advanced_gain_option=cfg.advanced_gain_option, verbose=False)
    camera.set_from_config_file(cfg)    # same camera settings as the human interface
    return Rig(NI6008Multi(device=ni_device), ZaberAxisMover("lens", lens.axis), ZaberAxisMover("stage", stage.z_axis),
               camera, ShutterManager("human_interface"), jog_lens=lens.move_rel,
               jog_stage=lambda dz: stage.move_rel(dz=dz), devices=(camera, lens, stage))


class Worker(QObject):
    status = pyqtSignal(str)
    positions = pyqtSignal(float, float)        # lens z, stage z [um]
    connected = pyqtSignal(str)                 # error text, empty when fine
    result = pyqtSignal(object, object)         # log row (or None), recording
    frame = pyqtSignal(object, str, float, bool)    # Andor frame (None on error), axis, its z [um], live view
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, args: tuple):
        super().__init__()
        self._args = args
        self.rig: Optional[Rig] = None
        self.stop = threading.Event()

    def _emit_positions(self):
        self.positions.emit(self.rig.lens.get_position(), self.rig.stage.get_position())

    def _guarded(self, fn):
        try:
            fn()
            self._emit_positions()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        self.done.emit()

    @pyqtSlot()
    def connect_all(self):
        try:
            self.status.emit("Connecting (the camera may take a while) ...")
            self.rig = connect_rig(*self._args)
            self._emit_positions()
            self.connected.emit("")
        except Exception as e:
            self.connected.emit(f"{type(e).__name__}: {e}")

    @pyqtSlot(str, float)
    def jog(self, axis: str, dz: float):
        self._guarded(lambda: (self.rig.jog_lens if axis == "lens" else self.rig.jog_stage)(dz))

    @pyqtSlot(bool)
    def shutter(self, is_open: bool):
        self._guarded(lambda: open_sample_path(self.rig.shutters) if is_open else self.rig.shutters.close_all())

    @pyqtSlot(float, int, bool)
    def snap(self, exposure_s: float, gain: int, live: bool):
        try:
            self.rig.camera.set_exposure_time(exposure_s)
            self.rig.camera.set_emccd_gain(gain)
            f = np.asarray(self.rig.camera.snap())
            z = self.rig.lens.get_position()
            self.positions.emit(z, self.rig.stage.get_position())
            self.frame.emit(f, "lens", z, live)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.frame.emit(None, "lens", 0.0, False)

    @pyqtSlot(float, int, float, float)
    def sweep_here(self, exposure_s: float, gain: int, half_um: float, step_um: float):
        """Andor frames stepping the lens through its current z, then back there. Saves nothing."""
        def go():
            self.stop.clear()
            lens = self.rig.lens
            z0 = lens.get_position()
            lens.set_window(z0 - half_um - PRELOAD_UM - 5.0, z0 + half_um + 5.0)
            self.rig.camera.set_exposure_time(exposure_s)
            self.rig.camera.set_emccd_gain(gain)
            try:
                andor_sweep(self.rig, lens, z0, half_um, step_um, {"axis": "lens"}, self.stop,
                            lambda f, z: self.frame.emit(f, "lens", z, False))
            except Cancelled:
                self.status.emit("Sweep stopped; lens back at the start.")
        self._guarded(go)

    @pyqtSlot(float)
    def lens_to(self, z_um: float):
        """Lens to z_um from below; refused if that is more than MAX_TO_BRIGHTEST_UM from where it is."""
        def go():
            lens = self.rig.lens
            z0 = lens.get_position()
            if abs(z_um - z0) > MAX_TO_BRIGHTEST_UM:
                raise ValueError(f"Brightest frame is {z_um - z0:+.0f} um away (limit {MAX_TO_BRIGHTEST_UM:.0f} um); "
                                 f"clear the summed plot and sweep again.")
            lens.set_window(min(z0, z_um) - PRELOAD_UM - 5.0, max(z0, z_um) + 5.0)
            approach(lens, z_um)
        self._guarded(go)

    @pyqtSlot()
    def park(self):
        def go():
            self.stop.clear()
            self.rig.daq.configure(sample_rate_hz=5000.0)
            z, rec = park_lens_on_plane(self.rig, PARK_HALF_UM, PARK_SPEED_UM_S, self.stop)
            self.status.emit(f"Lens parked on the homodyne plane at {z:.1f} um." if z is not None
                             else "No clear plane within the slew; lens left where it was.")
            self.result.emit(None, rec)
        self._guarded(go)

    @pyqtSlot(object, object)
    def run(self, cfg: ProtocolConfig, run_dir: Path):
        def go():
            self.stop.clear()
            axis = {"now": "lens"}

            def on_result(row, rec):
                self.result.emit(row, rec)

            def on_frame(f, z):
                self.frame.emit(f, axis["now"], z, False)

            def on_status(text):
                if "andor stage" in text:
                    axis["now"] = "stage"
                elif "andor lens" in text:
                    axis["now"] = "lens"
                self.status.emit(text)

            c = run_protocol(self.rig, cfg, run_dir, self.stop, on_status=on_status, on_result=on_result,
                             on_frame=on_frame)
            word = "stopped" if self.stop.is_set() else "finished"
            self.status.emit(f"Run {word}: {c['n_done']} steps ({c['n_slew']} slews, {c['n_stationary']} stationary, "
                             f"{c['n_andor']} Andor sweeps). Folder: {run_dir}")
        self._guarded(go)

    @pyqtSlot()
    def shutdown(self):
        if self.rig is None:
            return
        try:
            self.rig.shutters.close_all()
        except Exception:
            pass
        for d in self.rig.devices:
            try:
                d.close()
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


class ZCarrierWindow(QWidget):
    _req_connect = pyqtSignal()
    _req_jog = pyqtSignal(str, float)
    _req_shutter = pyqtSignal(bool)
    _req_snap = pyqtSignal(float, int, bool)
    _req_sweep = pyqtSignal(float, int, float, float)
    _req_lens_to = pyqtSignal(float)
    _req_park = pyqtSignal()
    _req_run = pyqtSignal(object, object)
    _req_shutdown = pyqtSignal()

    def __init__(self, *, use_dummy: bool, lens_port: str = "COM5", stage_port: str = "COM6", ni_device: str = "Dev1",
                 save_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Lens Z vs stage Z carrier test" + ("  [DUMMY]" if use_dummy else ""))
        self._busy = True
        self._running = False
        self._snap_pending = False
        self._fitter = SpectrumFitter()     # the human interface's spectrum: same row band and pixel offsets
        self._sum_axis = "lens"
        self._sum_z: list[float] = []
        self._sum_v: list[float] = []
        self._current_sum: Optional[tuple[float, float]] = None
        self._build_ui(save_dir or Path.home() / "Documents" / "z_carrier_runs")

        self._thread = QThread(self)
        self._worker = Worker((use_dummy, lens_port, stage_port, ni_device))
        self._worker.moveToThread(self._thread)
        self._worker.status.connect(self._log)
        self._worker.positions.connect(self._on_positions)
        self._worker.connected.connect(self._on_connected)
        self._worker.result.connect(self._on_result)
        self._worker.frame.connect(self._on_frame)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._req_connect.connect(self._worker.connect_all)
        self._req_jog.connect(self._worker.jog)
        self._req_shutter.connect(self._worker.shutter)
        self._req_snap.connect(self._worker.snap)
        self._req_sweep.connect(self._worker.sweep_here)
        self._req_lens_to.connect(self._worker.lens_to)
        self._req_park.connect(self._worker.park)
        self._req_run.connect(self._worker.run)
        self._req_shutdown.connect(self._worker.shutdown)
        self._thread.start()
        self._set_busy(True)
        self._req_connect.emit()

    # ------------------------------------------------------------------ UI

    def _build_ui(self, save_dir: Path):
        root = QHBoxLayout(self)
        panel = QWidget()
        left = QVBoxLayout(panel)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)
        root.addWidget(scroll, 0)

        pos = QGroupBox("Position")
        f = QFormLayout(pos)
        self.lens_label, self.stage_label = QLabel("-"), QLabel("-")
        f.addRow("Lens z", self.lens_label)
        f.addRow("Stage z", self.stage_label)
        self.jog_step = _dspin(0.5, 500, 10, 5, 1, " um")
        f.addRow("Jog step", self.jog_step)
        for axis in ("lens", "stage"):
            row = QHBoxLayout()
            for sign, text in ((-1, "-"), (1, "+")):
                b = QPushButton(f"{axis.capitalize()} {text}")
                b.clicked.connect(lambda _, a=axis, s=sign: self._jog(a, s))
                row.addWidget(b)
            f.addRow(row)
        row = QHBoxLayout()
        self.b_open, self.b_close = QPushButton("Open sample shutters"), QPushButton("Close all shutters")
        self.b_open.clicked.connect(lambda: self._shutter(True))
        self.b_close.clicked.connect(lambda: self._shutter(False))
        row.addWidget(self.b_open)
        row.addWidget(self.b_close)
        f.addRow(row)
        left.addWidget(pos)

        cam = QGroupBox("Andor (find the plane)")
        f = QFormLayout(cam)
        self.exposure = _dspin(0.001, 5, 0.1, 0.05, 3, " s")
        self.exposure.setToolTip("Used by Snap, Live, the sweep here, and the protocol's Andor sweeps.")
        self.gain = QSpinBox()
        self.gain.setRange(0, 300)
        f.addRow("Exposure", self.exposure)
        f.addRow("EM gain", self.gain)
        row = QHBoxLayout()
        self.b_snap = QPushButton("Snap")
        self.b_snap.clicked.connect(self._snap)
        self.b_live = QPushButton("Live")
        self.b_live.setCheckable(True)
        self.b_live.toggled.connect(self._toggle_live)
        row.addWidget(self.b_snap)
        row.addWidget(self.b_live)
        f.addRow(row)
        self.sweep_half = _dspin(4, 100, 40, 5, 0, " um")
        self.sweep_step = _dspin(0.5, 20, 2, 0.5, 1, " um")
        f.addRow("Sweep here +-", self.sweep_half)
        f.addRow("Sweep step", self.sweep_step)
        row = QHBoxLayout()
        self.b_sweep = QPushButton("Andor sweep here (lens)")
        self.b_sweep.setToolTip("Frames stepping the lens through its current z, then back. Nothing is saved.")
        self.b_sweep.clicked.connect(self._sweep_here)
        self.b_bright = QPushButton("Move lens to brightest")
        self.b_bright.clicked.connect(self._to_brightest)
        row.addWidget(self.b_sweep)
        row.addWidget(self.b_bright)
        f.addRow(row)
        row = QHBoxLayout()
        self.b_clear = QPushButton("Clear summed plot")
        self.b_clear.clicked.connect(self._clear_sum)
        self.b_park = QPushButton(f"Park lens on homodyne plane")
        self.b_park.setToolTip(f"Lens slew +-{PARK_HALF_UM:.0f} um through here; lens to the S plane.")
        self.b_park.clicked.connect(self._park)
        row.addWidget(self.b_clear)
        row.addWidget(self.b_park)
        f.addRow(row)
        left.addWidget(cam)

        cfg = QGroupBox("Protocol")
        f = QFormLayout(cfg)
        self.sample = QComboBox()
        self.sample.setEditable(True)
        self.sample.addItems(["glass", "phantom"])
        self.blocks = QSpinBox()
        self.blocks.setRange(1, 50)
        self.blocks.setValue(5)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(int(datetime.now().strftime("%H%M%S")))
        self.stationary = _dspin(1, 60, 10, 1, 1, " s")
        self.slew_half = _dspin(50, 400, 200, 10, 0, " um")
        self.speeds = QLineEdit("400")
        self.andor_half = _dspin(4, 100, 40, 5, 0, " um")
        self.andor_step = _dspin(0.5, 20, 2, 0.5, 1, " um")
        self.save_dir = QLineEdit(str(save_dir))
        for label, w in (("Sample", self.sample), ("Blocks", self.blocks), ("Seed", self.seed),
                         ("Stationary", self.stationary), ("Slew +-", self.slew_half), ("Speeds (um/s)", self.speeds),
                         ("Andor sweep +-", self.andor_half), ("Andor step", self.andor_step), ("Save to", self.save_dir)):
            f.addRow(label, w)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._update_info)
        self.exposure.valueChanged.connect(self._update_info)
        self.speeds.textChanged.connect(self._update_info)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        row = QHBoxLayout()
        self.b_run = QPushButton("Run protocol")
        self.b_run.setStyleSheet("font-weight: bold;")
        self.b_run.clicked.connect(self._run)
        self.b_stop = QPushButton("Stop")
        self.b_stop.clicked.connect(self._stop)
        row.addWidget(self.b_run)
        row.addWidget(self.b_stop)
        f.addRow(row)
        left.addWidget(cfg)
        left.addStretch(1)

        right = QVBoxLayout()
        root.addLayout(right, 1)
        right.addWidget(self._build_camera_view(), 5)
        self.sum_label = QLabel("Summed intensity: -")
        self.sum_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        right.addWidget(self.sum_label)
        self.fig = Figure(figsize=(9, 2.6), tight_layout=True)
        self.ax_s = self.fig.add_subplot(121)
        self.ax_a = self.fig.add_subplot(122)
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas, 2)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log, 1)
        self._controls = [b for b in panel.findChildren(QPushButton) if b not in (self.b_stop, self.b_live)]
        self._update_info()

    def _build_camera_view(self) -> pg.GraphicsLayoutWidget:
        """Image and spectrum as in the human interface, plus summed intensity against z."""
        pg.setConfigOptions(useOpenGL=False, antialias=True, imageAxisOrder="row-major", background="w", foreground="k")
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setSpacing(12)
        self.vb_img = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.vb_img.invertY(True)
        self.img_item = pg.ImageItem(autoDownsample=True)
        self.vb_img.addItem(self.img_item)
        self.vb_img.setBorder((170, 170, 170))
        self.glw.nextRow()
        self.spec_plot = self.glw.addPlot(labels={"left": "Intensity", "bottom": "Pixel (X)"})
        self.spec_plot.setMenuEnabled(False)
        self.spec_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4, symbolPen=pg.mkPen("k"), symbolBrush="k")
        self.spec_plot.addItem(self.spec_points)
        self.glw.nextRow()
        self.sum_plot = self.glw.addPlot(labels={"left": "Summed intensity", "bottom": "Lens z (um)"})
        self.sum_plot.setMenuEnabled(False)
        self.sum_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5, symbolPen=pg.mkPen("k"), symbolBrush="k")
        self.sum_best = pg.PlotDataItem(pen=None, symbol="o", symbolSize=10, symbolPen=pg.mkPen("r"), symbolBrush="r")
        self.sum_plot.addItem(self.sum_points)
        self.sum_plot.addItem(self.sum_best)
        return self.glw

    # ------------------------------------------------------------------ helpers

    def _log(self, text: str):
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in self._controls:
            b.setEnabled(not busy)
        self.b_stop.setEnabled(busy)
        self.b_live.setEnabled(not self._running)

    def _on_done(self):
        self._running = False
        self._set_busy(False)

    def _on_connected(self, err: str):
        if err:
            self._log(f"NOT CONNECTED: {err}")
            QMessageBox.critical(self, "Connection", err)
            self.b_stop.setEnabled(False)
            return
        self._log("Connected. Lens and stage were attached without moving.")
        self._set_busy(False)

    def _on_positions(self, lens: float, stage: float):
        self.lens_label.setText(f"{lens:.1f} um")
        self.stage_label.setText(f"{stage:.1f} um")

    def _on_failed(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.warning(self, "Error", msg)

    def _config(self) -> ProtocolConfig:
        speeds = tuple(float(s) for s in self.speeds.text().replace(";", ",").split(",") if s.strip())
        return ProtocolConfig(sample=self.sample.currentText().strip() or "sample", n_blocks=self.blocks.value(),
                              seed=self.seed.value(), stationary_s=self.stationary.value(),
                              slew_half_um=self.slew_half.value(), speeds_um_s=speeds,
                              andor_half_um=self.andor_half.value(), andor_step_um=self.andor_step.value(),
                              exposure_s=self.exposure.value(), gain=self.gain.value())

    def _update_info(self, *_):
        try:
            c = self._config()
            c.validate()
            self.info.setText(f"{len(c.steps())} steps, about {c.estimated_s() / 60:.1f} min. Each block: stationary + "
                              f"{4 * len(c.speeds_um_s)} slews (lens and stage, fwd and bwd). Andor: "
                              f"{c.andor_frames()} frames per sweep, 4 sweeps.")
            self.info.setStyleSheet("")
        except ValueError as e:
            self.info.setText(str(e))
            self.info.setStyleSheet("color: #c0392b;")

    # ------------------------------------------------------------------ Andor view

    def _snap(self):
        if not self._snap_pending:
            self._snap_pending = True
            self._req_snap.emit(self.exposure.value(), self.gain.value(), False)

    def _toggle_live(self, on: bool):
        if on and not self._snap_pending:
            self._snap_pending = True
            self._req_snap.emit(self.exposure.value(), self.gain.value(), True)

    def _on_frame(self, frame, axis: str, z: float, live: bool):
        if live or not (self._running or self._busy):
            self._snap_pending = False
        if frame is None:
            self.b_live.setChecked(False)
            self._snap_pending = False
            return
        if axis != self._sum_axis:
            self._sum_axis = axis
            self._clear_sum()
            self.sum_plot.setLabel("bottom", f"{axis.capitalize()} z (um)")
        elif not live and self._sum_z and z < self._sum_z[-1] - 1e-6 and (self._running or self._busy):
            self._clear_sum()       # a new sweep started
        total = self._display_frame(frame, axis, z)
        self._sum_z.append(z)
        self._sum_v.append(total)
        del self._sum_z[:-MAX_SUM_POINTS], self._sum_v[:-MAX_SUM_POINTS]
        self._update_sum_plot()
        if live and self.b_live.isChecked() and not self._running:
            self._snap_pending = True
            self._req_snap.emit(self.exposure.value(), self.gain.value(), True)

    def _display_frame(self, frame: np.ndarray, axis: str, z: float) -> float:
        """Min-max per frame, as in the human interface. Returns the summed spectrum intensity."""
        frame = np.ascontiguousarray(frame).astype(np.float32, copy=False)
        fmin, fmax = float(np.nanmin(frame)), float(np.nanmax(frame))
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            fmin, fmax = 0.0, 1.0
        self.img_item.setImage(frame, autoLevels=False, levels=(fmin, fmax))
        px, sline = self._fitter.get_px_sline_from_image(frame)
        self.spec_points.setData(px, sline)
        total = self._fitter.get_total_sline_value(sline)
        flag = "   NEAR SATURATION" if fmax >= SATURATION_COUNTS else ""
        self.spec_plot.setTitle(f"Spectrum at {axis} z = {z:.1f} um, frame max {fmax:.0f} counts{flag}",
                                color=(192, 57, 43) if flag else (40, 40, 40))
        self._current_sum = (z, total)
        return total

    def _update_sum_plot(self):
        self.sum_points.setData(self._sum_z, self._sum_v)
        if self._sum_v:
            i = int(np.argmax(self._sum_v))
            self.sum_best.setData([self._sum_z[i]], [self._sum_v[i]])
        else:
            self.sum_best.setData([], [])
        z, total = self._current_sum if self._current_sum is not None else (None, None)
        text = "Summed intensity: -" if z is None else f"Summed intensity {total:.0f} at {self._sum_axis} z = {z:.1f} um"
        if self._sum_v:
            i = int(np.argmax(self._sum_v))
            text += f"     brightest {self._sum_v[i]:.0f} at {self._sum_z[i]:.1f} um"
        self.sum_label.setText(text)

    def _clear_sum(self):
        self._sum_z, self._sum_v = [], []
        self._current_sum = None
        self._fitter = SpectrumFitter()     # its automatic row band may have frozen on a dark frame
        self._update_sum_plot()

    def _sweep_here(self):
        self.b_live.setChecked(False)
        self._sum_axis = "lens"
        self._clear_sum()
        self._set_busy(True)
        self._req_sweep.emit(self.exposure.value(), self.gain.value(), self.sweep_half.value(), self.sweep_step.value())

    def _to_brightest(self):
        if self._sum_axis != "lens" or not self._sum_v:
            QMessageBox.information(self, "Brightest", "Run 'Andor sweep here' (or Live while jogging the lens) first.")
            return
        self._set_busy(True)
        self._req_lens_to.emit(self._sum_z[int(np.argmax(self._sum_v))])

    # ------------------------------------------------------------------ actions

    def _jog(self, axis: str, sign: int):
        if self._busy:
            return
        if axis == "stage":
            self._clear_sum()
        self._set_busy(True)
        self._req_jog.emit(axis, sign * self.jog_step.value())

    def _shutter(self, is_open: bool):
        self._set_busy(True)
        self._req_shutter.emit(is_open)

    def _park(self):
        self.b_live.setChecked(False)
        self._set_busy(True)
        self._req_park.emit()

    def _stop(self):
        self._worker.stop.set()
        self.b_live.setChecked(False)

    def _run(self):
        try:
            cfg = self._config()
            cfg.validate()
        except ValueError as e:
            QMessageBox.warning(self, "Protocol", str(e))
            return
        name = f"zcarrier_{cfg.sample.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = Path(self.save_dir.text()) / name
        r = QMessageBox.question(
            self, "Run",
            f"Lens on the plane and the sample shutters open?\n\nBoth the lens and stage Z will move up to "
            f"{cfg.window_half_um():.0f} um either side of where they are now (stage Z moves the whole head, cameras "
            f"included).\n\n{len(cfg.steps())} steps, about {cfg.estimated_s() / 60:.1f} min.\nFolder: {run_dir}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        self.b_live.setChecked(False)
        self._running = True
        self._set_busy(True)
        self._log(f"Run started: {run_dir}")
        self._req_run.emit(cfg, run_dir)

    # ------------------------------------------------------------------ result plots

    def _on_result(self, row, rec):
        if row is not None:
            pk = row.get("peak_rel_um")
            self._log(f"  {row['file']}: {row['status']}" + (f", peak {pk:+.1f} um from start" if pk is not None else "")
                      + (f", DC {row['dc_v']} V" if row.get("dc_v") else ""))
        if hasattr(rec, "frames"):
            z = rec.z_um - rec.meta.get("z0_um", 0.0)
            self.ax_a.clear()
            self.ax_a.semilogy(z, rec.frame_sums(), "o-", ms=3)
            self.ax_a.set(xlabel=f"{rec.meta.get('axis', '')} z - start (um)", ylabel="Andor frame sum",
                          title=f"Last protocol Andor sweep: {rec.meta.get('axis', '')}")
        elif getattr(rec, "mode", "") == "slew":
            z = rec.z_um()
            if z is not None:
                s = zero_phase_s(rec.values, rec.sample_rate_hz, HomodyneFinderParams())
                n0 = int(PRE_ROLL_S * rec.sample_rate_hz)
                self.ax_s.clear()
                self.ax_s.plot(z[n0:] - rec.meta.get("z0_um", z[n0]), s[n0:], lw=0.8)
                axis = rec.meta.get("axis", "lens") or "lens"
                d = rec.meta.get("direction", 1)
                self.ax_s.set(xlabel=f"{axis} z - start (um)", ylabel="S (V^0.5)",
                              title=f"Last slew: {axis} {'fwd' if d > 0 else 'bwd'}")
        self.canvas.draw_idle()

    def closeEvent(self, event):
        self._worker.stop.set()
        self._req_shutdown.emit()
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(event)
