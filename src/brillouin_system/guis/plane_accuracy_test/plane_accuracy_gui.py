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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from brillouin_system.guis.homodyne_test.experiment_plan import (
    StageRef, stage_ref_distance_mm, stage_ref_sphere_radius_mm,
)
from brillouin_system.guis.plane_accuracy_test.runner import (
    Devices, RunConfig, open_sample_path, run_experiment, run_find,
)
from brillouin_system.scan_managers.homodyne_plane_finder import HomodyneFinderParams
from brillouin_system.scan_managers.homodyne_processing import process_channels

DEFAULT_SAVE_DIR = Path.home() / "plane_accuracy_runs"


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
                                    on_trial=lambda row, rec: self.trace.emit(row, rec))
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
        self._worker.run_finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_failed)
        self._req_connect.connect(self._worker.connect_all)
        self._req_stage.connect(self._worker.jog_stage)
        self._req_lens.connect(self._worker.jog_lens)
        self._req_shutter.connect(self._worker.set_shutter)
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
        self.exposure = _dspin(0.001, 60.0, 0.2, 0.05, 3, " s")
        self.gain = QSpinBox()
        self.gain.setRange(0, 1000)
        self.close_end = QCheckBox("Close shutters at the end")
        self.close_end.setChecked(True)
        self.save_dir = QLineEdit(str(save_dir))
        self.info = QLabel()
        self.info.setWordWrap(True)
        for label, w in (("R step:", self.r_step), ("Blocks:", self.blocks), ("Order seed:", self.seed),
                         ("Find from plane ±:", self.half), ("Finder speed:", self.speed),
                         ("Finder threshold:", self.n_sigma), ("Finder stop after:", self.stop_after),
                         ("Axial before plane:", self.ax_before), ("Axial after plane:", self.ax_after),
                         ("Axial step:", self.ax_step), ("Andor exposure:", self.exposure), ("Andor gain:", self.gain)):
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
        self.fig = Figure(figsize=(9, 7), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax_s = self.fig.add_subplot(311)
        self.ax_axial = self.fig.add_subplot(312)
        self.ax_err = self.fig.add_subplot(313)
        right.addWidget(self.canvas, 3)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        right.addWidget(self.log_box, 1)
        root.addLayout(right, 1)
        self._controls = [w for w in panel.findChildren(QPushButton) if w is not self.stop_btn]
        self._err = {"find_fwd": ([], []), "find_bwd": ([], []), "axial": ([], [])}
        self._update_info()

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

    def _find(self, direction: float):
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
        a = self.ax_axial
        a.clear()
        s = rec.frame_sums()
        a.plot(rec.z_um, s, "o-", ms=3, lw=0.8, color="#1f1f1e")
        a.axvline(row["lens_pred_um"], color="#8a8a85", lw=0.8, ls=":")
        a.set(xlabel="lens z (um)", ylabel="frame sum (counts)", title=f"Last axial scan, R = {row['r_mm']:.2f} mm")

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
        self._req_shutdown.emit()
        self._thread.quit()
        self._thread.wait(5000)
        event.accept()
