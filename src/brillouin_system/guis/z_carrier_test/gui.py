"""
Lens-Z vs stage-Z carrier test (standalone; borrows the DAQ, Zaber, Andor and shutter modules).

Park the lens on the plane, then Run protocol: Andor fine sweeps along lens and stage Z, then blocks of a
stationary record plus lens and stage-Z slews in random order, then the Andor sweeps again. Everything is
centred on where the lens and stage were when Run was pressed, and both go back there after every step.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from brillouin_system.guis.plane_accuracy_test.runner import open_sample_path
from brillouin_system.guis.z_carrier_test.motion import SimAxisMover, ZaberAxisMover
from brillouin_system.guis.z_carrier_test.protocol import PRE_ROLL_S, ProtocolConfig
from brillouin_system.guis.z_carrier_test.runner import Rig, park_lens_on_plane, run_protocol
from brillouin_system.scan_managers.homodyne_plane_finder import HomodyneFinderParams, zero_phase_s

PARK_HALF_UM = 150.0
PARK_SPEED_UM_S = 400.0


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

    @pyqtSlot()
    def park(self):
        def go():
            self.stop.clear()
            self.rig.daq.configure(sample_rate_hz=5000.0)
            z, rec = park_lens_on_plane(self.rig, PARK_HALF_UM, PARK_SPEED_UM_S, self.stop)
            self.status.emit(f"Lens parked on the plane at {z:.1f} um." if z is not None
                             else "No clear plane within the slew; lens left where it was.")
            self.result.emit(None, rec)
        self._guarded(go)

    @pyqtSlot(object, object)
    def run(self, cfg: ProtocolConfig, run_dir: Path):
        def go():
            self.stop.clear()
            c = run_protocol(self.rig, cfg, run_dir, self.stop, on_status=self.status.emit,
                             on_result=lambda row, rec: self.result.emit(row, rec))
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
    _req_park = pyqtSignal()
    _req_run = pyqtSignal(object, object)
    _req_shutdown = pyqtSignal()

    def __init__(self, *, use_dummy: bool, lens_port: str = "COM5", stage_port: str = "COM6", ni_device: str = "Dev1",
                 save_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Lens Z vs stage Z carrier test" + ("  [DUMMY]" if use_dummy else ""))
        self._busy = True
        self._build_ui(save_dir or Path.home() / "Documents" / "z_carrier_runs")
        self._thread = QThread(self)
        self._worker = Worker((use_dummy, lens_port, stage_port, ni_device))
        self._worker.moveToThread(self._thread)
        self._worker.status.connect(self._log)
        self._worker.positions.connect(self._on_positions)
        self._worker.connected.connect(self._on_connected)
        self._worker.result.connect(self._on_result)
        self._worker.done.connect(lambda: self._set_busy(False))
        self._worker.failed.connect(self._on_failed)
        self._req_connect.connect(self._worker.connect_all)
        self._req_jog.connect(self._worker.jog)
        self._req_shutter.connect(self._worker.shutter)
        self._req_park.connect(self._worker.park)
        self._req_run.connect(self._worker.run)
        self._req_shutdown.connect(self._worker.shutdown)
        self._thread.start()
        self._set_busy(True)
        self._req_connect.emit()

    # ------------------------------------------------------------------ UI

    def _build_ui(self, save_dir: Path):
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 0)

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
                b = QPushButton(f"{axis} {text}")
                b.clicked.connect(lambda _, a=axis, s=sign: self._jog(a, s))
                row.addWidget(b)
            f.addRow(row)
        left.addWidget(pos)

        sh = QGroupBox("Shutters")
        r = QHBoxLayout(sh)
        self.b_open, self.b_close = QPushButton("Open sample"), QPushButton("Close all")
        self.b_open.clicked.connect(lambda: self._shutter(True))
        self.b_close.clicked.connect(lambda: self._shutter(False))
        r.addWidget(self.b_open)
        r.addWidget(self.b_close)
        left.addWidget(sh)

        self.b_park = QPushButton(f"Park lens on plane (lens slew +-{PARK_HALF_UM:.0f} um)")
        self.b_park.clicked.connect(self._park)
        left.addWidget(self.b_park)

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
        self.exposure = _dspin(0.001, 5, 0.1, 0.05, 3, " s")
        self.gain = QSpinBox()
        self.gain.setRange(0, 300)
        self.save_dir = QLineEdit(str(save_dir))
        for label, w in (("Sample", self.sample), ("Blocks", self.blocks), ("Seed", self.seed),
                         ("Stationary", self.stationary), ("Slew +-", self.slew_half), ("Speeds (um/s)", self.speeds),
                         ("Andor +-", self.andor_half), ("Andor step", self.andor_step), ("Exposure", self.exposure),
                         ("EM gain", self.gain), ("Save to", self.save_dir)):
            f.addRow(label, w)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._update_info)
        self.speeds.textChanged.connect(self._update_info)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        left.addWidget(cfg)

        row = QHBoxLayout()
        self.b_run = QPushButton("Run protocol")
        self.b_run.setStyleSheet("font-weight: bold;")
        self.b_run.clicked.connect(self._run)
        self.b_stop = QPushButton("Stop")
        self.b_stop.clicked.connect(lambda: self._worker.stop.set())
        row.addWidget(self.b_run)
        row.addWidget(self.b_stop)
        left.addLayout(row)
        left.addStretch(1)

        right = QVBoxLayout()
        root.addLayout(right, 1)
        self.fig = Figure(figsize=(8, 7), tight_layout=True)
        self.ax_s = self.fig.add_subplot(211)
        self.ax_a = self.fig.add_subplot(212)
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas, 3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log, 1)
        self._update_info()

    # ------------------------------------------------------------------ helpers

    def _log(self, text: str):
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    def _set_busy(self, busy: bool):
        self._busy = busy
        for b in (self.b_park, self.b_run, self.b_open, self.b_close):
            b.setEnabled(not busy)
        self.b_stop.setEnabled(busy)

    def _on_connected(self, err: str):
        self._set_busy(False)
        if err:
            self._set_busy(True)
            self.b_stop.setEnabled(False)
            self._log(f"NOT CONNECTED: {err}")
            QMessageBox.critical(self, "Connection", err)
        else:
            self._log("Connected. Lens and stage were attached without moving.")

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
            n = len(c.steps())
            self.info.setText(f"{n} steps, about {c.estimated_s() / 60:.1f} min. Each block: stationary + "
                              f"{4 * len(c.speeds_um_s)} slews (lens and stage, fwd and bwd). Andor: "
                              f"{c.andor_frames()} frames per sweep, 4 sweeps.")
            self.info.setStyleSheet("")
        except ValueError as e:
            self.info.setText(str(e))
            self.info.setStyleSheet("color: #c0392b;")

    # ------------------------------------------------------------------ actions

    def _jog(self, axis: str, sign: int):
        if self._busy:
            return
        self._set_busy(True)
        self._req_jog.emit(axis, sign * self.jog_step.value())

    def _shutter(self, is_open: bool):
        self._set_busy(True)
        self._req_shutter.emit(is_open)

    def _park(self):
        self._set_busy(True)
        self._req_park.emit()

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
            f"Lens on the plane and the sample shutter path ready?\n\nBoth the lens and stage Z will move up to "
            f"{cfg.window_half_um():.0f} um either side of where they are now (stage Z moves the whole head, cameras "
            f"included).\n\n{len(cfg.steps())} steps, about {cfg.estimated_s() / 60:.1f} min.\nFolder: {run_dir}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        self._set_busy(True)
        self._log(f"Run started: {run_dir}")
        self._req_run.emit(cfg, run_dir)

    # ------------------------------------------------------------------ plots

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
                          title=f"Andor sweep, {rec.meta.get('axis', '')}")
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
