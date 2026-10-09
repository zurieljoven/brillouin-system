"""
Homodyne plane test GUI (standalone).

Attaches to the eye-lens Zaber WITHOUT homing or moving it, so the current
position (assumed to be at the reflection plane) is kept. Two actions:

  1. Record (no motion): acquire both DAQ channels for a set time.
  2. Back off + slew: move back by a set distance, slew over a set axial
     range at a set speed while recording (Zaber position logged on the
     same perf_counter timeline), then optionally return to the start z.

The Shutters group opens the sample (objective) shutter, since closing the
main GUI closes all shutters; nothing is opened on startup and all shutters
are closed when this GUI exits.

Every acquisition is plotted (raw channels, fringe envelopes, and
S = sqrt(A_H^2/DC_H + A_V^2/DC_V)). Nothing is saved automatically: "Save
recording" writes the last acquisition to .npz (load with
homodyne_recording.load_recording); the GUI asks before discarding an
unsaved recording.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from brillouin_system.devices.ni.ni6008_multi import DIFF_RANGES_V, effective_range_v
from brillouin_system.scan_managers.homodyne_processing import (
    fringe_band_hz, fringe_frequency_hz, locate_peak, process_channels,
)
from brillouin_system.scan_managers.homodyne_recording import (
    HomodyneRecording, load_recording, save_recording,
)

# Motion limits for this test tool (eye-lens axis).
MAX_SPEED_UM_S = 2000.0
MAX_RANGE_UM = 3000.0
PRE_ROLL_S = 0.3        # DAQ runs this long before the slew starts (background)
POST_ROLL_S = 0.3       # and this long after the slew should have ended
SETTLE_S = 0.2          # pause after the back-off move
Z_POLL_S = 0.016        # Zaber position-log period (same as the plane finder)


@dataclass(frozen=True)
class DaqSettings:
    channels: tuple[str, ...]
    sample_rate_hz: float
    terminal: str
    v_range: float


@dataclass(frozen=True)
class SlewSettings:
    backoff_um: float
    range_um: float
    speed_um_s: float
    reverse: bool           # False: back off toward -z, slew toward +z
    return_to_start: bool


class _Cancelled(Exception):
    pass


class HomodyneWorker(QObject):
    """Runs DAQ acquisitions and Zaber motion off the GUI thread."""

    status = pyqtSignal(str)
    position = pyqtSignal(float)
    finished = pyqtSignal(object)   # HomodyneRecording
    failed = pyqtSignal(str)

    def __init__(self, daq, zaber):
        super().__init__()
        self.daq = daq
        self.zaber = zaber
        self._stop = threading.Event()

    def request_stop(self) -> None:
        """Thread-safe; called directly from the GUI thread."""
        self._stop.set()

    def _apply(self, d: DaqSettings) -> None:
        self.daq.configure(channels=d.channels, sample_rate_hz=d.sample_rate_hz,
                           terminal=d.terminal, v_range=d.v_range)
        self.daq.validate()

    @pyqtSlot()
    def read_position(self):
        try:
            self.position.emit(float(self.zaber.get_position()))
        except Exception as e:
            self.failed.emit(f"Position read failed: {type(e).__name__}: {e}")

    @pyqtSlot(object, float, object)
    def run_stationary(self, daq_settings: DaqSettings, duration_s: float, meta: dict):
        self._stop.clear()
        try:
            self._apply(daq_settings)
            z = float(self.zaber.get_position())
            self.position.emit(z)
            self.status.emit(f"Recording {duration_s:.1f} s, no motion ...")
            res = self.daq.acquire(duration_s, stop_evt=self._stop)
            meta = dict(meta, z_um=z, stopped_early=self._stop.is_set(),
                        duration_s=res.values.shape[-1] / res.sample_rate_hz)
            self.finished.emit(HomodyneRecording(
                mode="stationary", values=res.values, sample_rate_hz=res.sample_rate_hz,
                t0_perf=res.t0_perf, channels=tuple(res.channels), meta=meta))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    @pyqtSlot(object, object, object)
    def run_slew(self, daq_settings: DaqSettings, s: SlewSettings, meta: dict):
        self._stop.clear()
        zaber = self.zaber
        z0: Optional[float] = None
        moved = False
        state = {"log": False, "slew": False}
        zlog = None
        res = None
        error: Optional[str] = None
        try:
            self._apply(daq_settings)
            z0 = float(zaber.get_position())
            self.position.emit(z0)

            direction = -1.0 if s.reverse else 1.0     # slew direction
            z_start = z0 - direction * s.backoff_um
            self.status.emit(f"Backing off {s.backoff_um:.0f} um to z = {z_start:.1f} um ...")
            moved = True
            zaber.move_abs(z_start)
            if self._stop.is_set():
                raise _Cancelled()
            time.sleep(SETTLE_S)

            fs = self.daq.sample_rate_hz
            t_motion = s.range_um / s.speed_um_s
            duration = PRE_ROLL_S + t_motion + 0.2 + POST_ROLL_S   # +0.2 s accel margin
            n_pre = int(PRE_ROLL_S * fs)

            def on_chunk(n: int) -> None:
                if not state["log"]:
                    zaber.start_position_log(poll_s=Z_POLL_S, alpha=0.25)
                    state["log"] = True
                if not state["slew"] and n >= n_pre:
                    zaber.start_slewing_guarded(direction * s.speed_um_s, s.range_um)
                    state["slew"] = True

            self.status.emit(f"Slewing {s.range_um:.0f} um at {s.speed_um_s:.0f} um/s "
                             f"({duration:.1f} s record) ...")
            res = self.daq.acquire(duration, on_chunk=on_chunk, stop_evt=self._stop)
            meta = dict(meta, z0_um=z0, z_start_um=z_start, direction=direction,
                        pre_roll_s=PRE_ROLL_S, post_roll_s=POST_ROLL_S,
                        stopped_early=self._stop.is_set(), **asdict(s))
        except _Cancelled:
            error = "Stopped before the slew."
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        finally:
            if state["slew"] or moved:
                try:
                    zaber.stop_slewing()
                except Exception as e:
                    error = error or f"stop_slewing failed: {e}"
            if state["log"]:
                try:
                    zlog = zaber.stop_position_log()
                except Exception:
                    zlog = None
            if moved and s.return_to_start and z0 is not None:
                try:
                    self.status.emit(f"Returning to start z = {z0:.1f} um ...")
                    zaber.move_abs(z0)
                except Exception as e:
                    error = (error + "; " if error else "") + f"RETURN TO START FAILED: {e}"
            try:
                self.position.emit(float(zaber.get_position()))
            except Exception:
                pass

        if error is not None and res is None:
            self.failed.emit(error)
            return
        if error is not None:
            self.status.emit(f"Warning: {error}")
        self.finished.emit(HomodyneRecording(
            mode="slew", values=res.values, sample_rate_hz=res.sample_rate_hz,
            t0_perf=res.t0_perf, channels=tuple(res.channels), meta=meta,
            zlog_t_perf=np.asarray(zlog.t_perf) if zlog is not None else np.empty(0),
            zlog_z_um=np.asarray(zlog.z_um) if zlog is not None else np.empty(0)))


def _dspin(lo, hi, val, step, decimals=1, suffix="") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class HomodyneTestWindow(QWidget):
    _req_stationary = pyqtSignal(object, float, object)
    _req_slew = pyqtSignal(object, object, object)
    _req_position = pyqtSignal()

    def __init__(self, *, use_dummy: bool, zaber_port: str = "COM5", ni_device: str = "Dev1",
                 save_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Homodyne Plane Test" + ("  [DUMMY]" if use_dummy else ""))
        self.use_dummy = use_dummy
        self.ni_device = ni_device
        self._busy = False
        self._last_rec: Optional[HomodyneRecording] = None
        self._last_path: Optional[Path] = None
        self._unsaved = False   # last acquisition not saved yet

        self.daq, self.zaber, connect_error = self._connect(use_dummy, zaber_port, ni_device)
        self.shutters, shutter_error = self._connect_shutters(use_dummy)
        if shutter_error:
            connect_error = f"{connect_error}\n{shutter_error}" if connect_error else shutter_error

        self._build_ui(save_dir or Path.home() / "Documents" / "homodyne_recordings")
        self._update_fringe_info()

        self._thread = QThread(self)
        self._worker = None
        if self.zaber is not None:
            self._worker = HomodyneWorker(self.daq, self.zaber)
            self._worker.moveToThread(self._thread)
            self._worker.status.connect(self._set_status)
            self._worker.position.connect(self._set_position)
            self._worker.finished.connect(self._on_finished)
            self._worker.failed.connect(self._on_failed)
            self._req_stationary.connect(self._worker.run_stationary)
            self._req_slew.connect(self._worker.run_slew)
            self._req_position.connect(self._worker.read_position)
            self._thread.start()
            self._req_position.emit()
        self._set_busy(False)

        if connect_error:
            self._set_status(connect_error)
            QMessageBox.critical(self, "Hardware connection failed", connect_error)

    # ------------------------------------------------------------------ devices

    @staticmethod
    def _connect(use_dummy: bool, zaber_port: str, ni_device: str):
        try:
            if use_dummy:
                from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
                from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
                zaber = SimZaberLens(start_um=8000.0)
                # Dummy starts AT the plane, like the real use case.
                return SimHomodyneDAQ(zaber, plane_um=8000.0), zaber, None

            from brillouin_system.devices.ni.ni6008_multi import NI6008Multi
            from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
            # home_on_connect=False: attach WITHOUT homing or moving the lens.
            zaber = ZaberEyeLens(port=zaber_port, home_on_connect=False)
            return NI6008Multi(device=ni_device), zaber, None
        except Exception as e:
            return None, None, f"Could not connect ({type(e).__name__}: {e}). Actions are disabled."

    @staticmethod
    def _connect_shutters(use_dummy: bool):
        """Same shutters as the main human-interface GUI. Nothing is opened here."""
        try:
            if use_dummy:
                from brillouin_system.devices.shutter_device import ShutterManagerDummy
                return ShutterManagerDummy("human_interface"), None
            from brillouin_system.devices.shutter_device import ShutterManager
            return ShutterManager("human_interface"), None
        except Exception as e:
            return None, f"Shutters not available ({type(e).__name__}: {e})."

    def _open_sample_shutter(self):
        """Sample mode, as in the main GUI: reference closed, objective open."""
        try:
            self.shutters.change_to_objective()
            self.shutter_label.setText("Sample (objective) OPEN")
            self.shutter_label.setStyleSheet("font-weight: bold; color: #c0392b;")
        except Exception as e:
            QMessageBox.warning(self, "Shutter", f"{type(e).__name__}: {e}")

    def _close_shutters(self):
        try:
            self.shutters.close_all()
            self.shutter_label.setText("All closed")
            self.shutter_label.setStyleSheet("font-weight: bold;")
        except Exception as e:
            QMessageBox.warning(self, "Shutter", f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ UI

    def _build_ui(self, save_dir: Path):
        controls = QWidget()
        cl = QVBoxLayout(controls)

        # Position
        g = QGroupBox("Eye-lens position (not moved on startup)")
        f = QFormLayout(g)
        self.pos_label = QLabel("—")
        self.pos_label.setStyleSheet("font-weight: bold;")
        self.read_pos_btn = QPushButton("Read position")
        self.read_pos_btn.clicked.connect(lambda: self._req_position.emit())
        f.addRow("z (um):", self.pos_label)
        f.addRow(self.read_pos_btn)
        cl.addWidget(g)

        # Shutters (state unknown at startup: the main GUI closes all on exit)
        g = QGroupBox("Shutters")
        f = QFormLayout(g)
        self.shutter_label = QLabel("Unknown (not changed)")
        self.shutter_label.setStyleSheet("font-weight: bold;")
        self.open_shutter_btn = QPushButton("Open sample shutter")
        self.open_shutter_btn.setToolTip("Sample mode: closes the reference shutter and opens the "
                                         "objective shutter (laser reaches the sample).")
        self.open_shutter_btn.clicked.connect(self._open_sample_shutter)
        self.close_shutter_btn = QPushButton("Close all shutters")
        self.close_shutter_btn.clicked.connect(self._close_shutters)
        row = QHBoxLayout()
        row.addWidget(self.open_shutter_btn)
        row.addWidget(self.close_shutter_btn)
        f.addRow("State:", self.shutter_label)
        f.addRow(row)
        for w in (self.open_shutter_btn, self.close_shutter_btn):
            w.setEnabled(self.shutters is not None)
        cl.addWidget(g)

        # DAQ
        g = QGroupBox(f"DAQ ({'simulated' if self.use_dummy else self.ni_device})")
        f = QFormLayout(g)
        self.ch_h = QLineEdit("ai0")
        self.ch_v = QLineEdit("ai1")
        self.terminal = QComboBox()
        self.terminal.addItems(["RSE", "DIFF"])
        self.v_range = QComboBox()
        self.v_range.addItems([f"{r:g}" for r in DIFF_RANGES_V])
        self.v_range.setCurrentText("10")
        self.fs = QSpinBox()
        self.fs.setRange(100, 5000)
        self.fs.setSingleStep(500)
        self.fs.setValue(5000)
        self.fs.setSuffix(" S/s per ch")
        self.terminal.currentTextChanged.connect(self._update_fringe_info)
        self.v_range.currentTextChanged.connect(self._update_fringe_info)
        self.fs.valueChanged.connect(self._update_fringe_info)
        f.addRow("Channel H:", self.ch_h)
        f.addRow("Channel V:", self.ch_v)
        f.addRow("Terminal:", self.terminal)
        f.addRow("Range (+- V):", self.v_range)
        f.addRow("Sample rate:", self.fs)
        cl.addWidget(g)

        # 1. Stationary
        g = QGroupBox("1. Record (no motion)")
        f = QFormLayout(g)
        self.duration = _dspin(0.1, 120.0, 10.0, 1.0, 1, " s")
        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self._start_stationary)
        f.addRow("Duration:", self.duration)
        f.addRow(self.record_btn)
        cl.addWidget(g)

        # 2. Slew
        g = QGroupBox("2. Back off + slew")
        f = QFormLayout(g)
        self.backoff = _dspin(0.0, MAX_RANGE_UM, 150.0, 10.0, 1, " um")
        self.scan_range = _dspin(1.0, MAX_RANGE_UM, 300.0, 10.0, 1, " um")
        self.speed = _dspin(1.0, MAX_SPEED_UM_S, 400.0, 50.0, 1, " um/s")
        self.reverse = QCheckBox("Back off toward +z, slew toward -z")
        self.return_to_start = QCheckBox("Return to start z afterwards")
        self.return_to_start.setChecked(True)
        self.fringe_info = QLabel()
        self.fringe_info.setWordWrap(True)
        self.slew_btn = QPushButton("Back off + slew")
        self.slew_btn.clicked.connect(self._start_slew)
        self.speed.valueChanged.connect(self._update_fringe_info)
        f.addRow("Back off:", self.backoff)
        f.addRow("Scan range:", self.scan_range)
        f.addRow("Speed:", self.speed)
        f.addRow(self.reverse)
        f.addRow(self.return_to_start)
        f.addRow(self.fringe_info)
        f.addRow(self.slew_btn)
        cl.addWidget(g)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self._stop)
        cl.addWidget(self.stop_btn)

        # Processing
        g = QGroupBox("Processing")
        f = QFormLayout(g)
        self.wavelength = _dspin(400.0, 1600.0, 780.24, 0.01, 2, " nm")
        self.auto_band = QCheckBox("Band from slew speed (fringe ±40%)")
        self.auto_band.setChecked(True)
        self.band_lo = _dspin(1.0, 50_000.0, 600.0, 50.0, 0, " Hz")
        self.band_hi = _dspin(1.0, 50_000.0, 1450.0, 50.0, 0, " Hz")
        self.dc_cutoff = _dspin(0.1, 1000.0, 20.0, 5.0, 1, " Hz")
        self.dark_h = _dspin(-1.0, 1.0, 0.0, 0.001, 4, " V")
        self.dark_v = _dspin(-1.0, 1.0, 0.0, 0.001, 4, " V")
        self.reprocess_btn = QPushButton("Re-process last")
        self.reprocess_btn.clicked.connect(lambda: self._last_rec is not None and self._show(self._last_rec))
        self.auto_band.toggled.connect(self._update_fringe_info)
        self.wavelength.valueChanged.connect(self._update_fringe_info)
        f.addRow("Wavelength:", self.wavelength)
        f.addRow(self.auto_band)
        f.addRow("Band low:", self.band_lo)
        f.addRow("Band high:", self.band_hi)
        f.addRow("DC low-pass:", self.dc_cutoff)
        f.addRow("Dark offset H:", self.dark_h)
        f.addRow("Dark offset V:", self.dark_v)
        f.addRow(self.reprocess_btn)
        cl.addWidget(g)

        # Save / load
        g = QGroupBox("Save")
        f = QFormLayout(g)
        self.save_dir = QLineEdit(str(save_dir))
        browse = QPushButton("...")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse_dir)
        row = QHBoxLayout()
        row.addWidget(self.save_dir)
        row.addWidget(browse)
        self.save_btn = QPushButton("Save recording ...")
        self.save_btn.clicked.connect(self._save_file)
        self.load_btn = QPushButton("Load recording ...")
        self.load_btn.clicked.connect(self._load_file)
        f.addRow("Folder:", row)
        f.addRow(self.save_btn)
        f.addRow(self.load_btn)
        cl.addWidget(g)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        cl.addWidget(self.status_label)
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(370)

        # Plots + summary
        self.fig = Figure(figsize=(9, 8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(170)
        right = QVBoxLayout()
        right.addWidget(NavigationToolbar2QT(self.canvas, self))
        right.addWidget(self.canvas, stretch=1)
        right.addWidget(self.summary)

        main = QHBoxLayout(self)
        main.addWidget(scroll)
        main.addLayout(right, stretch=1)

    # ------------------------------------------------------------------ helpers

    def _daq_settings(self) -> DaqSettings:
        return DaqSettings(
            channels=(self.ch_h.text().strip(), self.ch_v.text().strip()),
            sample_rate_hz=float(self.fs.value()),
            terminal=self.terminal.currentText(),
            v_range=float(self.v_range.currentText()),
        )

    def _base_meta(self) -> dict:
        return {
            "created": datetime.now().isoformat(timespec="seconds"),
            "dummy": self.use_dummy,
            "ni_device": self.ni_device,
            "wavelength_nm": self.wavelength.value(),
            "effective_range_v": effective_range_v(self.terminal.currentText(),
                                                   float(self.v_range.currentText())),
            **asdict(self._daq_settings()),
        }

    def _band(self, fs: float, speed_um_s: Optional[float]) -> tuple[float, float]:
        if self.auto_band.isChecked():
            speed = speed_um_s if speed_um_s else self.speed.value()
            return fringe_band_hz(speed, self.wavelength.value(), fs)
        return self.band_lo.value(), self.band_hi.value()

    def _update_fringe_info(self, *_):
        fs = float(self.fs.value())
        f0 = fringe_frequency_hz(self.speed.value(), self.wavelength.value())
        nyq = fs / 2
        text = f"Fringe: {f0:.0f} Hz   (Nyquist {nyq:.0f} Hz)"
        if f0 >= 0.95 * nyq:
            text += "\n<b>Above Nyquist: lower the speed!</b>"
        elif f0 > 0.7 * nyq:
            text += "\nClose to Nyquist; check your anti-alias filter."
        if self.terminal.currentText() == "RSE" and float(self.v_range.currentText()) != 10.0:
            text += "\nUSB-6008 RSE is fixed at ±10 V (range ignored)."
        self.fringe_info.setText(text)
        try:
            lo, hi = fringe_band_hz(self.speed.value(), self.wavelength.value(), fs)
            if self.auto_band.isChecked():
                self.band_lo.setValue(lo)
                self.band_hi.setValue(hi)
        except ValueError:
            pass
        self.band_lo.setEnabled(not self.auto_band.isChecked())
        self.band_hi.setEnabled(not self.auto_band.isChecked())

    def _set_busy(self, busy: bool):
        self._busy = busy
        connected = self._worker is not None
        for w in (self.record_btn, self.slew_btn, self.read_pos_btn):
            w.setEnabled(connected and not busy)
        self.load_btn.setEnabled(not busy)
        self.reprocess_btn.setEnabled(not busy)
        self.save_btn.setEnabled(not busy and self._last_rec is not None)
        self.stop_btn.setEnabled(busy)

    @pyqtSlot(str)
    def _set_status(self, text: str):
        self.status_label.setText(text)

    @pyqtSlot(float)
    def _set_position(self, z: float):
        self.pos_label.setText(f"{z:.2f}")

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Save folder", self.save_dir.text())
        if d:
            self.save_dir.setText(d)

    # ------------------------------------------------------------------ actions

    def _start_stationary(self):
        if not self._confirm_discard_unsaved("record a new one"):
            return
        try:
            self._daq_settings_validated()
        except ValueError as e:
            QMessageBox.warning(self, "DAQ settings", str(e))
            return
        self._set_busy(True)
        self._req_stationary.emit(self._daq_settings(), float(self.duration.value()), self._base_meta())

    def _start_slew(self):
        if not self._confirm_discard_unsaved("record a new one"):
            return
        try:
            self._daq_settings_validated()
        except ValueError as e:
            QMessageBox.warning(self, "DAQ settings", str(e))
            return
        s = SlewSettings(
            backoff_um=self.backoff.value(),
            range_um=self.scan_range.value(),
            speed_um_s=self.speed.value(),
            reverse=self.reverse.isChecked(),
            return_to_start=self.return_to_start.isChecked(),
        )
        if s.backoff_um >= s.range_um:
            r = QMessageBox.question(
                self, "Plane outside scan?",
                f"Back-off ({s.backoff_um:.0f} um) >= scan range ({s.range_um:.0f} um): "
                "the slew will end before reaching the start position. Continue?")
            if r != QMessageBox.Yes:
                return
        meta = dict(self._base_meta(),
                    expected_fringe_hz=fringe_frequency_hz(s.speed_um_s, self.wavelength.value()))
        self._set_busy(True)
        self._req_slew.emit(self._daq_settings(), s, meta)

    def _daq_settings_validated(self):
        from brillouin_system.devices.ni.ni6008_multi import NI6008Multi
        d = self._daq_settings()
        if not all(d.channels):
            raise ValueError("Both channel names are required (e.g. ai0, ai1).")
        NI6008Multi(channels=d.channels, sample_rate_hz=d.sample_rate_hz,
                    terminal=d.terminal, v_range=d.v_range).validate()

    def _stop(self):
        if self._worker is not None:
            self._worker.request_stop()
            self._set_status("Stopping ...")

    @pyqtSlot(object)
    def _on_finished(self, rec: HomodyneRecording):
        self._last_rec = rec
        self._last_path = None
        self._unsaved = True
        self._set_busy(False)
        self._set_status("Done. Not saved yet (press Save recording).")
        self._show(rec)

    def _confirm_discard_unsaved(self, action: str) -> bool:
        """True if there is nothing unsaved, or the user agrees to discard it."""
        if not self._unsaved:
            return True
        r = QMessageBox.question(
            self, "Unsaved recording",
            f"The last recording has not been saved. Discard it and {action}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def _save_file(self):
        rec = self._last_rec
        if rec is None:
            return
        try:
            stamp = datetime.fromisoformat(rec.meta["created"]).strftime("%Y%m%d_%H%M%S")
        except (KeyError, ValueError):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested = Path(self.save_dir.text()) / f"{stamp}_{rec.mode}.npz"
        path, _ = QFileDialog.getSaveFileName(self, "Save recording", str(suggested), "Recordings (*.npz)")
        if not path:
            return
        try:
            self._last_path = save_recording(rec, path)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self._unsaved = False
        self.save_dir.setText(str(self._last_path.parent))
        self._set_status(f"Saved {self._last_path}")
        self._show(rec)  # refresh the summary header with the file name

    @pyqtSlot(str)
    def _on_failed(self, msg: str):
        self._set_busy(False)
        self._set_status(f"Error: {msg}")
        QMessageBox.warning(self, "Acquisition failed", msg)

    def _load_file(self):
        if not self._confirm_discard_unsaved("load a file"):
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load recording", self.save_dir.text(), "Recordings (*.npz)")
        if not path:
            return
        try:
            rec = load_recording(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"{type(e).__name__}: {e}")
            return
        self._last_rec = rec
        self._last_path = Path(path)
        self._unsaved = False
        self._set_busy(False)
        self._show(rec)

    # ------------------------------------------------------------------ processing + plots

    def _show(self, rec: HomodyneRecording):
        v = np.atleast_2d(rec.values)
        fs = rec.sample_rate_hz
        n = v.shape[-1]
        lines = [f"{self._last_path.name if self._last_path else '(unsaved)'}   mode={rec.mode}   "
                 f"{n} samples x {v.shape[0]} ch @ {fs:.0f} S/s = {n / fs:.2f} s"]

        z = rec.z_um() if rec.mode == "slew" else None
        x, xlabel = (z, "z (um)") if z is not None else (rec.t_s(), "time (s)")

        trace = None
        if n >= 10:
            try:
                band = self._band(fs, rec.meta.get("speed_um_s"))
                trace = process_channels(v, fs, band, dc_cutoff_hz=self.dc_cutoff.value(),
                                         dark_v=np.array([self.dark_h.value(), self.dark_v.value()][:v.shape[0]]))
                lines.append(f"Band {band[0]:.0f}-{band[1]:.0f} Hz, DC low-pass {self.dc_cutoff.value():g} Hz")
            except ValueError as e:
                lines.append(f"Processing skipped: {e}")

        rng = float(rec.meta.get("effective_range_v", 10.0))
        names = ["H", "V"]
        for i in range(v.shape[0]):
            ch = rec.channels[i] if i < len(rec.channels) else f"ch{i}"
            vmax = float(np.max(np.abs(v[i]))) if n else 0.0
            s = (f"{names[i] if i < 2 else ch} ({ch}): DC mean {np.mean(v[i]):.4f} V, "
                 f"raw std {np.std(v[i]) * 1e3:.2f} mV, min/max {np.min(v[i]):.3f}/{np.max(v[i]):.3f} V")
            if trace is not None:
                s += (f", fringe median {np.median(trace.envelope[i]) * 1e3:.2f} mV, "
                      f"max {np.max(trace.envelope[i]) * 1e3:.2f} mV")
            if vmax >= 0.9 * rng:
                s += f"   ** NEAR SATURATION (range ±{rng:g} V) **"
            lines.append(s)
        if v.shape[0] >= 2:
            dc = np.mean(v, axis=-1)
            if dc[1] != 0:
                lines.append(f"LO balance DC_H/DC_V = {dc[0] / dc[1]:.3f}  (adjust paddles toward 1.0)")

        peak = None
        if trace is not None:
            edge = int(2 * fs / max(trace.band_hz[0], 1.0))
            lines.append(f"S: median {np.median(trace.s):.4f}, max {np.max(trace.s):.4f} sqrt(V)")
            if rec.mode == "slew":
                peak = locate_peak(x, trace.s, edge_samples=edge)
                if peak.found:
                    lines.append(f"Surface peak at {xlabel.split()[0]} = {peak.x_peak:.2f} "
                                 f"(SNR {peak.snr:.1f}, S {peak.s_peak:.4f})")
                else:
                    lines.append(f"No surface peak (SNR {peak.snr:.1f} < threshold)")
        self.summary.setPlainText("\n".join(lines))

        # plots (decimated for display only)
        step = max(1, n // 200_000)
        xs = x[::step]
        self.fig.clear()
        axs = self.fig.subplots(3, 1, sharex=True)
        for i in range(v.shape[0]):
            axs[0].plot(xs, v[i, ::step], lw=0.5, label=names[i] if i < 2 else f"ch{i}")
        axs[0].set_ylabel("raw (V)")
        axs[0].legend(loc="upper right", fontsize=7)
        if trace is not None:
            for i in range(v.shape[0]):
                axs[1].plot(xs, trace.envelope[i, ::step] * 1e3, lw=0.7, label=f"A_{names[i] if i < 2 else i}")
            axs[1].set_ylabel("fringe A (mV)")
            axs[1].legend(loc="upper right", fontsize=7)
            axs[2].plot(xs, trace.s[::step], lw=0.8, color="k")
            if peak is not None and peak.found:
                axs[2].axvline(peak.x_peak, color="r", lw=0.8, ls="--")
        axs[2].set_ylabel("S (sqrt V)")
        axs[2].set_xlabel(xlabel)
        axs[0].set_title(f"{rec.mode} — {rec.meta.get('created', '')}", fontsize=9)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ shutdown

    def closeEvent(self, event):
        if self._busy:
            r = QMessageBox.question(self, "Busy", "An acquisition is running. Stop it and close?")
            if r != QMessageBox.Yes:
                event.ignore()
                return
            self._stop()
        elif not self._confirm_discard_unsaved("close"):
            event.ignore()
            return
        self._thread.quit()
        if not self._thread.wait(15000):
            QMessageBox.warning(self, "Busy", "Worker did not finish; closing anyway.")
        if self.zaber is not None:
            try:
                self.zaber.close()     # closes the serial port only; never moves
            except Exception:
                pass
        if self.shutters is not None:
            try:
                self.shutters.close_all()      # like the main GUI: no light left on the sample
                self.shutters.shutdown_all()
            except Exception:
                pass
        event.accept()
