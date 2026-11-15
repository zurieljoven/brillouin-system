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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy.signal import welch
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from brillouin_system.guis.plane_accuracy_test.runner import open_sample_path
from brillouin_system.guis.z_carrier_test.motion import SimAxisMover, ZaberAxisMover
from brillouin_system.guis.z_carrier_test.protocol import PRE_ROLL_S, PRELOAD_UM, ProtocolConfig, alias_hz, channels_for
from brillouin_system.guis.z_carrier_test.geometry import SphereFit, fit_sphere, scan_locations, shuffled
from brillouin_system.guis.z_carrier_test.runner import (
    Cancelled, Location, Rig, andor_sweep, approach, check_locations, finder_params, park_lens_on_plane,
    record_stationary,
    run_locations, run_protocol,
)
from brillouin_system.scan_managers.homodyne_recording import save_recording
from brillouin_system.scan_managers.homodyne_plane_finder import zero_phase_s
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

PARK_HALF_UM = 150.0
PARK_SPEED_UM_S = 400.0
SATURATION_COUNTS = 60000.0     # the iXon digitizes to 16 bits
MAX_SUM_POINTS = 2000
MAX_TO_BRIGHTEST_UM = 200.0
DC_READ_S = 0.2                 # one live DC reading
DC_HISTORY_S = 60.0
DC_WARN_V = 8.5


def connect_rig(use_dummy: bool, lens_port: str, stage_port: str, ni_device: str) -> Rig:
    if use_dummy:
        from brillouin_system.devices.shutter_device import ShutterManagerDummy
        from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import \
            ZaberHumanInterfaceDummy
        from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
        from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
        from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
        lens, stage_z = SimZaberLens(start_um=8030.0), SimZaberLens(start_um=12000.0, move_speed_um_s=3000.0)
        xy = ZaberHumanInterfaceDummy(home_on_connect=False)
        xy.move_abs(x=10000.0, y=10000.0)
        daq = SimHomodyneDAQ(lens, plane_um=8000.0, peak_amplitude_v=0.2, noise_std_v=0.006)
        return Rig(daq, SimAxisMover("lens", lens), SimAxisMover("stage", stage_z), SimAndor(lens, plane_um=8000.0),
                   ShutterManagerDummy("human_interface"), jog_lens=lens.move_rel, jog_stage=stage_z.move_rel,
                   xy_get=lambda: xy.get_position()[:2], xy_move=lambda x, y: xy.move_abs(x=x, y=y))

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
               jog_stage=lambda dz: stage.move_rel(dz=dz), xy_get=lambda: stage.get_position()[:2],
               xy_move=lambda x, y: stage.move_abs(x=x, y=y), devices=(camera, lens, stage))


class Worker(QObject):
    status = pyqtSignal(str)
    positions = pyqtSignal(float, float, float, float)     # lens z, stage x, y, z [um]
    connected = pyqtSignal(str)                 # error text, empty when fine
    result = pyqtSignal(object, object)         # log row (or None), recording
    frame = pyqtSignal(object, str, float, bool)    # Andor frame (None on error), axis, its z [um], live view
    dc = pyqtSignal(object, object)                 # per-channel mean and std [V] (None on error)
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, args: tuple):
        super().__init__()
        self._args = args
        self.rig: Optional[Rig] = None
        self.stop = threading.Event()

    def _emit_positions(self):
        x, y = self.rig.xy_get()
        self.positions.emit(self.rig.lens.get_position(), float(x), float(y), self.rig.stage.get_position())

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
        def go():
            if axis in ("x", "y"):
                x, y = self.rig.xy_get()
                self.rig.xy_move(x + (dz if axis == "x" else 0.0), y + (dz if axis == "y" else 0.0))
            else:
                (self.rig.jog_lens if axis == "lens" else self.rig.jog_stage)(dz)
        self._guarded(go)

    @pyqtSlot(bool)
    def shutter(self, is_open: bool):
        self._guarded(lambda: open_sample_path(self.rig.shutters) if is_open else self.rig.shutters.close_all())

    @pyqtSlot()
    def dc_read(self):
        """One short H + V record at 5 kS/s: DC level and fluctuation per channel (for balancing the LO)."""
        try:
            self.rig.daq.configure(sample_rate_hz=5000.0, channels=channels_for(5000.0))
            v = self.rig.daq.acquire(DC_READ_S).values
            self.dc.emit([float(x.mean()) for x in v], [float(x.std()) for x in v])
        except Exception as e:
            self.failed.emit(f"DC read: {type(e).__name__}: {e}")
            self.dc.emit(None, None)

    @pyqtSlot(float, int, bool)
    def snap(self, exposure_s: float, gain: int, live: bool):
        try:
            self.rig.camera.set_exposure_time(exposure_s)
            self.rig.camera.set_emccd_gain(gain)
            f = np.asarray(self.rig.camera.snap())
            z = self.rig.lens.get_position()
            self._emit_positions()
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

    @pyqtSlot(float, float, object, object)
    def stationary(self, fs: float, duration_s: float, path: Path, meta: dict):
        """Record with nothing moving at sample rate fs and save it. The protocol and park set 5 kS/s again."""
        def go():
            self.stop.clear()
            self.rig.daq.configure(sample_rate_hz=fs, channels=channels_for(fs))
            x, y = self.rig.xy_get()
            m = dict(meta, sample_rate_hz=fs, lens_um=self.rig.lens.get_position(), stage_x_um=float(x),
                     stage_y_um=float(y), stage_z_um=self.rig.stage.get_position())
            rec = record_stationary(self.rig, duration_s, m, self.stop)
            save_recording(rec, path)
            dc = " ".join(f"{np.median(v):.3f}" for v in rec.values)
            self.status.emit(f"Saved {path.name}: {rec.values.shape[-1] / fs:.1f} s at {fs:.0f} S/s, DC {dc} V")
            self.result.emit(None, rec)
        self._guarded(go)

    @pyqtSlot()
    def park(self):
        def go():
            self.stop.clear()
            self.rig.daq.configure(sample_rate_hz=5000.0, channels=channels_for(5000.0))
            z, rec = park_lens_on_plane(self.rig, PARK_HALF_UM, PARK_SPEED_UM_S, self.stop)
            self.status.emit(f"Lens parked on the homodyne plane at {z:.1f} um." if z is not None
                             else "No clear plane within the slew; lens left where it was.")
            self.result.emit(None, rec)
        self._guarded(go)

    @pyqtSlot(object, object, object, bool)
    def run(self, cfg: ProtocolConfig, run_dir: Path, locations, recentre: bool):
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

            if locations:
                c = run_locations(self.rig, cfg, locations, run_dir, self.stop, recentre=recentre,
                                  on_status=on_status, on_result=on_result, on_frame=on_frame)
                self.status.emit(f"{c['n_locations']} locations done.")
            else:
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
        for m in (self.rig.lens, self.rig.stage):
            try:
                if hasattr(m, "restore_speed"):
                    m.restore_speed()
            except Exception:
                pass
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
    _req_dc = pyqtSignal()
    _req_sweep = pyqtSignal(float, int, float, float)
    _req_lens_to = pyqtSignal(float)
    _req_park = pyqtSignal()
    _req_stationary = pyqtSignal(float, float, object, object)
    _req_run = pyqtSignal(object, object, object, bool)
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
        self._pos: Optional[tuple] = None           # lens, stage x, y, z [um]
        self._dc_pending = False
        self._dc_t, self._dc_h, self._dc_v = [], [], []
        self._dc_t0: Optional[float] = None
        self._center: Optional[Location] = None
        self._edge: Optional[Location] = None
        self._mids: list = []
        self._fit: Optional[SphereFit] = None
        self._build_ui(save_dir or Path.home() / "Documents" / "z_carrier_runs")

        self._thread = QThread(self)
        self._worker = Worker((use_dummy, lens_port, stage_port, ni_device))
        self._worker.moveToThread(self._thread)
        self._worker.status.connect(self._log)
        self._worker.positions.connect(self._on_positions)
        self._worker.connected.connect(self._on_connected)
        self._worker.result.connect(self._on_result)
        self._worker.frame.connect(self._on_frame)
        self._worker.dc.connect(self._on_dc)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._req_connect.connect(self._worker.connect_all)
        self._req_jog.connect(self._worker.jog)
        self._req_shutter.connect(self._worker.shutter)
        self._req_snap.connect(self._worker.snap)
        self._req_dc.connect(self._worker.dc_read)
        self._req_sweep.connect(self._worker.sweep_here)
        self._req_lens_to.connect(self._worker.lens_to)
        self._req_park.connect(self._worker.park)
        self._req_stationary.connect(self._worker.stationary)
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
        f.addRow("Stage x / y / z", self.stage_label)
        self.jog_step = _dspin(0.5, 500, 10, 5, 1, " um")
        f.addRow("Lens / stage Z jog", self.jog_step)
        for axis in ("lens", "stage"):
            row = QHBoxLayout()
            for sign, text in ((-1, "-"), (1, "+")):
                b = QPushButton(f"{'Lens' if axis == 'lens' else 'Stage Z'} {text}")
                b.clicked.connect(lambda _, a=axis, s=sign: self._jog(a, s))
                row.addWidget(b)
            f.addRow(row)
        self.xy_step = _dspin(1, 2000, 250, 50, 0, " um")
        f.addRow("Stage X/Y jog", self.xy_step)
        row = QHBoxLayout()
        for axis, sign, text in (("x", -1, "X -"), ("x", 1, "X +"), ("y", -1, "Y -"), ("y", 1, "Y +")):
            b = QPushButton(text)
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

        dcg = QGroupBox("H / V monitor (balance the LO with the half-wave plate)")
        f = QVBoxLayout(dcg)
        row = QHBoxLayout()
        self.b_dc = QPushButton("Live H / V")
        self.b_dc.setCheckable(True)
        self.b_dc.toggled.connect(self._toggle_dc)
        b = QPushButton("Clear")
        b.clicked.connect(self._clear_dc)
        row.addWidget(self.b_dc)
        row.addWidget(b)
        f.addLayout(row)
        self.dc_label = QLabel("H -   V -")
        self.dc_label.setStyleSheet("font-family: monospace; font-size: 13pt; font-weight: bold;")
        f.addWidget(self.dc_label)
        self.dc_detail = QLabel("Target: H and V equal, each 4-6 V, below 8 V.")
        self.dc_detail.setStyleSheet("font-family: monospace;")
        f.addWidget(self.dc_detail)
        self.dc_plot = pg.PlotWidget(background="w")
        self.dc_plot.setMinimumHeight(130)
        self.dc_plot.setMaximumHeight(170)
        self.dc_plot.setLabel("left", "DC (V)")
        self.dc_plot.setLabel("bottom", "time (s)")
        self.dc_plot.addLegend(offset=(5, 5))
        self.dc_curve_h = self.dc_plot.plot(pen=pg.mkPen((31, 119, 180), width=2), name="H (ai0)")
        self.dc_curve_v = self.dc_plot.plot(pen=pg.mkPen((255, 127, 14), width=2), name="V (ai1)")
        self.dc_plot.addLine(y=DC_WARN_V, pen=pg.mkPen((192, 57, 43), style=2))
        f.addWidget(self.dc_plot)
        left.addWidget(dcg)

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

        rs = QGroupBox("Rig settings (written into every file; nothing is changed)")
        f = QFormLayout(rs)
        self.pd_gain = QSpinBox()
        self.pd_gain.setRange(0, 70)
        self.pd_gain.setSingleStep(10)
        self.pd_gain.setValue(70)
        self.pd_gain.setSuffix(" dB")
        self.pd_gain.setToolTip("The photodiode gain switch setting. Only written into the files, nothing is changed.")
        self.fp_mod = _dspin(1.0, 200000.0, 3396.5, 100.0, 1, " Hz")
        self.fp_mod.setToolTip("The FP lock modulation frequency. Used for the alias markers and written into the files.")
        f.addRow("PD gain", self.pd_gain)
        f.addRow("FP modulation", self.fp_mod)
        left.addWidget(rs)

        st = QGroupBox("Stationary record (saved; nothing moves)")
        f = QFormLayout(st)
        self.st_fs = QSpinBox()
        self.st_fs.setRange(1000, 10000)
        self.st_fs.setSingleStep(500)
        self.st_fs.setValue(5000)
        self.st_fs.setSuffix(" S/s per ch")
        self.st_fs.setToolTip("USB-6008: 10 kS/s in total. Up to 5000 per channel both H and V are recorded; above, H only.")
        self.st_dur = _dspin(0.5, 120, 10, 1, 1, " s")
        self.st_fs.valueChanged.connect(self._update_alias)
        self.fp_mod.valueChanged.connect(self._update_alias)
        self.alias_label = QLabel("")
        self.alias_label.setWordWrap(True)
        f.addRow("Sample rate", self.st_fs)
        f.addRow("Duration", self.st_dur)
        f.addRow(self.alias_label)
        self.b_stat = QPushButton("Record stationary")
        self.b_stat.clicked.connect(self._record_stationary)
        f.addRow(self.b_stat)
        left.addWidget(st)
        self._update_alias()

        geo = QGroupBox("Sample geometry (leave empty for one spot, e.g. glass)")
        f = QFormLayout(geo)
        hint = QLabel("Lens on the plane (Andor) at each point, stage Z fixed. 3+ points fit the apex.")
        hint.setWordWrap(True)
        f.addRow(hint)
        row = QHBoxLayout()
        for text, which in (("Set center", "center"), ("Add intermediate", "mid"), ("Set edge", "edge")):
            b = QPushButton(text)
            b.clicked.connect(lambda _, w=which: self._set_point(w))
            row.addWidget(b)
        f.addRow(row)
        row = QHBoxLayout()
        b = QPushButton("Remove last intermediate")
        b.clicked.connect(self._remove_mid)
        row.addWidget(b)
        b = QPushButton("Clear points")
        b.clicked.connect(self._clear_points)
        row.addWidget(b)
        f.addRow(row)
        self.r_step = _dspin(0.05, 5, 0.5, 0.25, 2, " mm")
        self.r_step.valueChanged.connect(self._update_geometry)
        f.addRow("Radius step", self.r_step)
        self.random_order = QCheckBox("Random location order (center first)")
        self.random_order.setChecked(True)
        self.recentre = QCheckBox("Re-centre the lens on the Andor peak at each location")
        self.recentre.setChecked(True)
        f.addRow(self.random_order)
        f.addRow(self.recentre)
        self.geom_label = QLabel("No points set: Run works at the current spot only.")
        self.geom_label.setWordWrap(True)
        self.geom_label.setStyleSheet("font-family: monospace;")
        f.addRow(self.geom_label)
        left.addWidget(geo)

        cfg = QGroupBox("Protocol")
        f = QFormLayout(cfg)
        self.sample = QComboBox()
        self.sample.setEditable(True)
        self.sample.addItems(["glass", "phantom"])
        self.blocks = QSpinBox()
        self.blocks.setRange(1, 50)
        self.blocks.setValue(3)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(int(datetime.now().strftime("%H%M%S")))
        self.stationary = _dspin(1, 60, 10, 1, 1, " s")
        self.slew_half = _dspin(50, 400, 200, 10, 0, " um")
        d = ProtocolConfig()
        self.speeds = QLineEdit(", ".join(f"{v:.0f}" for v in d.speeds_um_s))
        self.rates = QLineEdit(", ".join(f"{v:.0f}" for v in d.sample_rates_hz))
        self.rates.setToolTip("DAQ sample rates per channel, swept within every block (stationary and slews). "
                              "Above 5000 only H (ai0) is recorded.")
        self.andor_half = _dspin(4, 100, 40, 5, 0, " um")
        self.andor_step = _dspin(0.5, 20, 2, 0.5, 1, " um")
        self.save_dir = QLineEdit(str(save_dir))
        for label, w in (("Sample", self.sample), ("Blocks", self.blocks), ("Seed", self.seed),
                         ("Stationary", self.stationary), ("Slew +-", self.slew_half), ("Speeds (um/s)", self.speeds),
                         ("Sample rates (S/s)", self.rates),
                         ("Andor sweep +-", self.andor_half), ("Andor step", self.andor_step), ("Save to", self.save_dir)):
            f.addRow(label, w)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._update_info)
        self.exposure.valueChanged.connect(self._update_info)
        self.speeds.textChanged.connect(self._update_info)
        self.rates.textChanged.connect(self._update_info)
        self.rates.textChanged.connect(self._update_alias)
        self.fp_mod.valueChanged.connect(self._update_info)
        self.pd_gain.valueChanged.connect(self._update_info)
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
        self._controls = [b for b in panel.findChildren(QPushButton) if b not in (self.b_stop, self.b_live, self.b_dc)]
        self._update_info()
        self._update_alias()

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

    def _on_positions(self, lens: float, x: float, y: float, z: float):
        self._pos = (lens, x, y, z)
        self.lens_label.setText(f"{lens:.1f} um")
        self.stage_label.setText(f"{x:.1f} / {y:.1f} / {z:.1f} um")

    # ------------------------------------------------------------------ sample geometry

    def _set_point(self, which: str):
        if self._pos is None:
            return
        lens, x, y, z = self._pos
        if which == "mid":
            name = f"p{len(self._mids) + 1}"
            while name in {m.name for m in self._mids}:
                name += "b"
            self._mids.append(Location(name, x, y, z, lens))
        else:
            setattr(self, f"_{which}", Location(which, x, y, z, lens))
        self._log(f"Set {which if which != 'mid' else self._mids[-1].name}: stage {x:.1f} / {y:.1f} / {z:.1f} um, "
                  f"lens {lens:.1f} um")
        self._update_geometry()

    def _remove_mid(self):
        if self._mids:
            self._mids.pop()
            self._update_geometry()

    def _clear_points(self):
        self._center = self._edge = None
        self._mids = []
        self._update_geometry()

    def _scan_plan(self) -> Optional[list]:
        """Scan locations from the fit, or None when center and edge are not both set. Raises ValueError."""
        if self._center is None or self._edge is None:
            return None
        fit = fit_sphere(self._center, self._edge, self._mids)
        locs = scan_locations(fit, self.r_step.value())
        check_locations(locs)
        self._fit = fit
        return locs

    def _update_geometry(self, *_):
        lines = []
        for p in [self._center, *self._mids, self._edge]:
            if p is not None:
                r = f"R {p.r_mm(self._center):.2f} mm, " if self._center is not None else ""
                lines.append(f"{p.name:>6}: {r}lens {p.lens_um:.1f} um")
        try:
            locs = self._scan_plan()
            if locs is not None:
                f = self._fit
                lines.append(f.summary())
                if np.isfinite(f.rms_um) and f.rms_um > 5.0:
                    lines.append(f"WARNING: points sit {f.rms_um:.0f} um rms off the sphere; check them.")
                for p in [self._center, *self._mids, self._edge]:
                    lines.append(f"  {p.name} fit residual {p.lens_um - f.lens_at(p.stage_x_um, p.stage_y_um):+.1f} um")
                lines.append(f"{len(locs)} locations:")
                for l in locs:
                    lines.append(f"  {l.name}: lens {l.lens_um:.1f} um, tilt {f.tilt_deg(l.stage_x_um, l.stage_y_um):.0f} deg")
            self.geom_label.setStyleSheet("font-family: monospace;")
        except ValueError as e:
            lines.append(str(e))
            self.geom_label.setStyleSheet("font-family: monospace; color: #c0392b;")
        self.geom_label.setText("\n".join(lines) if lines else "No points set: Run works at the current spot only.")
        self._update_info()

    def _on_failed(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.warning(self, "Error", msg)

    @staticmethod
    def _numbers(text: str) -> tuple:
        try:
            return tuple(float(s) for s in text.replace(";", ",").split(",") if s.strip())
        except ValueError:
            raise ValueError(f"Not a list of numbers: {text!r}")

    def _config(self) -> ProtocolConfig:
        speeds = self._numbers(self.speeds.text())
        return ProtocolConfig(sample=self.sample.currentText().strip() or "sample", n_blocks=self.blocks.value(),
                              seed=self.seed.value(), stationary_s=self.stationary.value(),
                              slew_half_um=self.slew_half.value(), speeds_um_s=speeds,
                              andor_half_um=self.andor_half.value(), andor_step_um=self.andor_step.value(),
                              exposure_s=self.exposure.value(), gain=self.gain.value(),
                              pd_gain_db=self.pd_gain.value(), fp_mod_hz=self.fp_mod.value(),
                              sample_rates_hz=self._numbers(self.rates.text()))

    def _update_info(self, *_):
        try:
            c = self._config()
            c.validate()
            n_loc = self._n_locations()
            where = f" per location; {n_loc} locations, about {n_loc * (c.estimated_s() + 25.0) / 60:.0f} min in all" \
                if n_loc else ""
            skip = c.skipped()
            skipped = (" Skipped (stage-Z fringe above Nyquist): " + ", ".join(f"{v:.0f} um/s at {fs:.0f}"
                                                                                for v, fs in skip) + ".") if skip else ""
            self.info.setText(f"{len(c.steps())} steps, about {c.estimated_s() / 60:.1f} min{where}. Each block, random "
                              f"order: {len(c.sample_rates_hz)} stationary records (one per sample rate) + "
                              f"{4 * len(c.combos())} slews (lens and stage, fwd and bwd, every speed x rate).{skipped} "
                              f"Andor: {c.andor_frames()} frames per sweep, 4 sweeps.")
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

    # ------------------------------------------------------------------ H / V monitor

    def _toggle_dc(self, on: bool):
        if on and not self._dc_pending and not self._running:
            self._dc_pending = True
            self._req_dc.emit()
        elif on and self._running:
            self.b_dc.setChecked(False)

    def _clear_dc(self):
        self._dc_t, self._dc_h, self._dc_v = [], [], []
        self._dc_t0 = None
        self.dc_curve_h.setData([], [])
        self.dc_curve_v.setData([], [])

    def _on_dc(self, means, stds):
        self._dc_pending = False
        if means is None:
            self.b_dc.setChecked(False)
            return
        now = time.perf_counter()
        if self._dc_t0 is None:
            self._dc_t0 = now
        h, v = means[0], means[1] if len(means) > 1 else float("nan")
        self._dc_t.append(now - self._dc_t0)
        self._dc_h.append(h)
        self._dc_v.append(v)
        while self._dc_t and self._dc_t[-1] - self._dc_t[0] > DC_HISTORY_S:
            del self._dc_t[0], self._dc_h[0], self._dc_v[0]
        self.dc_curve_h.setData(self._dc_t, self._dc_h)
        self.dc_curve_v.setData(self._dc_t, self._dc_v)
        ratio = v / h if h > 1e-3 else float("nan")
        warn = max(h, v) > DC_WARN_V
        self.dc_label.setText(f"H {h:5.2f} V   V {v:5.2f} V   V/H {ratio:4.2f}")
        self.dc_label.setStyleSheet("font-family: monospace; font-size: 13pt; font-weight: bold;"
                                    + (" color: #c0392b;" if warn else ""))
        sd = "  ".join(f"{s * 1e3:.1f}" for s in stds)
        self.dc_detail.setText(f"sum {h + v:.2f} V, difference {v - h:+.2f} V; fluctuation (std) {sd} mV"
                               + ("   NEAR SATURATION: lower the LO" if warn else ""))
        if self.b_dc.isChecked() and not self._running:
            self._dc_pending = True
            self._req_dc.emit()

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
        self.b_dc.setChecked(False)
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
        if axis != "lens":
            self._clear_sum()       # another spot, or the surface moved relative to the focus
        self._set_busy(True)
        step = self.xy_step.value() if axis in ("x", "y") else self.jog_step.value()
        self._req_jog.emit(axis, sign * step)

    def _shutter(self, is_open: bool):
        self._set_busy(True)
        self._req_shutter.emit(is_open)

    def _update_alias(self, *_):
        fm = self.fp_mod.value()
        text = f"FP {fm:.1f} Hz appears at {alias_hz(fm, float(self.st_fs.value())):.1f} Hz here"
        try:
            rates = self._numbers(self.rates.text()) if hasattr(self, "rates") else ()
            text += "; protocol rates: " + ", ".join(f"{alias_hz(fm, fs):.0f} Hz at {fs:.0f}" for fs in rates)
        except ValueError:
            pass
        self.alias_label.setText(text)

    def _record_stationary(self):
        self.b_dc.setChecked(False)
        fs, sample = float(self.st_fs.value()), (self.sample.currentText().strip() or "sample").replace(" ", "_")
        path = (Path(self.save_dir.text()) / "stationary" /
                f"stationary_{sample}_fs{fs:.0f}_pd{self.pd_gain.value()}dB_{datetime.now():%Y%m%d_%H%M%S}.h5")
        fm = self.fp_mod.value()
        meta = dict(sample=sample, kind="stationary", pd_gain_db=self.pd_gain.value(), fp_mod_hz=fm, fs_hz=fs,
                    fp_alias_hz=alias_hz(fm, fs), wavelength_nm=780.24)
        self.b_live.setChecked(False)
        self._set_busy(True)
        self._req_stationary.emit(fs, float(self.st_dur.value()), path, meta)

    def _park(self):
        self.b_dc.setChecked(False)
        self.b_live.setChecked(False)
        self._set_busy(True)
        self._req_park.emit()

    def _stop(self):
        self._worker.stop.set()
        self.b_live.setChecked(False)

    def _n_locations(self) -> int:
        try:
            locs = self._scan_plan()
        except ValueError:
            return 0
        return len(locs) if locs else 0

    def _run(self):
        try:
            cfg = self._config()
            cfg.validate()
            locs = self._scan_plan()
        except ValueError as e:
            QMessageBox.warning(self, "Protocol", str(e))
            return
        if locs and self.random_order.isChecked():
            locs = shuffled(locs, cfg.seed)
        name = f"zcarrier_{cfg.sample.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = Path(self.save_dir.text()) / name
        if locs:
            where = (f"{len(locs)} locations from the center to the edge (order: {', '.join(l.name for l in locs)}), "
                     f"lens z from the sphere fit{', re-centred on the Andor peak at each' if self.recentre.isChecked() else ''}."
                     f"\nThe lens backs off {cfg.slew_half_um + 100:.0f} um before every X/Y move.\n\nAbout "
                     f"{len(locs) * (cfg.estimated_s() + 25.0) / 60:.0f} min.")
        else:
            where = f"At the current spot only: {len(cfg.steps())} steps, about {cfg.estimated_s() / 60:.1f} min."
        r = QMessageBox.question(
            self, "Run",
            f"Sample shutters open and the lens on the plane?\n\nAt each location the lens and stage Z move up to "
            f"{cfg.window_half_um():.0f} um either side of the plane (stage Z moves the whole head, cameras "
            f"included).\n\n{where}\nFolder: {run_dir}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        self.b_live.setChecked(False)
        self.b_dc.setChecked(False)
        self._running = True
        self._set_busy(True)
        self._log(f"Run started: {run_dir}")
        self._req_run.emit(cfg, run_dir, locs, self.recentre.isChecked())

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
        elif getattr(rec, "mode", "") == "stationary":
            fs = float(rec.sample_rate_hz)
            self.ax_s.clear()
            for v, lab in zip(rec.values, ("H (ai0)", "V (ai1)")):
                f, P = welch(v - v.mean(), fs, nperseg=min(v.size, int(fs)))
                self.ax_s.semilogy(f[1:], np.sqrt(P[1:]), lw=0.6, label=lab)
            fa = alias_hz(float(rec.meta.get("fp_mod_hz", self.fp_mod.value())), fs)
            self.ax_s.axvline(fa, color="gray", ls="--", lw=0.8)
            self.ax_s.text(fa, self.ax_s.get_ylim()[1], f" FP alias {fa:.0f}", fontsize=7, va="top", color="gray")
            self.ax_s.legend(fontsize=7)
            self.ax_s.set(xlabel="frequency (Hz)", ylabel="ASD (V/rtHz)", xlim=(0, fs / 2),
                          title=f"Stationary, {fs:.0f} S/s, PD {rec.meta.get('pd_gain_db', '?')} dB")
        elif getattr(rec, "mode", "") == "slew":
            z = rec.z_um()
            if z is not None:
                s = zero_phase_s(rec.values, rec.sample_rate_hz,
                                 finder_params(rec.sample_rate_hz, float(rec.meta.get("speed_um_s", 400.0))))
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
