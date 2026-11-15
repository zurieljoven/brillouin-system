"""
Andor live view and stage-Z motion for the homodyne plane test GUI (borrowed from the z-carrier test).

connect_andor      the iXon with the human interface's camera settings (or SimAndor for --dummy)
stage_z_mover      the rig stage's Z axis as an AxisMover (slews at a set speed, position log), or a dummy
andor_sweep        Andor frames stepping one axis through its current z (approached from below), then back
AndorView          image, spectrum (the human interface's row band) and summed intensity against z
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from brillouin_system.guis.plane_accuracy_test.axial_recording import AxialRecording
from brillouin_system.guis.z_carrier_test.motion import AxisMover, ZaberAxisMover

PRELOAD_UM = 30.0           # sweeps and moves approach from below, so backlash is taken up the same way
FRAME_SETTLE_S = 0.05
SATURATION_COUNTS = 60000.0  # the iXon digitizes to 16 bits
MAX_SUM_POINTS = 2000


class SweepCancelled(Exception):
    pass


def connect_andor(use_dummy: bool, lens=None):
    """The iXon with the human interface's settings. Dummy: brightness follows the (simulated) lens z."""
    if use_dummy:
        from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
        return SimAndor(lens, plane_um=8000.0)
    from brillouin_system.devices.cameras.andor.andor_frame.andor_config import andor_frame_config
    from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
    cfg = andor_frame_config.get()
    cam = IxonUltra(index=0, temperature="off", fan_mode="full", x_start=cfg.x_start, x_end=cfg.x_end,
                    y_start=cfg.y_start, y_end=cfg.y_end, vbin=cfg.vbin, hbin=cfg.hbin,
                    advanced_gain_option=cfg.advanced_gain_option, verbose=False)
    cam.set_from_config_file(cfg)
    return cam


class DummyStageZMover(AxisMover):
    """ZaberHumanInterfaceDummy's Z with a constant-speed slew (the dummy itself jumps)."""

    def __init__(self, stage):
        super().__init__("stage")
        self.stage = stage
        self._slew = None           # (t0, z0, target, speed)

    def _now_z(self) -> float:
        if self._slew is None:
            return float(self.stage.get_position()[2])
        t0, z0, zt, v = self._slew
        d = zt - z0
        z = z0 + np.sign(d) * min(abs(d), v * (time.perf_counter() - t0))
        if z == zt:
            self.stage.move_abs(z=zt)
            self._slew = None
        return float(z)

    def _get(self) -> float:
        return self._now_z()

    def _move_abs(self, z_um: float) -> None:
        self._slew = None
        self.stage.move_abs(z=z_um)

    def _slew_to(self, z_um: float, speed_um_s: float) -> None:
        self._slew = (time.perf_counter(), float(self.stage.get_position()[2]), z_um, speed_um_s)

    def _wait_idle(self, timeout_s: float) -> None:
        t_end = time.perf_counter() + timeout_s
        while self._slew is not None and time.perf_counter() < t_end:
            self._now_z()
            time.sleep(0.01)

    def stop(self) -> None:
        if self._slew is not None:
            self.stage.move_abs(z=self._now_z())
            self._slew = None


def stage_z_mover(stage) -> Optional[AxisMover]:
    if stage is None:
        return None
    if hasattr(stage, "z_axis"):
        return ZaberAxisMover("stage", stage.z_axis)
    return DummyStageZMover(stage)


def andor_sweep(camera, get_z: Callable[[], float], move_to: Callable[[float], None], half_um: float, step_um: float,
                stop: threading.Event, on_frame: Optional[Callable[[np.ndarray, float], None]] = None,
                meta: Optional[dict] = None) -> AxialRecording:
    """Frames from z0 - half to z0 + half (increasing, approached from below), then back to z0 from below."""
    z0 = float(get_z())
    n = int(round(2.0 * half_um / step_um)) + 1
    targets = z0 - half_um + step_um * np.arange(n)
    frames, zs, ts = [], [], []
    try:
        move_to(float(targets[0]) - PRELOAD_UM)
        for z in targets:
            if stop.is_set():
                raise SweepCancelled()
            move_to(float(z))
            time.sleep(FRAME_SETTLE_S)
            zs.append(float(get_z()))
            ts.append(time.perf_counter())
            frames.append(np.asarray(camera.snap()))
            if on_frame is not None:
                on_frame(frames[-1], zs[-1])
    finally:
        move_to(z0 - PRELOAD_UM)
        move_to(z0)
    meta = dict(meta or {}, z0_um=z0, half_um=half_um, step_um=step_um,
                exposure_s=float(camera.get_exposure_time()), gain=float(camera.get_emccd_gain()),
                roi=[int(v) for v in camera.get_roi()], binning=[int(v) for v in camera.get_binning()])
    return AxialRecording(frames=np.stack(frames), z_um=np.array(zs), t_perf=np.array(ts), meta=meta)


class AndorView(QWidget):
    """Image and spectrum as in the human interface, plus summed intensity against the chosen axis' z."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter
        self._fitter_cls = SpectrumFitter
        self._fitter = SpectrumFitter()
        self.axis = "lens"
        self.z: list[float] = []
        self.v: list[float] = []
        self._current: Optional[tuple[float, float]] = None

        pg.setConfigOptions(useOpenGL=False, antialias=True, imageAxisOrder="row-major", background="w", foreground="k")
        glw = pg.GraphicsLayoutWidget()
        glw.ci.setSpacing(12)
        self.vb_img = glw.addViewBox(lockAspect=True, enableMenu=False)
        self.vb_img.invertY(True)
        self.img = pg.ImageItem(autoDownsample=True)
        self.vb_img.addItem(self.img)
        self.vb_img.setBorder((170, 170, 170))
        glw.nextRow()
        self.spec_plot = glw.addPlot(labels={"left": "Intensity", "bottom": "Pixel (X)"})
        self.spec_plot.setMenuEnabled(False)
        self.spec_pts = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4, symbolPen=pg.mkPen("k"), symbolBrush="k")
        self.spec_plot.addItem(self.spec_pts)
        glw.nextRow()
        self.sum_plot = glw.addPlot(labels={"left": "Summed intensity", "bottom": "Lens z (um)"})
        self.sum_plot.setMenuEnabled(False)
        self.sum_pts = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5, symbolPen=pg.mkPen("k"), symbolBrush="k")
        self.sum_best = pg.PlotDataItem(pen=None, symbol="o", symbolSize=10, symbolPen=pg.mkPen("r"), symbolBrush="r")
        self.sum_plot.addItem(self.sum_pts)
        self.sum_plot.addItem(self.sum_best)
        self.label = QLabel("Summed intensity: -")
        self.label.setStyleSheet("font-family: monospace;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(glw, stretch=1)
        lay.addWidget(self.label)

    def set_axis(self, axis: str) -> None:
        if axis != self.axis:
            self.axis = axis
            self.clear()
            self.sum_plot.setLabel("bottom", f"{'Lens' if axis == 'lens' else 'Stage'} z (um)")

    def clear(self) -> None:
        self.z, self.v = [], []
        self._current = None
        self._fitter = self._fitter_cls()       # its automatic row band may have frozen on a dark frame
        self._redraw()

    def brightest(self) -> Optional[float]:
        return self.z[int(np.argmax(self.v))] if self.v else None

    def add_frame(self, frame: np.ndarray, z: float) -> None:
        frame = np.ascontiguousarray(frame).astype(np.float32, copy=False)
        fmin, fmax = float(np.nanmin(frame)), float(np.nanmax(frame))
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            fmin, fmax = 0.0, 1.0
        self.img.setImage(frame, autoLevels=False, levels=(fmin, fmax))
        px, sline = self._fitter.get_px_sline_from_image(frame)
        self.spec_pts.setData(px, sline)
        total = float(self._fitter.get_total_sline_value(sline))
        flag = "   NEAR SATURATION" if fmax >= SATURATION_COUNTS else ""
        self.spec_plot.setTitle(f"Spectrum at {self.axis} z = {z:.1f} um, frame max {fmax:.0f} counts{flag}",
                                color=(192, 57, 43) if flag else (40, 40, 40))
        self._current = (z, total)
        self.z.append(float(z))
        self.v.append(total)
        del self.z[:-MAX_SUM_POINTS], self.v[:-MAX_SUM_POINTS]
        self._redraw()

    def _redraw(self) -> None:
        self.sum_pts.setData(self.z, self.v)
        b = self.brightest()
        self.sum_best.setData([b] if b is not None else [], [max(self.v)] if self.v else [])
        text = "Summed intensity: -" if self._current is None else \
            f"Summed intensity {self._current[1]:.0f} at {self.axis} z = {self._current[0]:.1f} um"
        if b is not None:
            text += f"     brightest {max(self.v):.0f} at {b:.1f} um"
        self.label.setText(text)
