"""
Scope tab for the homodyne plane test GUI: H and V live against time, and H against V (the ellipse).

Nothing is saved. The DAQ streams with the DAQ settings of the main panel; STOP (or Stop scope) ends it.
Display modes: Raw (volts as read), AC (each channel's window mean removed) or Band-pass (zero-phase Butterworth
over the set band, applied to the shown window each redraw).

Readouts are over the XY window: DC per channel, AC rms per channel, the correlation rho between the AC parts, and
D = arccos(rho), the H/V fringe phase difference. D is only meaningful when the fringe phase sweeps through 2 pi
within the window (vibration or a slew); a still fringe gives a dot or a line, whatever D is. The sign of D is not
recoverable from rho, so |D| is shown.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)
from scipy.signal import butter, sosfiltfilt

MAX_BUFFER_S = 10.0
REDRAW_MS = 50
NAMES = ("H", "V")
COLORS = ("#1f77b4", "#d62728")


def _spin(lo, hi, val, step, decimals, suffix):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
    w.setSuffix(suffix)
    return w


def ellipse_stats(h: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
    """(rms_h, rms_v, rho, |D| deg) of two AC signals."""
    h = h - h.mean()
    v = v - v.mean()
    sh, sv = float(np.sqrt(np.mean(h ** 2))), float(np.sqrt(np.mean(v ** 2)))
    if sh <= 0 or sv <= 0:
        return sh, sv, float("nan"), float("nan")
    rho = float(np.clip(np.mean(h * v) / (sh * sv), -1.0, 1.0))
    return sh, sv, rho, float(np.degrees(np.arccos(rho)))


class ScopeView(QWidget):
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fs = 0.0
        self._names: tuple[str, ...] = ()
        self._buf: Optional[np.ndarray] = None
        self._n = 0          # valid samples in _buf (newest at the end)
        self._dirty = False

        self.start_btn = QPushButton("Start scope")
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn = QPushButton("Stop scope")
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.mode = QComboBox()
        self.mode.addItems(["AC", "Raw", "Band-pass"])
        self.window_s = _spin(0.01, MAX_BUFFER_S, 0.5, 0.1, 2, " s")
        self.xy_s = _spin(0.005, MAX_BUFFER_S, 0.1, 0.05, 3, " s")
        self.band_lo = _spin(1.0, 1e5, 100.0, 50.0, 0, " Hz")
        self.band_hi = _spin(1.0, 1e5, 2400.0, 100.0, 0, " Hz")
        self.hold = QCheckBox("Hold")
        self.autoscale = QCheckBox("Autoscale")
        self.autoscale.setChecked(True)
        self.readout = QLabel("Not running.")
        self.readout.setWordWrap(True)

        row1 = QHBoxLayout()
        for w in (self.start_btn, self.stop_btn, QLabel("Display"), self.mode, self.hold, self.autoscale):
            row1.addWidget(w)
        row1.addStretch()
        row2 = QHBoxLayout()
        for lbl, w in (("Trace window", self.window_s), ("XY window", self.xy_s),
                       ("Band", self.band_lo), ("to", self.band_hi)):
            row2.addWidget(QLabel(lbl))
            row2.addWidget(w)
        row2.addStretch()

        self.fig = Figure(figsize=(9, 8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        gs = self.fig.add_gridspec(2, 2, height_ratios=(1, 1.3))
        self.ax_h = self.fig.add_subplot(gs[0, 0])
        self.ax_v = self.fig.add_subplot(gs[0, 1])
        self.ax_xy = self.fig.add_subplot(gs[1, :])
        self.line_h, = self.ax_h.plot([], [], lw=0.8, color=COLORS[0])
        self.line_v, = self.ax_v.plot([], [], lw=0.8, color=COLORS[1])
        self.line_xy, = self.ax_xy.plot([], [], ".", ms=1.5, color="0.2")
        for ax, n in ((self.ax_h, "H"), (self.ax_v, "V")):
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"{n} (V)")
            ax.set_title(n)
        self.ax_xy.set_xlabel("H (V)")
        self.ax_xy.set_ylabel("V (V)")
        self.ax_xy.set_aspect("equal", adjustable="datalim")
        self.ax_xy.set_title("H vs V")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, stretch=1)
        lay.addWidget(self.readout)

        self._timer = QTimer(self)
        self._timer.setInterval(REDRAW_MS)
        self._timer.timeout.connect(self._redraw)
        self.set_state(can_start=False, running=False)

    # ---- control ---------------------------------------------------------------------------------------------

    def set_state(self, can_start: bool, running: bool) -> None:
        self.start_btn.setEnabled(can_start and not running)
        self.stop_btn.setEnabled(running)

    def begin(self, fs: float, names: tuple[str, ...]) -> None:
        self._fs = float(fs)
        self._names = tuple(names)
        self._buf = np.zeros((len(names), int(MAX_BUFFER_S * fs)))
        self._n = 0
        self._timer.start()
        self.readout.setText("Running ...")

    def end(self, msg: str = "") -> None:
        self._timer.stop()
        self._redraw(force=True)
        self.readout.setText((self.readout.text() + "\n" if self._n else "") + (msg or "Stopped."))

    @pyqtSlot(object)
    def add_block(self, block) -> None:
        if self._buf is None:
            return
        b = np.atleast_2d(np.asarray(block, dtype=np.float64))
        m = b.shape[1]
        cap = self._buf.shape[1]
        if m >= cap:
            self._buf[:] = b[:, -cap:]
        else:
            self._buf[:, :-m] = self._buf[:, m:]
            self._buf[:, -m:] = b
        self._n = min(cap, self._n + m)
        self._dirty = True

    # ---- drawing ---------------------------------------------------------------------------------------------

    def _process(self, x: np.ndarray) -> np.ndarray:
        mode = self.mode.currentText()
        if mode == "Raw":
            return x
        if mode == "AC":
            return x - x.mean(axis=1, keepdims=True)
        lo, hi = self.band_lo.value(), min(self.band_hi.value(), 0.45 * self._fs)
        if not 0 < lo < hi or x.shape[1] < 64:
            return x - x.mean(axis=1, keepdims=True)
        sos = butter(4, (lo, hi), btype="bandpass", fs=self._fs, output="sos")
        return sosfiltfilt(sos, x, axis=1)

    def _redraw(self, force: bool = False) -> None:
        if self._buf is None or self._n == 0 or (self.hold.isChecked() and not force):
            return
        if not (self._dirty or force):
            return
        self._dirty = False
        fs = self._fs
        n_win = max(2, min(self._n, int(self.window_s.value() * fs)))
        n_xy = max(2, min(n_win, int(self.xy_s.value() * fs)))
        raw = self._buf[:, -n_win:]
        y = self._process(raw)
        t = (np.arange(n_win) - n_win) / fs
        self.line_h.set_data(t, y[0])
        if y.shape[0] > 1:
            self.line_v.set_data(t, y[1])
            self.line_xy.set_data(y[0, -n_xy:], y[1, -n_xy:])
        else:
            self.line_v.set_data([], [])
            self.line_xy.set_data([], [])
        if self.autoscale.isChecked():
            for ax in (self.ax_h, self.ax_v, self.ax_xy):
                ax.relim()
                ax.autoscale_view()
        self.canvas.draw_idle()

        dc = raw[:, -n_xy:].mean(axis=1)
        parts = [f"fs {fs:.0f} S/s, XY over {n_xy / fs * 1000:.0f} ms ({self.mode.currentText()})"]
        parts.append("DC " + ", ".join(f"{n} {d:.3f} V" for n, d in zip(NAMES, dc)))
        if y.shape[0] > 1:
            sh, sv, rho, d = ellipse_stats(y[0, -n_xy:], y[1, -n_xy:])
            parts.append(f"AC rms H {sh * 1000:.1f} mV, V {sv * 1000:.1f} mV   rho {rho:+.3f}   |D| {d:.0f} deg")
        else:
            ac = y[0, -n_xy:] - y[0, -n_xy:].mean()
            parts.append(f"AC rms H {np.sqrt(np.mean(ac ** 2)) * 1000:.1f} mV (H only above 5 kS/s: no ellipse)")
        self.readout.setText("\n".join(parts))
