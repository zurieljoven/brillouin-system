"""
Spectrogram tab for the homodyne plane test GUI.

One spectrogram per channel against time (Hann window; 1024 samples for stationary records, 256 for slews so the
fringe-rate sweep is resolved), colour in dB re 1 V^2/Hz on one scale shared by all channels so H and V compare
directly. Slews add the axis z against the same time axis, and the slew peak is marked on every panel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from scipy.signal import spectrogram

NPERSEG = {"stationary": 1024, "slew": 256}
OVERLAP_FRACTION = {"stationary": 0.75, "slew": 0.875}
NAMES = ("H", "V")


def channel_spectrograms(values: np.ndarray, fs: float, mode: str):
    """(f, t, S) with S shaped (n_ch, n_f, n_t) in V^2/Hz; each channel's mean removed first."""
    v = np.atleast_2d(np.asarray(values, dtype=np.float64))
    nper = min(NPERSEG.get(mode, 1024), v.shape[-1])
    nover = int(nper * OVERLAP_FRACTION.get(mode, 0.75))
    out = []
    for x in v:
        f, t, s = spectrogram(x - x.mean(), fs, window="hann", nperseg=nper, noverlap=nover)
        out.append(s)
    return f, t, np.stack(out)


class SpectrogramView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._last = None
        self.fig = Figure(figsize=(9, 8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.auto = QCheckBox("Auto colour range")
        self.auto.setChecked(True)
        self.vmin = QDoubleSpinBox(); self.vmin.setRange(-250, 50); self.vmin.setValue(-120); self.vmin.setSuffix(" dB")
        self.vmax = QDoubleSpinBox(); self.vmax.setRange(-250, 50); self.vmax.setValue(-60); self.vmax.setSuffix(" dB")
        self.fmax = QDoubleSpinBox(); self.fmax.setRange(10, 1e6); self.fmax.setDecimals(0); self.fmax.setValue(2500)
        self.fmax.setSuffix(" Hz")
        for w in (self.auto, self.vmin, self.vmax, self.fmax):
            (w.toggled if w is self.auto else w.valueChanged).connect(self._redraw)
        row = QHBoxLayout()
        row.addWidget(self.auto)
        for lbl, w in (("min", self.vmin), ("max", self.vmax), ("f max", self.fmax)):
            row.addWidget(QLabel(lbl)); row.addWidget(w)
        row.addStretch()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, stretch=1)

    def show_recording(self, rec, z: Optional[np.ndarray], t_peak: Optional[float], title: str) -> None:
        if rec.n_samples < 64:
            self._last = None
            self.fig.clear(); self.canvas.draw_idle()
            return
        f, t, s = channel_spectrograms(rec.values, rec.sample_rate_hz, rec.mode)
        self._last = dict(f=f, t=t, s_db=10 * np.log10(np.clip(s, 1e-30, None)), z=z, t_all=rec.t_s(),
                          t_peak=t_peak, title=title, axis=rec.meta.get("axis", "lens"),
                          channels=rec.channels)
        self._redraw()

    def _redraw(self, *_):
        d = self._last
        self.vmin.setEnabled(not self.auto.isChecked())
        self.vmax.setEnabled(not self.auto.isChecked())
        if d is None:
            return
        f, t, s_db = d["f"], d["t"], d["s_db"]
        fk = f <= self.fmax.value()
        if self.auto.isChecked():
            sel = s_db[:, fk & (f > 50)] if np.any(fk & (f > 50)) else s_db
            p5, p50, hi = np.percentile(sel, [5, 50, 99.9])
            lo = max(p5, p50 - 40.0)        # dead stretches (flat ADC codes, dropouts) sit at the 1e-30 floor
            self.vmin.blockSignals(True); self.vmax.blockSignals(True)
            self.vmin.setValue(lo); self.vmax.setValue(hi)
            self.vmin.blockSignals(False); self.vmax.blockSignals(False)
        lo, hi = self.vmin.value(), self.vmax.value()

        n_ch = s_db.shape[0]
        has_z = d["z"] is not None
        self.fig.clear()
        ratios = [3] * n_ch + ([1] if has_z else [])
        axs = self.fig.subplots(len(ratios), 1, sharex=True, gridspec_kw={"height_ratios": ratios})
        axs = np.atleast_1d(axs)
        m = None
        for i in range(n_ch):
            m = axs[i].pcolormesh(t, f[fk], s_db[i][fk], shading="auto", cmap="magma", vmin=lo, vmax=hi,
                                  rasterized=True)
            ch = d["channels"][i] if i < len(d["channels"]) else f"ch{i}"
            axs[i].set_ylabel(f"{NAMES[i] if i < 2 else ch} ({ch})\nfrequency (Hz)")
        if has_z:
            step = max(1, d["t_all"].size // 20_000)
            axs[-1].plot(d["t_all"][::step], d["z"][::step], lw=0.8, color="k")
            axs[-1].set_ylabel(f"{d['axis']} z (um)")
        if d["t_peak"] is not None:
            for a in axs:
                a.axvline(d["t_peak"], color="c", lw=0.8, ls="--")
        axs[-1].set_xlabel("time (s)")
        axs[0].set_title(d["title"], fontsize=9)
        if m is not None:
            self.fig.colorbar(m, ax=list(axs[:n_ch]), label="PSD (dB re 1 V$^2$/Hz)", shrink=0.9)
        self.canvas.draw_idle()
