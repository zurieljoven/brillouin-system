"""
"Protocols" tab of the homodyne plane test GUI: the automated glass and phantom runs, borrowed unchanged from the
z-carrier test (glass) and the plane-accuracy test (phantom). Only the settings live here; the runs themselves are
z_carrier_test.runner.run_protocol and plane_accuracy_test.runner.run_experiment, called by HomodyneWorker.

Glass (lens vs stage Z): start with the lens parked on the plane and the Andor connected. Randomized blocks of a
    stationary record plus lens and stage-Z slews (fwd and bwd) through the start positions, Andor sweeps on both
    axes at the start and the end. Both axes return to their start after every step.
Phantom (plane accuracy): set the center and edge in the "Stage-referenced radial run" box first. Randomized blocks
    of find_fwd / find_bwd (the live finder) and an Andor axial scan at every radius from center to edge.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

import numpy as np

from brillouin_system.guis.homodyne_test.experiment_plan import (
    StageRef, circle_fit_mm, profile_nodes, stage_ref_point_r_mm,
)
from brillouin_system.guis.plane_accuracy_test.runner import RunConfig
from brillouin_system.guis.z_carrier_test.protocol import ProtocolConfig


def _dspin(lo, hi, val, step, decimals=1, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    s.setSuffix(suffix)
    return s


def _ispin(lo, hi, val) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s


def _floats(text: str) -> tuple[float, ...]:
    out = tuple(float(t) for t in text.replace(";", ",").split(",") if t.strip())
    if not out:
        raise ValueError(f"No numbers in '{text}'")
    return out


def _fmt_s(s: float) -> str:
    return f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s"


class ProtocolPanel(QWidget):
    start_glass = pyqtSignal(object)        # ProtocolConfig
    start_phantom = pyqtSignal(object)      # RunConfig
    capture_point = pyqtSignal()            # store the current stage XYZ + lens z as an intermediate plane

    def __init__(self, parent=None):
        super().__init__(parent)
        self._center: Optional[StageRef] = None
        self._edge: Optional[StageRef] = None
        self._points: list[StageRef] = []
        lay = QVBoxLayout(self)

        # ---- glass: lens vs stage Z ------------------------------------------------------------------------
        g = QGroupBox("Glass: lens vs stage-Z slews (z-carrier protocol)")
        f = QFormLayout(g)
        f.addRow(QLabel("Park the lens on the plane (slew + Move to last slew peak, or Andor Move to brightest) and "
                        "connect the Andor first. Every step is centred on the lens and stage-Z start positions and "
                        "returns there; nothing moves more than the slew/sweep half-width from them."))
        self.g_sample = QLineEdit("glass")
        self.g_blocks = _ispin(1, 50, 5)
        self.g_seed = _ispin(0, 10_000, 0)
        self.g_speeds = QLineEdit("400, 800, 1200, 1600, 2000")
        self.g_speeds.setToolTip("um/s, comma separated, up to 2000. The lens runs every speed, fwd and bwd; the stage only "
                                 "where its 2v/lambda fringe stays below Nyquist.")
        self.g_rates = QLineEdit("5000, 10000")
        self.g_rates.setToolTip("S/s per channel, comma separated (above 5000 only H is recorded).")
        self.g_stat = _dspin(0.5, 120.0, 10.0, 1.0, 1, " s")
        self.g_half = _dspin(10.0, 400.0, 200.0, 25.0, 0, " um")
        self.g_andor_half = _dspin(5.0, 100.0, 40.0, 5.0, 0, " um")
        self.g_andor_step = _dspin(0.5, 50.0, 2.0, 0.5, 1, " um")
        self.g_exposure = _dspin(0.001, 10.0, 0.1, 0.05, 3, " s")
        self.g_gain = _ispin(0, 300, 0)
        self.g_fp = _dspin(0.0, 1e5, 3396.5, 1.0, 1, " Hz")
        self.g_close = QCheckBox("Close all shutters when done")
        self.g_close.setChecked(True)
        self.g_info = QLabel()
        self.g_btn = QPushButton("Start glass protocol")
        self.g_btn.clicked.connect(self._emit_glass)
        for lbl, w in (("Sample name:", self.g_sample), ("Blocks:", self.g_blocks), ("Seed:", self.g_seed),
                       ("Speeds:", self.g_speeds), ("Sample rates:", self.g_rates), ("Stationary:", self.g_stat),
                       ("Slew ± start:", self.g_half), ("Andor ± start:", self.g_andor_half),
                       ("Andor step:", self.g_andor_step), ("Andor exposure:", self.g_exposure),
                       ("Andor EM gain:", self.g_gain),
                       ("FP mod (logged):", self.g_fp)):
            f.addRow(lbl, w)
        f.addRow(self.g_close)
        f.addRow(self.g_info)
        f.addRow(self.g_btn)
        lay.addWidget(g)

        # ---- phantom: plane accuracy -----------------------------------------------------------------------
        g = QGroupBox("Phantom: plane accuracy (finder fwd/bwd + Andor axial per radius)")
        f = QFormLayout(g)
        f.addRow(QLabel("Uses the center and edge set in the \"Stage-referenced radial run\" box (Positioning "
                        "column). Optionally add intermediate planes: move the stage along the center -> edge line "
                        "(X/Y only), put the lens on the plane there (slew + Move to last slew peak, or Andor), press "
                        "Add point. The predicted plane then passes through every measured plane (PCHIP) instead of "
                        "the 2-point sphere. Stage Z is never moved; the lens backs off before every stage XY move."))
        self.p_refs = QLabel("Center / edge: not set")
        self.p_refs.setStyleSheet("font-family: monospace;")
        self.p_add = QPushButton("Add point here")
        self.p_add.setToolTip("Store the current stage X/Y/Z and lens z as a measured plane on the line.")
        self.p_add.clicked.connect(self.capture_point.emit)
        self.p_clear = QPushButton("Clear points")
        self.p_clear.clicked.connect(self.clear_points)
        self.p_profile = QLabel("")
        self.p_profile.setStyleSheet("font-family: monospace;")
        self.p_dr = _dspin(0.01, 5.0, 0.25, 0.05, 2, " mm")
        self.p_blocks = _ispin(1, 50, 5)
        self.p_seed = _ispin(0, 10_000, 0)
        self.p_half = _dspin(100.0, 1000.0, 400.0, 50.0, 0, " um")
        self.p_speed = _dspin(50.0, 2000.0, 400.0, 50.0, 0, " um/s")
        self.p_sigma = _dspin(1.0, 1000.0, 20.0, 1.0, 0, " σ")
        self.p_stop_after = _dspin(30.0, 1000.0, 150.0, 10.0, 0, " um")
        self.p_ax_before = _dspin(10.0, 1000.0, 150.0, 10.0, 0, " um")
        self.p_ax_after = _dspin(10.0, 1000.0, 200.0, 10.0, 0, " um")
        self.p_ax_step = _dspin(0.5, 100.0, 10.0, 1.0, 1, " um")
        self.p_exposure = _dspin(0.001, 10.0, 0.2, 0.05, 3, " s")
        self.p_gain = _ispin(0, 300, 0)
        self.p_rate = _dspin(1000.0, 5000.0, 5000.0, 500.0, 0, " S/s")
        self.p_close = QCheckBox("Close all shutters when done")
        self.p_close.setChecked(True)
        self.p_info = QLabel()
        self.p_btn = QPushButton("Start phantom protocol")
        self.p_btn.clicked.connect(self._emit_phantom)
        f.addRow(self.p_refs)
        f.addRow(self.p_add, self.p_clear)
        f.addRow(self.p_profile)
        for lbl, w in (("R step:", self.p_dr), ("Blocks:", self.p_blocks), ("Seed:", self.p_seed),
                       ("Find ± predicted plane:", self.p_half), ("Speed:", self.p_speed),
                       ("Finder threshold:", self.p_sigma), ("Finder stop after:", self.p_stop_after),
                       ("Axial before plane:", self.p_ax_before), ("Axial after plane:", self.p_ax_after),
                       ("Axial step:", self.p_ax_step), ("Andor exposure:", self.p_exposure),
                       ("Andor EM gain:", self.p_gain), ("Sample rate:", self.p_rate)):
            f.addRow(lbl, w)
        f.addRow(self.p_close)
        f.addRow(self.p_info)
        f.addRow(self.p_btn)
        lay.addWidget(g)
        lay.addStretch()
        for lbl in self.findChildren(QLabel):
            lbl.setWordWrap(True)

        for w in self.findChildren((QDoubleSpinBox, QSpinBox)):
            w.valueChanged.connect(self.update_info)
        for w in (self.g_speeds, self.g_rates):
            w.textChanged.connect(self.update_info)
        self.update_info()

    # ---- configs ---------------------------------------------------------------------------------------------

    def glass_config(self) -> ProtocolConfig:
        cfg = ProtocolConfig(
            sample=self.g_sample.text().strip() or "glass", n_blocks=self.g_blocks.value(), seed=self.g_seed.value(),
            stationary_s=self.g_stat.value(), slew_half_um=self.g_half.value(),
            speeds_um_s=_floats(self.g_speeds.text()), sample_rates_hz=_floats(self.g_rates.text()),
            andor_half_um=self.g_andor_half.value(), andor_step_um=self.g_andor_step.value(),
            exposure_s=self.g_exposure.value(), gain=self.g_gain.value(), pd_gain_db=None,
            fp_mod_hz=self.g_fp.value(), close_shutters_at_end=self.g_close.isChecked(), lens_above_nyquist=True)
        cfg.validate()
        return cfg

    def set_refs(self, center: Optional[StageRef], edge: Optional[StageRef]) -> None:
        self._center, self._edge = center, edge
        self.update_info()

    def add_point(self, p: StageRef) -> None:
        self._points.append(p)
        self.update_info()

    def clear_points(self) -> None:
        self._points = []
        self.update_info()

    def _profile_text(self) -> str:
        c, e = self._center, self._edge
        rows = ["  R (mm)   lens z (um)   source"]
        for i, p in enumerate(self._points, 1):
            r, off = stage_ref_point_r_mm(c, e, p)
            rows.append(f"  {r:6.3f}   {p.lens_um:10.1f}   point {i} ({off:.0f} um off line)")
        r_n, z_n = profile_nodes(c, e, self._points)
        rows[1:] = sorted(rows[1:])
        rows.insert(1, f"  {0.0:6.3f}   {c.lens_um:10.1f}   center")
        rows.append(f"  {r_n[-1]:6.3f}   {e.lens_um:10.1f}   edge")
        fit = circle_fit_mm(r_n, z_n)
        if fit is not None:
            R, x0, res = fit
            rows.append(f"Circle fit: R = {R:.2f} mm, apex at {x0:+.2f} mm, residuals "
                        + " ".join(f"{v:+.0f}" for v in res) + " um")
        return "\n".join(rows)

    def phantom_config(self) -> RunConfig:
        if self._center is None or self._edge is None:
            raise ValueError("Set the center and the edge first (Stage-referenced radial run box).")
        cfg = RunConfig(
            center=self._center, edge=self._edge, r_step_mm=self.p_dr.value(), n_blocks=self.p_blocks.value(),
            seed=self.p_seed.value(), half_window_um=self.p_half.value(), speed_um_s=self.p_speed.value(),
            n_sigma=self.p_sigma.value(), stop_after_um=self.p_stop_after.value(),
            axial_before_um=self.p_ax_before.value(), axial_after_um=self.p_ax_after.value(),
            axial_step_um=self.p_ax_step.value(), exposure_s=self.p_exposure.value(), gain=self.p_gain.value(),
            sample_rate_hz=self.p_rate.value(), close_shutters_at_end=self.p_close.isChecked(),
            points=tuple(self._points))
        cfg.validate()
        return cfg

    def update_info(self, *_):
        try:
            cfg = self.glass_config()
            n = len(cfg.steps())
            skipped = cfg.skipped()
            n_lens = sum(a == "lens" for a, _, _ in cfg.slews())
            self.g_info.setText(f"{n} steps, about {_fmt_s(cfg.estimated_s())}. Per block: {2 * n_lens} lens slews, "
                                f"{2 * (len(cfg.slews()) - n_lens)} stage slews."
                                + (" Stage skipped (fringe above Nyquist): "
                                   + ", ".join(f"{v:.0f} um/s at {fs:.0f}" for v, fs in skipped) if skipped else ""))
        except ValueError as e:
            self.g_info.setText(f"Invalid: {e}")
        c, e = self._center, self._edge
        if c is None or e is None:
            self.p_refs.setText("Center / edge: not set")
            self.p_profile.setText(f"{len(self._points)} intermediate point(s) stored" if self._points else "")
            self.p_info.setText("")
            return
        self.p_refs.setText(f"Center: stage {c.stage_x_um:.0f} / {c.stage_y_um:.0f} um, lens {c.lens_um:.1f} um\n"
                            f"Edge:   stage {e.stage_x_um:.0f} / {e.stage_y_um:.0f} um, lens {e.lens_um:.1f} um")
        try:
            self.p_profile.setText(self._profile_text())
        except ValueError as ex:
            self.p_profile.setText(f"Points invalid: {ex}")
        try:
            cfg = self.phantom_config()
            radii = cfg.radii()
            self.p_info.setText(f"{len(radii)} radii ({radii[0]:.2f}-{radii[-1]:.2f} mm) x 3 kinds x "
                                f"{cfg.n_blocks} blocks = {len(cfg.trials())} trials, about {_fmt_s(cfg.estimated_s())}.")
        except ValueError as ex:
            self.p_info.setText(f"Invalid: {ex}")

    def set_enabled(self, glass: bool, phantom: bool, can_capture: bool = True) -> None:
        self.g_btn.setEnabled(glass)
        self.p_btn.setEnabled(phantom)
        self.p_add.setEnabled(can_capture)

    def _emit(self, make, signal, title: str):
        try:
            cfg = make()
        except ValueError as e:
            QMessageBox.warning(self, title, str(e))
            return
        signal.emit(cfg)

    def _emit_glass(self):
        self._emit(self.glass_config, self.start_glass, "Glass protocol")

    def _emit_phantom(self):
        self._emit(self.phantom_config, self.start_phantom, "Phantom protocol")
