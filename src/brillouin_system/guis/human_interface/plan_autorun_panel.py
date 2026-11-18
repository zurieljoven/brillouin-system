"""
"Auto-run plan" box for the human interface: runs the loaded predefined plan unattended.

  1. optional coarse calibration (default 4-8 GHz, 0.1 GHz step), saved to the run folder
  2. the plan's points in random order, one sweep scan each (the viewer's run_plan_step: Move XYZ, sweep settings,
     motion-gated sweep scan), retrying failed points until every point has a passing scan
     (or a per-point attempt cap is reached); every attempt logged by PlanAutoRun
  3. optional fine calibration (default 4-8 GHz, 0.01 GHz step) for the ePSF, saved to the run folder

The sequencing lives here; the decisions and records live in plan_autorun.PlanAutoRun (no Qt). The viewer only
provides run_plan_step(), the loaded plan steps and the signaller. STOP aborts at once (no fine calibration);
"Stop after this point" ends the point loop and still runs the fine calibration.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox,
)

from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.guis.human_interface.plan_autorun import PlanAutoRun
from brillouin_system.logging_utils.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_RUN_DIR = Path.home() / "autorun_runs"
PAUSE_MS = 800      # between attempts


def _spin(lo, hi, val, step, dec, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step); w.setValue(val); w.setSuffix(suffix)
    return w


class PlanAutoRunPanel(QGroupBox):
    def __init__(self, viewer, parent=None):
        super().__init__("Auto-run plan (until every point passes)", parent)
        self.viewer = viewer
        self.run: Optional[PlanAutoRun] = None
        self.phase = ""                 # "cal_coarse" | "points" | "cal_fine" | ""
        self._step = None
        self._waiting = False           # a sweep request of ours is in flight
        self._soft_stop = False
        self._abort = False
        self._eye_pre = None
        self._cal_backup = None

        self.max_attempts = QSpinBox(); self.max_attempts.setRange(0, 1000); self.max_attempts.setValue(0)
        self.max_attempts.setSpecialValueText("unlimited")
        self.cal_lo = _spin(0.0, 20.0, 4.0, 0.1, 2, " GHz")
        self.cal_hi = _spin(0.0, 20.0, 8.0, 0.1, 2, " GHz")
        self.coarse = QCheckBox("Coarse calibration first"); self.coarse.setChecked(True)
        self.coarse_step = _spin(0.001, 1.0, 0.1, 0.01, 3, " GHz")
        self.fine = QCheckBox("Fine calibration after (ePSF)"); self.fine.setChecked(True)
        self.fine_step = _spin(0.001, 1.0, 0.01, 0.005, 3, " GHz")
        self.folder = QLineEdit(str(DEFAULT_RUN_DIR))
        browse = QPushButton("…"); browse.setFixedWidth(30); browse.clicked.connect(self._browse)
        self.start_btn = QPushButton("Start auto-run"); self.start_btn.clicked.connect(self.start)
        self.soft_btn = QPushButton("Stop after this point"); self.soft_btn.clicked.connect(self.soft_stop)
        self.soft_btn.setEnabled(False)
        self.status = QLabel("Load a plan above, then start."); self.status.setWordWrap(True)

        f = QFormLayout(self)
        f.addRow("Max attempts / point:", self.max_attempts)
        row = QHBoxLayout(); row.addWidget(self.cal_lo); row.addWidget(QLabel("to")); row.addWidget(self.cal_hi)
        f.addRow("Calibration range:", row)
        row = QHBoxLayout(); row.addWidget(self.coarse); row.addWidget(self.coarse_step)
        f.addRow(row)
        row = QHBoxLayout(); row.addWidget(self.fine); row.addWidget(self.fine_step)
        f.addRow(row)
        row = QHBoxLayout(); row.addWidget(self.folder); row.addWidget(browse)
        f.addRow("Run folder:", row)
        row = QHBoxLayout(); row.addWidget(self.start_btn); row.addWidget(self.soft_btn)
        f.addRow(row)
        f.addRow(self.status)

        sig = viewer.brillouin_signaller
        sig.sweep_scan_started.connect(self._on_sweep_started)
        sig.sweep_scan_finished.connect(self._on_sweep_finished)
        sig.calibration_finished.connect(self._on_cal_finished)
        sig.calibration_failed.connect(self._on_cal_failed)

    # ---- control ----------------------------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.run is not None

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Auto-run folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def start(self):
        if self.active:
            return
        steps = list(getattr(self.viewer, "_predef_steps", []) or [])
        if not steps:
            QMessageBox.information(self, "Auto-run", "Load a measurement plan first (Import Plan…).")
            return
        if getattr(self.viewer, "_predef_run_active", False):
            QMessageBox.information(self, "Auto-run", "A predefined step is still running.")
            return
        if self.cal_hi.value() <= self.cal_lo.value():
            QMessageBox.warning(self, "Auto-run", "Calibration range: the upper frequency must be above the lower.")
            return
        run_dir = Path(self.folder.text()) / f"autorun_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cal = (f"coarse {self.coarse_step.value():g} GHz, " if self.coarse.isChecked() else "") + \
              (f"fine {self.fine_step.value():g} GHz" if self.fine.isChecked() else "")
        r = QMessageBox.question(
            self, "Start auto-run",
            f"{len(steps)} points, random order, each retried until it passes "
            f"({'no cap' if self.max_attempts.value() == 0 else f'max {self.max_attempts.value()} attempts'}).\n"
            f"Calibrations {self.cal_lo.value():g}-{self.cal_hi.value():g} GHz: {cal or 'none'}.\n"
            f"The eye lens and rig stage will move. Saving to\n{run_dir}\n\nStart?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        settings = dict(cal_range_ghz=[self.cal_lo.value(), self.cal_hi.value()],
                        coarse_step_ghz=self.coarse_step.value() if self.coarse.isChecked() else None,
                        fine_step_ghz=self.fine_step.value() if self.fine.isChecked() else None)
        try:
            self.run = PlanAutoRun(steps, run_dir, max_attempts=self.max_attempts.value(), settings=settings)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Auto-run", str(e))
            return
        self._soft_stop = self._abort = False
        self.start_btn.setEnabled(False)
        self.soft_btn.setEnabled(True)
        log.info(f"[Auto-run] Started: {len(steps)} points -> {run_dir}")
        if self.coarse.isChecked():
            self._calibrate("coarse", self.coarse_step.value())
        else:
            self._next()

    def soft_stop(self):
        """End the point loop after the current attempt; the fine calibration still runs."""
        if self.active:
            self._soft_stop = True
            self._set_status("Stopping after this point …")

    def abort(self):
        """STOP: end everything after the current request returns (no fine calibration)."""
        if self.active:
            self._abort = True
            self._set_status("Aborting (STOP) …")
            if not self._waiting and self.phase not in ("cal_coarse", "cal_fine"):
                self._finish("stopped")

    # ---- calibrations -----------------------------------------------------------------------------------------

    def _calibrate(self, name: str, step_ghz: float):
        self.phase = f"cal_{name}"
        cfg = calibration_config.get()
        self._cal_backup = dict(start=cfg.start, stop=cfg.stop, step=cfg.step)
        calibration_config.update(start=float(self.cal_lo.value()), stop=float(self.cal_hi.value()), step=float(step_ghz))
        n = len(calibration_config.get().calibration_freqs)
        self._set_status(f"{name.capitalize()} calibration: {n} frequencies at {step_ghz:g} GHz …")
        log.info(f"[Auto-run] {name} calibration: {n} frequencies")
        self.viewer.run_calibration_requested.emit()

    def _restore_cal_config(self):
        if self._cal_backup is not None:
            calibration_config.update(**self._cal_backup)
            self._cal_backup = None

    def _on_cal_finished(self):
        if not self.active or not self.phase.startswith("cal_"):
            return
        name = self.phase[4:]
        self._restore_cal_config()
        try:
            data = self.viewer.brillouin_signaller.backend.calibration_data
            path = self.run.save_calibration(name, data)
            log.info(f"[Auto-run] {name} calibration saved: {path}")
        except Exception as e:
            log.exception(f"[Auto-run] Could not save the {name} calibration: {e}")
        if name == "coarse":
            if self._abort:
                self._finish("stopped")
            else:
                self._next()
        else:
            self._finish("finished")

    def _on_cal_failed(self):
        if not self.active or not self.phase.startswith("cal_"):
            return
        name = self.phase[4:]
        self._restore_cal_config()
        self._finish(f"{name} calibration failed or was cancelled")

    # ---- points -----------------------------------------------------------------------------------------------

    def _next(self):
        if not self.active:
            return
        if self._abort:
            self._finish("stopped")
            return
        step = None if self._soft_stop else self.run.next_step()
        if step is None:
            if self.fine.isChecked():
                self._calibrate("fine", self.fine_step.value())
            else:
                self._finish("finished" if self.run.finished else "stopped")
            return
        self.phase = "points"
        self._step, self._eye_pre, self._waiting = step, None, True
        self.viewer.brillouin_signaller.backend.last_sweep_attempt = None
        self._set_status(f"{self.run.summary()}. Now {step.id} (try {self.run.tries[step.id] + 1}).")
        if not self.viewer.run_plan_step(step):
            self._waiting = False
            self.run.record(step, False, 0.0, None)
            QTimer.singleShot(PAUSE_MS, self._next)

    def _on_sweep_started(self, _budget_s: float):
        if self._waiting:
            self._eye_pre = self.viewer.lastest_eye_tracker_results

    def _on_sweep_finished(self, success: bool, elapsed: float):
        if not self._waiting:
            return
        self._waiting = False
        attempt = getattr(self.viewer.brillouin_signaller.backend, "last_sweep_attempt", None)
        row = self.run.record(self._step, bool(success), elapsed, attempt,
                              eye_pre=self._eye_pre, eye_post=self.viewer.lastest_eye_tracker_results)
        log.info(f"[Auto-run] {row['step_id']} try {row['try']}: "
                 f"{'PASS' if row['success'] else 'fail: ' + row['reason']}")
        QTimer.singleShot(PAUSE_MS, self._next)

    def _finish(self, how: str):
        run = self.run
        self._restore_cal_config()
        self.run, self.phase, self._waiting = None, "", False
        self.start_btn.setEnabled(True)
        self.soft_btn.setEnabled(False)
        msg = f"Auto-run {how}: {run.summary()}. Folder: {run.run_dir}" if run is not None else f"Auto-run {how}."
        self._set_status(msg)
        log.info(f"[Auto-run] {msg}")

    def _set_status(self, text: str):
        self.status.setText(text)
