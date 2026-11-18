"""
Unattended run of a predefined plan (no Qt, no hardware): keep taking sweep scans at the points not yet done, in
random order, until every point has a scan that passed its motion gate. Every attempt, failed or not, is logged and
its validation data saved, so the failure statistics can be checked later.

Run folder:
  attempts.csv                 one row per attempt (point, try number, outcome, reason, eye-tracker positions, files)
  plan.json                    the steps and settings
  scans/<id>.h5                each successful sweep scan (same HDF5 format as "Save all axial scans")
  finds/<nnnn>_<id>.h5         raw homodyne traces of every live find in the attempt (validation_recorder)
  eye/<nnnn>_<id>_pre.npz      eye-tracker frames when the sweep started (after Move XYZ) and when it ended
  eye/<nnnn>_<id>_post.npz
  calibration_<name>.h5        calibrations taken by the run (e.g. coarse before, fine for the ePSF after)

The frontend owns the sequencing (calibrations, Move XYZ, sweep request); this module only decides what comes next
and writes the records.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from brillouin_system.guis.human_interface.predefined_plan import PlanStep
from brillouin_system.scan_managers.validation_recorder import SweepAttempt, save_eye_snapshot, save_finder_traces

FIELDS = ["attempt", "time", "step_id", "R_mm", "phi_deg", "depth_um", "try", "success", "reason", "n_frames",
          "best_motion_um", "motion_limit_um", "elapsed_s", "n_finds", "n_found",
          "pre_laser_x_mm", "pre_laser_y_mm", "pre_dc_mm", "post_laser_x_mm", "post_laser_y_mm", "post_dc_mm",
          "scan_file", "finds_file", "eye_pre_file", "eye_post_file"]


def build_grid_steps(radii_mm, depths_um, *, phi_deg: float = 0.0, delta_c_mm: float = 0.0,
                     motion_limit_um: float = 100.0, max_time_s: float = 10.0, timed: bool = True,
                     cycles: Optional[int] = None) -> list[PlanStep]:
    """One step per (R, depth), replicate 1."""
    return [PlanStep(depth_um=float(d), num=1, timed=timed, cycles=None if timed else int(cycles),
                     max_time_s=float(max_time_s), R_mm=float(r), phi_deg=float(phi_deg),
                     delta_c_mm=float(delta_c_mm), motion_limit_um=float(motion_limit_um))
            for r in radii_mm for d in depths_um]


def _eye_xy_dc(res):
    if res is None or res.laser_position is None:
        return None, None, getattr(res, "delta_laser_corner", None)
    return float(res.laser_position[0]), float(res.laser_position[1]), res.delta_laser_corner


class PlanAutoRun:
    def __init__(self, steps: list[PlanStep], run_dir: Path, *, max_attempts: int = 0,
                 seed: Optional[int] = None, settings: Optional[dict] = None):
        """max_attempts: per point; 0 = keep trying until it passes (or the run is stopped)."""
        if not steps:
            raise ValueError("The plan has no steps.")
        ids = [s.id for s in steps]
        if len(set(ids)) != len(ids):
            raise ValueError("Plan step IDs must be unique.")
        self.steps = list(steps)
        self.run_dir = Path(run_dir)
        self.max_attempts = int(max_attempts)
        self.rng = random.Random(seed)
        self.done: set[str] = set()
        self.gave_up: set[str] = set()
        self.tries: dict[str, int] = {s.id: 0 for s in steps}
        self.n_attempts = 0
        self._last_id: Optional[str] = None
        self.run_dir.mkdir(parents=True, exist_ok=False)
        with open(self.run_dir / "plan.json", "w") as fh:
            json.dump(dict(created=datetime.now().isoformat(timespec="seconds"), max_attempts=self.max_attempts,
                           seed=seed, settings=settings or {}, steps=[asdict(s) | {"id": s.id} for s in steps]),
                      fh, indent=2, default=str)
        with open(self.run_dir / "attempts.csv", "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    # ---- sequencing -------------------------------------------------------------------------------------------

    def pending(self) -> list[PlanStep]:
        return [s for s in self.steps if s.id not in self.done and s.id not in self.gave_up]

    def next_step(self) -> Optional[PlanStep]:
        """A random pending point; not the one just tried unless it is the only one left. None when finished."""
        p = self.pending()
        if not p:
            return None
        if len(p) > 1 and self._last_id is not None:
            p = [s for s in p if s.id != self._last_id] or p
        s = self.rng.choice(p)
        self._last_id = s.id
        return s

    @property
    def finished(self) -> bool:
        return not self.pending()

    def summary(self) -> str:
        return (f"{len(self.done)}/{len(self.steps)} points done, {len(self.gave_up)} given up, "
                f"{self.n_attempts} attempts")

    # ---- records ----------------------------------------------------------------------------------------------

    def record(self, step: PlanStep, success: bool, elapsed_s: float, attempt: Optional[SweepAttempt] = None,
               eye_pre=None, eye_post=None) -> dict:
        """Log one attempt and save its validation data. attempt: backend.last_sweep_attempt (None if the sweep
        never ran, e.g. no calibration). Returns the CSV row."""
        self.n_attempts += 1
        self.tries[step.id] += 1
        n = self.n_attempts
        stem = f"{n:04d}_{step.id}"
        if attempt is not None and attempt.scan_id != step.id:
            attempt = None          # stale record from an earlier scan: do not attribute it to this point
        reason = "" if success else (attempt.reason if attempt is not None and attempt.reason else "sweep did not run")
        files = dict(scan_file="", finds_file="", eye_pre_file="", eye_post_file="")
        errors = []

        def _try(key, fn):
            try:
                files[key] = fn()
            except Exception as e:      # a failed save must not stop the run
                errors.append(f"{key}: {type(e).__name__}: {e}")

        if attempt is not None and attempt.finds:
            _try("finds_file", lambda: str(save_finder_traces(
                self.run_dir / "finds" / f"{stem}.h5", attempt.finds, step_id=step.id, success=bool(success),
                reason=reason).relative_to(self.run_dir)))
        for tag, res in (("pre", eye_pre), ("post", eye_post)):
            if res is not None:
                _try(f"eye_{tag}_file", lambda res=res, tag=tag: str(save_eye_snapshot(
                    self.run_dir / "eye" / f"{stem}_{tag}.npz", res).relative_to(self.run_dir)))
        if success and attempt is not None and attempt.scan is not None:
            _try("scan_file", lambda: str(self._save_h5([attempt.scan], self.run_dir / "scans" / f"{step.id}.h5")
                                          .relative_to(self.run_dir)))

        if success:
            self.done.add(step.id)
        elif self.max_attempts > 0 and self.tries[step.id] >= self.max_attempts:
            self.gave_up.add(step.id)

        pre, post = _eye_xy_dc(eye_pre), _eye_xy_dc(eye_post)
        finds = attempt.finds if attempt is not None else []
        row = dict(attempt=n, time=datetime.now().isoformat(timespec="seconds"), step_id=step.id, R_mm=step.R_mm,
                   phi_deg=step.phi_deg, depth_um=step.depth_um, success=bool(success),
                   reason=reason + (f" | save errors: {'; '.join(errors)}" if errors else ""),
                   n_frames=attempt.n_frames if attempt is not None else 0,
                   best_motion_um=attempt.best_motion_um if attempt is not None else None,
                   motion_limit_um=step.motion_limit_um, elapsed_s=round(float(elapsed_s), 2),
                   n_finds=len(finds), n_found=sum(1 for _, r in finds if r.found),
                   pre_laser_x_mm=pre[0], pre_laser_y_mm=pre[1], pre_dc_mm=pre[2],
                   post_laser_x_mm=post[0], post_laser_y_mm=post[1], post_dc_mm=post[2], **files)
        row["try"] = self.tries[step.id]
        with open(self.run_dir / "attempts.csv", "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
        return row

    def save_calibration(self, name: str, calibration_data) -> Path:
        return self._save_h5(calibration_data, self.run_dir / f"calibration_{name}.h5")

    @staticmethod
    def _save_h5(obj, path: Path) -> Path:
        from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
            dataclass_to_hdf5_native_dict, save_dict_to_hdf5,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict_to_hdf5(str(path), dataclass_to_hdf5_native_dict(obj))
        return path
