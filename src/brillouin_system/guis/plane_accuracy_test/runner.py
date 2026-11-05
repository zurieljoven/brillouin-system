"""
The precision + accuracy run, without Qt: stage, lens, homodyne DAQ, Andor.

Per trial: lens back if needed, rig stage X/Y to the point at radius R (stage Z
is never moved), then one of
    find_fwd / find_bwd   the live plane finder from predicted plane -+ half window
    axial                 Andor frames stepping through the predicted plane
Every trial is written to disk as it finishes, misses included.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from brillouin_system.guis.homodyne_test.experiment_plan import (
    MAX_STAGE_STEP_UM, StageRef, stage_ref_distance_mm, stage_ref_lens_um, stage_ref_radii,
    stage_ref_sphere_radius_mm, stage_ref_xy_um,
)
from brillouin_system.guis.plane_accuracy_test.axial_recording import AxialRecording, save_axial
from brillouin_system.guis.plane_accuracy_test.trial_plan import Trial, axial_positions_um, build_trials
from brillouin_system.scan_managers.homodyne_plane_finder import (
    PRE_ROLL_S, HomodyneFinderParams, find_reflection_homodyne,
)
from brillouin_system.scan_managers.homodyne_recording import HomodyneRecording, save_recording
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult

STAGE_PRELOAD_UM = 50.0     # every XY target is approached from the same side (random order reverses the stage)
LENS_PRELOAD_UM = 30.0      # axial scans always step toward +z
STAGE_SETTLE_S = 0.2
LENS_SETTLE_S = 0.2
FRAME_SETTLE_S = 0.05

LOG_FIELDS = ["trial", "block", "r_mm", "kind", "file", "status", "time", "t_perf", "stage_x_um", "stage_y_um",
              "lens_pred_um", "finder_found", "finder_z_um", "finder_peak", "finder_stop_z_um",
              "axial_n_frames", "axial_peak_z_um"]


@dataclass
class Devices:
    daq: object
    lens: object
    stage: object
    camera: object
    shutters: object = None


@dataclass(frozen=True)
class RunConfig:
    center: StageRef
    edge: StageRef
    r_step_mm: float = 0.25
    n_blocks: int = 5
    seed: int = 0
    half_window_um: float = 400.0
    speed_um_s: float = 400.0
    n_sigma: float = 20.0
    stop_after_um: float = 150.0
    axial_before_um: float = 150.0
    axial_after_um: float = 200.0
    axial_step_um: float = 10.0
    exposure_s: float = 0.2
    gain: int = 0
    sample_rate_hz: float = 5000.0
    close_shutters_at_end: bool = True

    def radii(self) -> list[float]:
        return stage_ref_radii(self.center, self.edge, self.r_step_mm)

    def trials(self) -> list[Trial]:
        return build_trials(self.radii(), self.n_blocks, self.seed)

    def finder_params(self) -> HomodyneFinderParams:
        return HomodyneFinderParams(
            speed_um_s=self.speed_um_s, n_sigma=self.n_sigma, stop_after_um=self.stop_after_um,
            max_distance_um=max(2.0 * self.half_window_um, self.half_window_um + self.stop_after_um + 50.0))

    def validate(self) -> None:
        p = HomodyneFinderParams()
        if p.band_hz[1] >= 0.5 * self.sample_rate_hz:
            raise ValueError(f"The sample rate must be above {2 * p.band_hz[1]:.0f} S/s per channel.")
        travel_bg = (p.bg_start_s + p.bg_duration_s) * self.speed_um_s
        if self.half_window_um < travel_bg + 50.0:
            raise ValueError(f"Scan ± plane must be at least {travel_bg + 50.0:.0f} um at this speed.")
        d_um = 1000.0 * stage_ref_distance_mm(self.center, self.edge)
        if d_um + STAGE_PRELOAD_UM > MAX_STAGE_STEP_UM:
            raise ValueError(f"Center to edge is {d_um:.0f} um, above the {MAX_STAGE_STEP_UM:.0f} um limit.")

    def estimated_s(self) -> float:
        n_r, b = len(self.radii()), self.n_blocks
        find = (self.half_window_um + self.stop_after_um) / self.speed_um_s + 2.0
        n_frames = axial_positions_um(0.0, self.axial_before_um, self.axial_after_um, self.axial_step_um).size
        axial = n_frames * (self.exposure_s + FRAME_SETTLE_S + 0.05) + 1.5
        return n_r * b * (2.0 * find + axial)

    def plan_dict(self) -> dict:
        radii = self.radii()
        return dict(asdict(self), radii_mm=radii, mode="plane_accuracy", in_vitro=True,
                    sphere_radius_mm=stage_ref_sphere_radius_mm(self.center, self.edge),
                    lens_pred_um=[stage_ref_lens_um(self.center, self.edge, r) for r in radii],
                    stage_preload_um=STAGE_PRELOAD_UM, lens_preload_um=LENS_PRELOAD_UM,
                    created=datetime.now().isoformat(timespec="seconds"),
                    trials=[asdict(t) for t in self.trials()])


class Cancelled(Exception):
    pass


def recording_from_find(r: ReflectionResult, daq, *, p: HomodyneFinderParams, z0: float, z_start: float,
                        direction: float, meta: dict) -> HomodyneRecording:
    z = r.zaber_lens_z_um if r.zaber_lens_z_um is not None else np.empty(0)
    meta = dict(meta, z0_um=z0, z_start_um=z_start, direction=direction, pre_roll_s=PRE_ROLL_S, post_roll_s=0.0,
                backoff_um=abs(z0 - z_start), range_um=p.max_distance_um, speed_um_s=p.speed_um_s,
                reverse=direction < 0, finder_found=r.found, finder_z_um=r.event_z_um, finder_peak=r.peak_value,
                finder_threshold=r.threshold_high, finder_background_mean=r.background_mean,
                finder_background_std=r.background_std, finder_i_arm=r.idx_first, finder_i_max=r.idx_last,
                finder_n_sigma=p.n_sigma, finder_stop_after_um=p.stop_after_um, finder_band_hz=list(p.band_hz),
                finder_rms_window_s=p.rms_window_s, finder_stop_z_um=float(z[-1]) if z.size else None,
                channels=list(daq.channels), sample_rate_hz=float(daq.sample_rate_hz), wavelength_nm=780.24)
    return HomodyneRecording(mode="slew", values=r.daq_values, sample_rate_hz=float(daq.sample_rate_hz),
                             t0_perf=float(r.daq_ts[0]), channels=tuple(daq.channels), meta=meta,
                             zlog_t_perf=r.zaber_lens_ts if r.zaber_lens_ts is not None else np.empty(0), zlog_z_um=z)


def move_stage_xy(stage, x: float, y: float) -> None:
    """Absolute X/Y with a fixed-side preload. Stage Z is never commanded."""
    stage.move_abs(x=x - STAGE_PRELOAD_UM, y=y - STAGE_PRELOAD_UM)
    stage.move_abs(x=x, y=y)
    time.sleep(STAGE_SETTLE_S)


def run_find(dev: Devices, p: HomodyneFinderParams, z_pred: float, half_um: float, direction: float, meta: dict,
             stop: threading.Event) -> HomodyneRecording | None:
    z_start = z_pred - direction * half_um
    dev.lens.move_abs(z_start)
    time.sleep(LENS_SETTLE_S)
    r = find_reflection_homodyne(dev.daq, dev.lens, p, direction=direction, stop_evt=stop)
    if r.daq_values is None:
        return None
    return recording_from_find(r, dev.daq, p=p, z0=z_pred, z_start=z_start, direction=direction, meta=meta)


def run_axial(dev: Devices, before_um: float, after_um: float, step_um: float, z_pred: float, meta: dict,
              stop: threading.Event,
              on_frame: Optional[Callable[[np.ndarray, float], None]] = None) -> AxialRecording:
    targets = axial_positions_um(z_pred, before_um, after_um, step_um)
    dev.lens.move_abs(targets[0] - LENS_PRELOAD_UM)
    frames, zs, ts = [], [], []
    for z in targets:
        if stop.is_set():
            raise Cancelled()
        dev.lens.move_abs(float(z))
        time.sleep(FRAME_SETTLE_S)
        zs.append(float(dev.lens.get_position()))
        ts.append(time.perf_counter())
        frame = np.asarray(dev.camera.snap())
        frames.append(frame)
        if on_frame is not None:
            on_frame(frame, zs[-1])
    meta = dict(meta, z_pred_um=z_pred, exposure_s=float(dev.camera.get_exposure_time()),
                gain=float(dev.camera.get_emccd_gain()), roi=[int(v) for v in dev.camera.get_roi()],
                binning=[int(v) for v in dev.camera.get_binning()],
                before_um=before_um, after_um=after_um, step_um=step_um)
    return AxialRecording(frames=np.stack(frames), z_um=np.array(zs), t_perf=np.array(ts), meta=meta)


def move_lens_from_below(lens, z_um: float) -> None:
    """Approach from the side the axial scans step in, so the z of a frame can be returned to."""
    lens.move_abs(z_um - LENS_PRELOAD_UM)
    lens.move_abs(z_um)


def open_sample_path(shutters) -> None:
    """Sample mode, as in the human interface: reference closed, objective and sample open."""
    shutters.change_to_objective()
    shutters.sample.open()


def run_experiment(
    dev: Devices,
    cfg: RunConfig,
    run_dir: Path,
    stop: threading.Event,
    *,
    on_status: Callable[[str], None] = lambda s: None,
    on_trial: Callable[[dict, object], None] = lambda row, rec: None,
    on_frame: Optional[Callable[[np.ndarray, float], None]] = None,
) -> dict:
    """Runs all trials. Returns counts. Device errors end the run and propagate; misses do not."""
    cfg.validate()
    trials = cfg.trials()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "plan.json", "w") as fh:
        json.dump(cfg.plan_dict(), fh, indent=2, default=float)
    log_path = run_dir / "run_log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=LOG_FIELDS).writeheader()

    dev.daq.configure(sample_rate_hz=cfg.sample_rate_hz)
    dev.camera.set_exposure_time(cfg.exposure_s)
    dev.camera.set_emccd_gain(cfg.gain)
    if dev.shutters is not None:
        open_sample_path(dev.shutters)

    params = cfg.finder_params()
    counts = dict(n_done=0, n_found=0, n_missed=0, n_axial=0)
    r_now: Optional[float] = None
    try:
        for t in trials:
            if stop.is_set():
                break
            on_status(f"Trial {t.index + 1}/{len(trials)}: block {t.block}, R = {t.r_mm:.2f} mm, {t.kind}")
            z_pred = stage_ref_lens_um(cfg.center, cfg.edge, t.r_mm)
            if r_now != t.r_mm:
                # lens back first: a deep lens position at the edge is inside the phantom at the apex
                z_safe = min(float(dev.lens.get_position()), z_pred - cfg.half_window_um)
                dev.lens.move_abs(z_safe)
                move_stage_xy(dev.stage, *stage_ref_xy_um(cfg.center, cfg.edge, t.r_mm))
                r_now = t.r_mm
            sx, sy, _ = dev.stage.get_position()
            meta = dict(experiment=run_dir.name, trial=t.index + 1, block=t.block, r_target_mm=t.r_mm, kind=t.kind,
                        lens_pred_um=z_pred, stage_x_um=sx, stage_y_um=sy, seed=cfg.seed)
            row = dict(trial=t.index + 1, block=t.block, r_mm=t.r_mm, kind=t.kind, stage_x_um=sx, stage_y_um=sy,
                       lens_pred_um=z_pred, t_perf=time.perf_counter(),
                       time=datetime.now().isoformat(timespec="seconds"), status="ok")
            name = f"{t.index + 1:03d}_b{t.block}_r{t.r_mm:.2f}_{t.kind}.h5"
            if t.kind == "axial":
                try:
                    rec = run_axial(dev, cfg.axial_before_um, cfg.axial_after_um, cfg.axial_step_um, z_pred, meta, stop,
                                    on_frame)
                except Cancelled:
                    break
                save_axial(rec, run_dir / name)
                row.update(axial_n_frames=rec.frames.shape[0], axial_peak_z_um=float(rec.z_um[np.argmax(rec.frame_sums())]))
                counts["n_axial"] += 1
            else:
                direction = 1.0 if t.kind == "find_fwd" else -1.0
                rec = run_find(dev, params, z_pred, cfg.half_window_um, direction, meta, stop)
                if rec is None:
                    row["status"] = "no data"
                else:
                    rec.meta["stopped_early"] = stop.is_set()
                    save_recording(rec, run_dir / name)
                    m = rec.meta
                    row.update(finder_found=m["finder_found"], finder_z_um=m["finder_z_um"], finder_peak=m["finder_peak"],
                               finder_stop_z_um=m["finder_stop_z_um"])
                    counts["n_found" if m["finder_found"] else "n_missed"] += 1
            row["file"] = name
            with open(log_path, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=LOG_FIELDS).writerow(row)
            counts["n_done"] += 1
            on_trial(row, rec)
    finally:
        # back out of the phantom, return to the center
        try:
            dev.lens.move_abs(cfg.center.lens_um - cfg.half_window_um)
            move_stage_xy(dev.stage, cfg.center.stage_x_um, cfg.center.stage_y_um)
            dev.lens.move_abs(cfg.center.lens_um)
        finally:
            if cfg.close_shutters_at_end and dev.shutters is not None:
                dev.shutters.close_all()
    return counts
