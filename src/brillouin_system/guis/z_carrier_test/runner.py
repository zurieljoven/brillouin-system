"""
The lens-Z vs stage-Z carrier test without Qt: homodyne DAQ, lens and stage Z (as AxisMovers), Andor, shutters.

Start with the lens parked on the plane (park_lens_on_plane does that from a lens slew). run_protocol then
records every step of ProtocolConfig.steps() into its own file and returns both axes to where they started
after every step. Nothing moves outside the window set on each axis at the start.

Files in the run folder:
    plan.json                       config, start positions, step list
    run_log.csv                     one row per step, written as it finishes
    NNN_bB_stationary.h5            homodyne recording, no motion
    NNN_bB_slew_<axis>_<dir>_vV.h5  homodyne recording; /zaber holds the MOVING axis' position log
    NNN_bB_andor_<axis>.h5          Andor frames stepped along that axis (andor_axial_recording format)
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

from brillouin_system.guis.plane_accuracy_test.axial_recording import AxialRecording, save_axial
from brillouin_system.guis.plane_accuracy_test.runner import open_sample_path
from brillouin_system.guis.z_carrier_test.motion import AxisMover
from brillouin_system.guis.z_carrier_test.protocol import (
    POST_ROLL_S, PRE_ROLL_S, PRELOAD_UM, ProtocolConfig, Step,
)
from brillouin_system.scan_managers.homodyne_plane_finder import HomodyneFinderParams, refine_plane
from brillouin_system.scan_managers.homodyne_recording import HomodyneRecording, save_recording

SETTLE_S = 0.2
FRAME_SETTLE_S = 0.05
DC_WARN_V = 8.5
ANDOR_SATURATION = 65000    # 16-bit counts
LOG_FIELDS = ["step", "block", "kind", "axis", "direction", "speed_um_s", "file", "status", "time", "t_perf",
              "lens_um", "stage_um", "dc_v", "peak_pos_um", "peak_rel_um", "peak_value"]


@dataclass
class Rig:
    daq: object
    lens: AxisMover
    stage: AxisMover
    camera: object
    shutters: object = None
    jog_lens: Callable[[float], None] = None     # manual moves, outside the run window (driver move_rel)
    jog_stage: Callable[[float], None] = None
    devices: tuple = ()             # driver objects to close at shutdown


class Cancelled(Exception):
    pass


def approach(mover: AxisMover, z_um: float) -> None:
    """Move to z_um from below, so backlash is taken up the same way every time."""
    mover.move_abs(z_um - PRELOAD_UM)
    mover.move_abs(z_um)


def s_peak(rec: HomodyneRecording, speed_um_s: float) -> tuple[Optional[float], float]:
    """Moving-axis position of the S^2-weighted plane (live-finder band and window), and the S maximum."""
    z = rec.z_um()
    if z is None:
        return None, float("nan")
    p = HomodyneFinderParams(speed_um_s=speed_um_s)
    rp = refine_plane(rec.values, rec.sample_rate_hz, p, i_min=int((PRE_ROLL_S + 0.1) * rec.sample_rate_hz))
    return float(np.interp(rp.index, np.arange(z.size), z)), rp.s_peak


def andor_peak(rec: AxialRecording) -> float:
    """Frame-sum peak position along the swept axis (3-point log parabola, as in the plane-accuracy analysis)."""
    s = rec.frame_sums().astype(float)
    k = int(np.clip(np.argmax(s), 1, s.size - 2))
    y = np.log(np.clip(s[k - 1:k + 2], 1.0, None))
    d = y[0] - 2.0 * y[1] + y[2]
    step = rec.z_um[k + 1] - rec.z_um[k]
    return float(rec.z_um[k] + (0.5 * (y[0] - y[2]) / d * step if d < 0 else 0.0))


def record_stationary(rig: Rig, duration_s: float, meta: dict, stop: threading.Event) -> HomodyneRecording:
    res = rig.daq.acquire(duration_s, stop_evt=stop)
    meta = dict(meta, duration_s=res.values.shape[-1] / res.sample_rate_hz, stopped_early=stop.is_set())
    return HomodyneRecording(mode="stationary", values=res.values, sample_rate_hz=res.sample_rate_hz,
                             t0_perf=res.t0_perf, channels=tuple(res.channels), meta=meta)


def record_slew(rig: Rig, mover: AxisMover, z0: float, half_um: float, direction: int, speed_um_s: float,
                meta: dict, stop: threading.Event) -> HomodyneRecording:
    """Slew mover from z0 - direction*half to z0 + direction*half at speed while recording, then back to z0."""
    start, end = z0 - direction * half_um, z0 + direction * half_um
    mover.move_abs(start)
    time.sleep(SETTLE_S)
    fs = float(rig.daq.sample_rate_hz)
    n_pre = int(PRE_ROLL_S * fs)
    duration = PRE_ROLL_S + 2.0 * half_um / speed_um_s + 0.2 + POST_ROLL_S
    state = {"log": False, "slew": False}

    def on_chunk(n: int) -> None:
        if not state["log"]:
            mover.start_log()
            state["log"] = True
        if not state["slew"] and n >= n_pre:
            mover.slew_to(end, speed_um_s)
            state["slew"] = True

    try:
        res = rig.daq.acquire(duration, on_chunk=on_chunk, stop_evt=stop)
        if stop.is_set():
            mover.stop()
        mover.wait_idle()
    finally:
        t, z = mover.stop_log() if state["log"] else (np.empty(0), np.empty(0))
        approach(mover, z0)
    meta = dict(meta, z0_um=z0, z_start_um=start, z_end_um=end, direction=direction, speed_um_s=speed_um_s,
                half_um=half_um, pre_roll_s=PRE_ROLL_S, post_roll_s=POST_ROLL_S, stopped_early=stop.is_set())
    return HomodyneRecording(mode="slew", values=res.values, sample_rate_hz=res.sample_rate_hz, t0_perf=res.t0_perf,
                             channels=tuple(res.channels), meta=meta, zlog_t_perf=t, zlog_z_um=z)


def andor_sweep(rig: Rig, mover: AxisMover, z0: float, half_um: float, step_um: float, meta: dict,
                stop: threading.Event, on_frame: Optional[Callable[[np.ndarray, float], None]] = None
                ) -> AxialRecording:
    """Andor frames stepping mover from z0 - half to z0 + half (always increasing), then back to z0."""
    n = int(round(2.0 * half_um / step_um)) + 1
    targets = z0 - half_um + step_um * np.arange(n)
    frames, zs, ts = [], [], []
    try:
        mover.move_abs(targets[0] - PRELOAD_UM)
        for z in targets:
            if stop.is_set():
                raise Cancelled()
            mover.move_abs(float(z))
            time.sleep(FRAME_SETTLE_S)
            zs.append(mover.get_position())
            ts.append(time.perf_counter())
            frames.append(np.asarray(rig.camera.snap()))
            if on_frame is not None:
                on_frame(frames[-1], zs[-1])
    finally:
        approach(mover, z0)
    cam = rig.camera
    meta = dict(meta, z0_um=z0, half_um=half_um, step_um=step_um, exposure_s=float(cam.get_exposure_time()),
                gain=float(cam.get_emccd_gain()), roi=[int(v) for v in cam.get_roi()],
                binning=[int(v) for v in cam.get_binning()])
    return AxialRecording(frames=np.stack(frames), z_um=np.array(zs), t_perf=np.array(ts), meta=meta)


def park_lens_on_plane(rig: Rig, half_um: float, speed_um_s: float, stop: threading.Event
                       ) -> tuple[Optional[float], HomodyneRecording]:
    """Lens slew through the current position; move the lens to the S plane if it was found."""
    z0 = rig.lens.get_position()
    rig.lens.set_window(z0 - half_um - PRELOAD_UM - 5.0, z0 + half_um + 5.0)
    rec = record_slew(rig, rig.lens, z0, half_um, +1, speed_um_s, {"kind": "park"}, stop)
    z_plane, peak = s_peak(rec, speed_um_s)
    if z_plane is not None and abs(z_plane - z0) < half_um - 20.0:
        approach(rig.lens, z_plane)
    else:
        z_plane = None
    return z_plane, rec


def run_protocol(rig: Rig, cfg: ProtocolConfig, run_dir: Path, stop: threading.Event, *,
                 on_status: Callable[[str], None] = lambda s: None,
                 on_result: Callable[[dict, object], None] = lambda row, rec: None,
                 on_frame: Optional[Callable[[np.ndarray, float], None]] = None) -> dict:
    """Every step of cfg.steps(). Device errors end the run and propagate; both axes go back to the start."""
    cfg.validate()
    steps = cfg.steps()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    lens0, stage0 = rig.lens.get_position(), rig.stage.get_position()
    w = cfg.window_half_um()
    rig.lens.set_window(lens0 - w, lens0 + w)
    rig.stage.set_window(stage0 - w, stage0 + w)
    movers = {"lens": rig.lens, "stage": rig.stage}
    start = {"lens": lens0, "stage": stage0}

    with open(run_dir / "plan.json", "w") as fh:
        json.dump(dict(cfg.as_dict(), lens_start_um=lens0, stage_start_um=stage0, window_half_um=w,
                       created=datetime.now().isoformat(timespec="seconds"), mode="z_carrier_test",
                       channels=list(rig.daq.channels), wavelength_nm=780.24,
                       steps=[asdict(s) for s in steps]), fh, indent=2, default=float)
    log_path = run_dir / "run_log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=LOG_FIELDS).writeheader()

    rig.daq.configure(sample_rate_hz=cfg.sample_rate_hz)
    rig.camera.set_exposure_time(cfg.exposure_s)
    rig.camera.set_emccd_gain(cfg.gain)
    if rig.shutters is not None:
        open_sample_path(rig.shutters)

    counts = dict(n_done=0, n_slew=0, n_andor=0, n_stationary=0)
    try:
        for s in steps:
            if stop.is_set():
                break
            label = s.kind if s.kind == "stationary" else f"{s.kind} {s.axis}" + (
                f" {'fwd' if s.direction > 0 else 'bwd'} {s.speed_um_s:.0f} um/s" if s.kind == "slew" else "")
            on_status(f"Step {s.index + 1}/{len(steps)} (block {s.block}): {label}")
            meta = dict(experiment=run_dir.name, sample=cfg.sample, step=s.index + 1, block=s.block, kind=s.kind,
                        axis=s.axis, lens_start_um=lens0, stage_start_um=stage0)
            row = dict(step=s.index + 1, block=s.block, kind=s.kind, axis=s.axis, direction=s.direction,
                       speed_um_s=s.speed_um_s, t_perf=time.perf_counter(), status="ok",
                       time=datetime.now().isoformat(timespec="seconds"),
                       lens_um=rig.lens.get_position(), stage_um=rig.stage.get_position())
            name = s.name() + ".h5"
            try:
                if s.kind == "andor":
                    rec = andor_sweep(rig, movers[s.axis], start[s.axis], cfg.andor_half_um, cfg.andor_step_um,
                                      meta, stop, on_frame)
                    save_axial(rec, run_dir / name)
                    if rec.frames.max() >= ANDOR_SATURATION:
                        on_status(f"WARNING: Andor frame reached {rec.frames.max():.0f} counts (saturated); "
                                  f"lower the exposure.")
                    z = andor_peak(rec)
                    row.update(peak_pos_um=z, peak_rel_um=z - start[s.axis], peak_value=float(rec.frame_sums().max()))
                    counts["n_andor"] += 1
                else:
                    if s.kind == "stationary":
                        rec = record_stationary(rig, cfg.stationary_s, meta, stop)
                        counts["n_stationary"] += 1
                    else:
                        rec = record_slew(rig, movers[s.axis], start[s.axis], cfg.slew_half_um, s.direction,
                                          s.speed_um_s, meta, stop)
                        z, peak = s_peak(rec, s.speed_um_s)
                        row.update(peak_pos_um=z, peak_rel_um=None if z is None else z - start[s.axis],
                                   peak_value=peak)
                        counts["n_slew"] += 1
                    dc = [float(np.median(v)) for v in rec.values]
                    row["dc_v"] = " ".join(f"{v:.3f}" for v in dc)
                    if max(dc) > DC_WARN_V:
                        on_status(f"WARNING: DC {max(dc):.2f} V is near the 10 V limit; lower the LO.")
                    save_recording(rec, run_dir / name)
            except Cancelled:
                break
            row["file"] = name
            with open(log_path, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=LOG_FIELDS).writerow(row)
            counts["n_done"] += 1
            on_result(row, rec)
    finally:
        try:
            approach(rig.lens, lens0)
            approach(rig.stage, stage0)
        finally:
            if cfg.close_shutters_at_end and rig.shutters is not None:
                rig.shutters.close_all()
    return counts
