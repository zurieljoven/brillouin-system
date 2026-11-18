"""
Validation recordings shared by the human interface and the homodyne test GUI (no Qt).

  save_eye_snapshot    the stereo eye-tracker frames and the result a move or find was based on (.npz)
  save_finder_traces   the raw H/V homodyne traces and Zaber log of every live find in one attempt (.h5)
  SweepAttempt         what one sweep-scan attempt did (every find, the outcome and why), so failed attempts,
                       which are never registered as scans, can still be saved and counted
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class SweepAttempt:
    """One sweep-scan attempt. finds: (label, ReflectionResult) in the order they ran, e.g. "bootstrap",
    "c03_in", "c03_out". reason: "" on success, else why the attempt failed."""
    scan_id: str
    success: bool = False
    reason: str = ""
    finds: list = field(default_factory=list)
    n_frames: int = 0
    best_motion_um: Optional[float] = None
    scan: Any = None            # the registered AxialScan on success


def _jsonable(o):
    if is_dataclass(o):
        return asdict(o)
    a = np.asarray(o)
    return a.tolist() if a.dtype != object else repr(o)


def save_eye_snapshot(path: str | Path, res, **extra) -> Path:
    """res: EyeTrackerResults (left_img, right_img, time_stamp, laser_position, delta_laser_corner, pupil3d).
    extra: scalars to keep alongside (e.g. lens_um, step)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pupil = getattr(res, "pupil3d", None)
    try:
        pupil_json = json.dumps(asdict(pupil), default=_jsonable) if pupil is not None else ""
    except TypeError:
        pupil_json = repr(pupil)
    lp = res.laser_position
    dc = res.delta_laser_corner
    np.savez_compressed(
        path, left=np.asarray(res.left_img), right=np.asarray(res.right_img), time_stamp=float(res.time_stamp),
        laser_position=np.asarray(lp if lp is not None else [np.nan] * 3, float),
        delta_c_mm=float(dc) if dc is not None else np.nan, pupil3d_json=pupil_json,
        **{k: (np.nan if v is None else v) for k, v in extra.items()})
    return path


_TRACE_ARRAYS = ("daq_values", "daq_ts", "zaber_lens_ts", "zaber_lens_z_um")


def save_finder_traces(path: str | Path, finds, **attrs) -> Path:
    """finds: iterable of (label, ReflectionResult). One group per find with its raw arrays and every scalar
    field as attributes; attrs go on the file root. Strings and numbers only (None is skipped)."""
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        for k, v in attrs.items():
            if v is not None:
                h.attrs[k] = v
        for i, (label, r) in enumerate(finds):
            g = h.create_group(f"{i:02d}_{label}")
            for f in fields(r):
                v = getattr(r, f.name)
                if v is None:
                    continue
                if f.name in _TRACE_ARRAYS:
                    g.create_dataset(f.name, data=np.asarray(v), compression="gzip")
                elif isinstance(v, (bool, int, float, str, np.number, np.bool_)):
                    g.attrs[f.name] = v
    return path
