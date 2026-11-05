"""
Raw Andor axial scan, .h5 (plain h5py, readable in HDFView / MATLAB):
    /                attrs: format, version, meta_json
    /frames          (n, ny, nx) counts
    /z_um            (n,) lens position read back after each move
    /t_perf          (n,) perf_counter time at the start of each frame
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

H5_FORMAT = "andor_axial_recording"


@dataclass
class AxialRecording:
    frames: np.ndarray
    z_um: np.ndarray
    t_perf: np.ndarray
    meta: dict = field(default_factory=dict)

    def frame_sums(self) -> np.ndarray:
        return np.sum(self.frames, axis=(1, 2))


def save_axial(rec: AxialRecording, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = H5_FORMAT
        f.attrs["version"] = 1
        f.attrs["meta_json"] = json.dumps(rec.meta, default=float)
        f.create_dataset("frames", data=rec.frames, compression="gzip", compression_opts=4)
        f.create_dataset("z_um", data=rec.z_um)
        f.create_dataset("t_perf", data=rec.t_perf)
    return path


def load_axial(path: str | Path) -> AxialRecording:
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != H5_FORMAT:
            raise ValueError(f"{path} is not an {H5_FORMAT} file")
        return AxialRecording(frames=f["frames"][()], z_um=f["z_um"][()], t_perf=f["t_perf"][()],
                              meta=json.loads(f.attrs["meta_json"]))
