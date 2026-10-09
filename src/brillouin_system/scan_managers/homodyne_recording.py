"""
Raw homodyne DAQ recordings: container + .h5 (default) / .npz save/load.

A recording holds the raw voltages of every DAQ channel on one shared
hardware sample clock, the perf_counter anchor of sample 0, and (for
slews) the Zaber position log on the same perf_counter timeline, so z can
be assigned to every sample afterwards. Settings go into `meta`
(JSON-serializable) so a file is self-describing.

HDF5 layout (plain h5py, readable in HDFView / MATLAB / h5py):
    /                 attrs: format, version, mode, sample_rate_hz, t0_perf,
                             channels, meta_json, plus every scalar meta entry
    /daq/values       (n_ch, n) raw volts, attrs: units, channels
    /daq/t_s          (n,) seconds since the first sample
    /daq/z_um         (n,) z per sample (slews with a position log only)
    /zaber/t_perf     (m,) Zaber position-log timestamps (perf_counter)
    /zaber/z_um       (m,) Zaber position-log positions
MATLAB's h5read returns /daq/values transposed, as (n, n_ch).

Load for analysis:
    from brillouin_system.scan_managers.homodyne_recording import load_recording
    rec = load_recording("..._slew.h5")
    rec.values      # (n_ch, n) volts
    rec.t_s()       # seconds since the first sample
    rec.z_um()      # z per sample (slew only; None for stationary)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

H5_FORMAT = "homodyne_recording"
H5_VERSION = 1
H5_SUFFIXES = (".h5", ".hdf5")


@dataclass
class HomodyneRecording:
    mode: str                       # "stationary" | "slew"
    values: np.ndarray              # (n_ch, n) volts
    sample_rate_hz: float           # per channel
    t0_perf: float                  # perf_counter time of sample 0 (best effort)
    channels: tuple[str, ...]       # e.g. ("ai0", "ai1")
    meta: dict = field(default_factory=dict)
    zlog_t_perf: np.ndarray = field(default_factory=lambda: np.empty(0))
    zlog_z_um: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.values).shape[-1])

    def t_s(self) -> np.ndarray:
        """Seconds since the first sample."""
        return np.arange(self.n_samples, dtype=np.float64) / float(self.sample_rate_hz)

    def t_perf(self) -> np.ndarray:
        """perf_counter time of every sample."""
        return float(self.t0_perf) + self.t_s()

    def z_um(self) -> Optional[np.ndarray]:
        """z of every sample, interpolated from the Zaber log (clamped at the
        log's ends). None if no position log was recorded."""
        t = np.asarray(self.zlog_t_perf, dtype=np.float64)
        z = np.asarray(self.zlog_z_um, dtype=np.float64)
        if t.size < 2:
            return None
        keep = np.concatenate(([True], np.diff(t) > 0))
        return np.interp(self.t_perf(), t[keep], z[keep])


def save_recording(rec: HomodyneRecording, path: str | Path) -> Path:
    """Save as HDF5 (.h5/.hdf5) or NumPy (.npz) by suffix; any other suffix
    (or none) becomes .h5. Returns the path written."""
    path = Path(path)
    if path.suffix.lower() not in H5_SUFFIXES + (".npz",):
        path = path.with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        _save_npz(rec, path)
    else:
        _save_h5(rec, path)
    return path


def load_recording(path: str | Path) -> HomodyneRecording:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return _load_npz(path)
    return _load_h5(path)


def _save_h5(rec: HomodyneRecording, path: Path) -> None:
    values = np.asarray(rec.values, dtype=np.float64)
    channels = [str(c) for c in rec.channels]
    with h5py.File(path, "w") as f:
        f.attrs["format"] = H5_FORMAT
        f.attrs["version"] = H5_VERSION
        f.attrs["mode"] = rec.mode
        f.attrs["sample_rate_hz"] = float(rec.sample_rate_hz)
        f.attrs["t0_perf"] = float(rec.t0_perf)
        f.attrs["channels"] = channels
        f.attrs["meta_json"] = json.dumps(rec.meta, default=float)
        for k, v in rec.meta.items():   # scalar settings, browsable in HDFView
            if isinstance(v, (bool, int, float, str)) and k not in f.attrs:
                f.attrs[k] = v

        daq = f.create_group("daq")
        ds = daq.create_dataset("values", data=values, compression="gzip")
        ds.attrs["units"] = "V"
        ds.attrs["channels"] = channels
        daq.create_dataset("t_s", data=rec.t_s(), compression="gzip").attrs["units"] = "s"
        z = rec.z_um()
        if z is not None:
            daq.create_dataset("z_um", data=z, compression="gzip").attrs["units"] = "um"

        zab = f.create_group("zaber")
        zab.create_dataset("t_perf", data=np.asarray(rec.zlog_t_perf, dtype=np.float64)).attrs["units"] = "s"
        zab.create_dataset("z_um", data=np.asarray(rec.zlog_z_um, dtype=np.float64)).attrs["units"] = "um"


def _load_h5(path: Path) -> HomodyneRecording:
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != H5_FORMAT:
            raise ValueError(f"{path.name} is not a homodyne recording")
        return HomodyneRecording(
            mode=str(f.attrs["mode"]),
            values=np.array(f["daq/values"]),
            sample_rate_hz=float(f.attrs["sample_rate_hz"]),
            t0_perf=float(f.attrs["t0_perf"]),
            channels=tuple(str(c) for c in f.attrs["channels"]),
            meta=json.loads(str(f.attrs["meta_json"])),
            zlog_t_perf=np.array(f["zaber/t_perf"]),
            zlog_z_um=np.array(f["zaber/z_um"]),
        )


def _save_npz(rec: HomodyneRecording, path: Path) -> None:
    np.savez_compressed(
        path,
        mode=np.array(rec.mode),
        values=np.asarray(rec.values, dtype=np.float64),
        sample_rate_hz=np.array(float(rec.sample_rate_hz)),
        t0_perf=np.array(float(rec.t0_perf)),
        channels=np.array(list(rec.channels)),
        meta_json=np.array(json.dumps(rec.meta, default=float)),
        zlog_t_perf=np.asarray(rec.zlog_t_perf, dtype=np.float64),
        zlog_z_um=np.asarray(rec.zlog_z_um, dtype=np.float64),
    )


def _load_npz(path: Path) -> HomodyneRecording:
    with np.load(path, allow_pickle=False) as d:
        return HomodyneRecording(
            mode=str(d["mode"]),
            values=np.array(d["values"]),
            sample_rate_hz=float(d["sample_rate_hz"]),
            t0_perf=float(d["t0_perf"]),
            channels=tuple(str(c) for c in d["channels"]),
            meta=json.loads(str(d["meta_json"])),
            zlog_t_perf=np.array(d["zlog_t_perf"]),
            zlog_z_um=np.array(d["zlog_z_um"]),
        )
