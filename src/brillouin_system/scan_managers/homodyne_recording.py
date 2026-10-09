"""
Raw homodyne DAQ recordings: container + .npz save/load.

A recording holds the raw voltages of every DAQ channel on one shared
hardware sample clock, the perf_counter anchor of sample 0, and (for
slews) the Zaber position log on the same perf_counter timeline, so z can
be assigned to every sample afterwards. Settings go into `meta`
(JSON-serializable) so a file is self-describing.

Load for analysis:
    from brillouin_system.scan_managers.homodyne_recording import load_recording
    rec = load_recording("..._slew.npz")
    rec.values      # (n_ch, n) volts
    rec.t_s()       # seconds since the first sample
    rec.z_um()      # z per sample (slew only; None for stationary)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


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
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return path


def load_recording(path: str | Path) -> HomodyneRecording:
    with np.load(Path(path), allow_pickle=False) as d:
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
