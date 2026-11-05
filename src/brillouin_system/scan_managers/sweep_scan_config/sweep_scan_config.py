"""
Config for the in-out sweep scan (repeated find-measure-find cycles).

Same pattern as scan_managers/scanning_config: dataclass + TOML + a global
ThreadSafeConfig instance. Detection and motion parameters (speed, thresholds,
sample rate, min_samples_above, ...) are NOT duplicated here — the sweep scan
takes those from the shared axial ScanningConfig, so each crossing behaves
identically to the GUI reflection finder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli
import tomli_w

from brillouin_system.configs import CONFIG_DIR
from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig, ThreadSafeConfig


@dataclass
class SweepScanConfig:
    # ----------------------------
    # In-out sweep scan
    # ----------------------------
    n_repeats: int = 12               # number of in-out cycles
    approach_um: float = 300.0        # run-up distance past the plane on each side
    target_depth_um: float = 50.0     # measure at in-crossing + this (positive = inward)
    settle_s: float = 0.05            # lens settle time before the camera snap
    # Max allowable wall-clock time for one sweep scan [s]. When elapsed time
    # reaches this, the scan stops before the next cycle (and once mid-cycle,
    # before the long out-search), then saves whatever cycles were completed.
    # 0 (or negative) disables the limit.
    max_time_s: float = 0.0
    # In-crossing gate: |in - previous plane estimate|. Must exceed the real
    # eye motion between cycles (~1.5 s), so it stays loose.
    plausibility_gate_um: float = 750.0
    # Out-crossing gate: |out - THIS cycle's in-crossing|. The two crossings are
    # ~1 s apart, so real motion is small (7-27 tracks: 90th pct 49 um, max 167)
    # and this can be far tighter than the in-crossing gate.
    out_gate_um: float = 200.0
    # Amplitude gate: reject a crossing whose peak is below this fraction of its
    # reference peak (initial find for in-crossings, same-cycle in-crossing for
    # out-crossings). Every false crossing seen on 2026-07-30 was a WEAK peak:
    # 0.12x the real one for the cuvette back wall, 0.006x for a finder outlier,
    # while genuine crossings held above 0.8x. Those are the old detector's numbers.
    # With the homodyne S, genuine crossings vary more: 0.15x at worst, 2-9 % below
    # 0.3x (320 slews, 2026-10-09 and 2026-11-02), so the gate moved to 0.15. The
    # cuvette back face is 0.17-0.30x and now passes this gate; the distance gates
    # still apply.
    min_peak_fraction: float = 0.15


SWEEP_SCAN_TOML_PATH = CONFIG_DIR / "sweep_scan_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(SweepScanConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: SweepScanConfig) -> dict[str, Any]:
    return {
        "n_repeats": int(cfg.n_repeats),
        "approach_um": float(cfg.approach_um),
        "target_depth_um": float(cfg.target_depth_um),
        "settle_s": float(cfg.settle_s),
        "max_time_s": float(cfg.max_time_s),
        "plausibility_gate_um": float(cfg.plausibility_gate_um),
        "out_gate_um": float(cfg.out_gate_um),
        "min_peak_fraction": float(cfg.min_peak_fraction),
    }


def load_sweep_scan_config(path: Path = SWEEP_SCAN_TOML_PATH, section: str = "sweep_scan") -> SweepScanConfig:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}
    return SweepScanConfig(**_toml_to_kwargs(raw))


def save_sweep_scan_config(config: ThreadSafeConfig, path: Path = SWEEP_SCAN_TOML_PATH, section: str = "sweep_scan") -> None:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}
    data[section] = _dataclass_to_toml_dict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)


sweep_scan_config = LazyThreadSafeConfig(lambda: load_sweep_scan_config())
