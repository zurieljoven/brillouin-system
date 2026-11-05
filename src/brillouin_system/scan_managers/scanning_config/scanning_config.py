from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli
import tomli_w

from brillouin_system.configs import CONFIG_DIR
from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig, ThreadSafeConfig


@dataclass
class ScanningConfig:
    # ----------------------------
    # Reflection Finder
    # ----------------------------
    ni_sample_rate_hz: float = 1000.0
    speed_um_s: float = 5000.0
    max_distance_um: float = 5000.0
    threshold_high_n_sigma: int = 10
    threshold_low_n_sigma: int = 4
    bg_acqui_s: float = 0.1
    debounce_s: float = 0.020
    z_poll_s: float = 0.016
    alpha: float = 0.25
    chunk_size: int = 1024
    idle_sleep_s: float = 0.001
    z_offset_um: float = 0.0
    min_samples_above: int = 3
    # ----------------------------
    # Homodyne finder (human interface GUI). z_offset_um above is shared. The
    # defaults are from the replay on 320 slews (2026-10-09, 2026-11-02), all at
    # 400 um/s: other speeds are untested until the phantom speed block is run.
    # ----------------------------
    homodyne_sample_rate_hz: float = 5000.0   # per channel
    homodyne_speed_um_s: float = 400.0
    homodyne_max_distance_um: float = 1500.0
    homodyne_n_sigma: float = 20.0
    homodyne_stop_after_um: float = 150.0
    homodyne_band_lo_hz: float = 100.0
    homodyne_band_hi_hz: float = 2400.0
    homodyne_rms_window_s: float = 0.005
    homodyne_chunk_size: int = 100


AXIAL_SCANNING_TOML_PATH = CONFIG_DIR / "scanning_config.toml"


# old-name shims for backward compatibility
_LEGACY_KEY_MAP = {
    "n_sigma": "threshold_high_n_sigma",
    "max_search_distance_um": "max_distance_um",
    "background_acquisition_time_ms": "bg_acqui_s",
}


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a TOML section dict into ScanningConfig kwargs.

    - Filters unknown keys.
    - Includes backward-compat shims for older key names.
    - Converts old millisecond background acquisition setting to seconds.
    """
    raw = dict(raw)

    if "threshold_high_n_sigma" not in raw and "n_sigma" in raw:
        raw["threshold_high_n_sigma"] = raw["n_sigma"]

    if "max_distance_um" not in raw and "max_search_distance_um" in raw:
        raw["max_distance_um"] = raw["max_search_distance_um"]

    if "bg_acqui_s" not in raw and "background_acquisition_time_ms" in raw:
        raw["bg_acqui_s"] = float(raw["background_acquisition_time_ms"]) / 1000.0

    allowed = set(ScanningConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}



def _dataclass_to_toml_dict(cfg: ScanningConfig) -> dict[str, Any]:
    return {
        "ni_sample_rate_hz": float(cfg.ni_sample_rate_hz),
        "speed_um_s": float(cfg.speed_um_s),
        "max_distance_um": float(cfg.max_distance_um),
        "threshold_high_n_sigma": int(cfg.threshold_high_n_sigma),
        "threshold_low_n_sigma": int(cfg.threshold_low_n_sigma),
        "bg_acqui_s": float(cfg.bg_acqui_s),
        "debounce_s": float(cfg.debounce_s),
        "z_poll_s": float(cfg.z_poll_s),
        "alpha": float(cfg.alpha),
        "chunk_size": int(cfg.chunk_size),
        "idle_sleep_s": float(cfg.idle_sleep_s),
        "z_offset_um": float(cfg.z_offset_um),
        "min_samples_above": int(cfg.min_samples_above),
        "homodyne_sample_rate_hz": float(cfg.homodyne_sample_rate_hz),
        "homodyne_speed_um_s": float(cfg.homodyne_speed_um_s),
        "homodyne_max_distance_um": float(cfg.homodyne_max_distance_um),
        "homodyne_n_sigma": float(cfg.homodyne_n_sigma),
        "homodyne_stop_after_um": float(cfg.homodyne_stop_after_um),
        "homodyne_band_lo_hz": float(cfg.homodyne_band_lo_hz),
        "homodyne_band_hi_hz": float(cfg.homodyne_band_hi_hz),
        "homodyne_rms_window_s": float(cfg.homodyne_rms_window_s),
        "homodyne_chunk_size": int(cfg.homodyne_chunk_size),
    }



def load_axial_scanning_config(path: Path, section: str = "axial_scanning") -> ScanningConfig:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}

    return ScanningConfig(**_toml_to_kwargs(raw))



def save_config_section(path: Path, section: str, config: ThreadSafeConfig) -> None:
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}

    cfg: ScanningConfig = config.get_raw()
    data[section] = _dataclass_to_toml_dict(cfg)

    with path.open("wb") as f:
        tomli_w.dump(data, f)


axial_scanning_config = LazyThreadSafeConfig(lambda: 
    load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH, "axial_scanning")
)
