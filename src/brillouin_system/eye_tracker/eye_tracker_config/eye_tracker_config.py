# eye_tracker_config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import tomli
import tomli_w

from brillouin_system.configs import CONFIG_DIR
from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig, ThreadSafeConfig


@dataclass
class EyeTrackerConfig:
    """
    Minimal, robust knobs for fast pupil ellipse fitting.
    Keep it small so you rarely need to touch it.
    """

    # Per-eye thresholds
    binary_threshold_left: int = 20
    binary_threshold_right: int = 20

    # Flood-fill helper (fill N vertical dark pixels)
    fill_n_vetical_dark_pixels_left: int = 10
    fill_n_vetical_dark_pixels_right: int = 10

    # Fill N horizontal dark pixels (0 = off)
    fill_n_horizontal_dark_pixels_left: int = 0
    fill_n_horizontal_dark_pixels_right: int = 0

    # Masking
    masking_radius_left: int = 500
    masking_radius_right: int = 500
    masking_center_left: tuple[int, int] = (0, 0)
    masking_center_right: tuple[int, int] = (0, 0)

    # Ellipse fitting & overlay
    do_ellipse_fitting: bool = False
    overlay_ellipse: bool = False

    # Which image is returned by the pipeline
    frame_returned: Literal["original", "binary", "floodfilled", "contour"] = "original"


# Path to the TOML config file
PUPIL_FIT_TOML_PATH = CONFIG_DIR / "eye_tracker_config.toml"


def _toml_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Convert TOML section to EyeTrackerConfig kwargs.

    We filter out unknown keys so older TOML files still load cleanly even if
    we've removed or renamed some fields from the dataclass.
    """
    allowed = set(EyeTrackerConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _dataclass_to_toml_dict(cfg: EyeTrackerConfig) -> dict[str, Any]:
    """Convert an EyeTrackerConfig instance to a simple dict for TOML."""
    return asdict(cfg)


def load_eye_tracker_config(path: Path, section: str = "eye_tracker") -> EyeTrackerConfig:
    """
    Load EyeTrackerConfig from a TOML file section.
    If the section is missing, we fall back to the dataclass defaults.
    """
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
        raw = data.get(section, {})
    except FileNotFoundError:
        raw = {}

    kwargs = _toml_to_kwargs(raw)
    return EyeTrackerConfig(**kwargs)


def save_config_section(path: Path, section: str, config: ThreadSafeConfig) -> None:
    """
    Persist a ThreadSafeConfig[EyeTrackerConfig] section back to TOML.
    """
    try:
        with path.open("rb") as f:
            data = tomli.load(f)
    except FileNotFoundError:
        data = {}

    data[section] = _dataclass_to_toml_dict(config.get_raw())

    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Single global configuration instance, loaded from TOML at import time
eye_tracker_config = LazyThreadSafeConfig(lambda: 
    load_eye_tracker_config(PUPIL_FIT_TOML_PATH, "eye_tracker")
)
