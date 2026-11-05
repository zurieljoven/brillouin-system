"""Dummy Andor for this experiment: frame brightness follows the lens-plane distance."""

from __future__ import annotations

import time

import numpy as np


class SimAndor:

    def __init__(self, lens, *, plane_um: float, sigma_um: float = 25.0, seed: int | None = None):
        self.lens = lens
        self.plane_um = float(plane_um)
        self.sigma_um = float(sigma_um)
        self._rng = np.random.default_rng(seed)
        self._exposure = 0.2
        self._gain = 0

    def set_exposure_time(self, seconds: float) -> None:
        self._exposure = float(seconds)

    def get_exposure_time(self) -> float:
        return self._exposure

    def set_emccd_gain(self, gain) -> None:
        self._gain = int(gain)

    def get_emccd_gain(self) -> int:
        return self._gain

    def get_roi(self) -> tuple[int, int, int, int]:
        return (0, 200, 298, 325)

    def get_binning(self) -> tuple[int, int]:
        return (1, 1)

    def snap(self) -> np.ndarray:
        time.sleep(self._exposure)
        dz = float(self.lens.get_position()) - self.plane_um
        x = np.arange(200)
        line = 800.0 * np.exp(-0.5 * (dz / self.sigma_um) ** 2) * np.exp(-0.5 * ((x - 100.0) / 5.0) ** 2)
        return self._rng.poisson(100.0 + line[None, :].repeat(27, axis=0))

    def close(self) -> None:
        pass
