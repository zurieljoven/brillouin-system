"""
Simulated two-channel homodyne DAQ for the homodyne test GUI (dummy mode).

Duck-types NI6008Multi (configure / validate / effective_range_v / acquire)
and couples the signal to a SimZaberLens, so slews produce real fringes at
2 v / lambda under a Gaussian surface envelope:

    V_i = DC_i + A_i(z) cos(4 pi z / lambda + phi_i(t)) + noise, quantized

with a slowly rotating signal polarization (the H/V split drifts), a random
phase walk, and small axial micro-vibration, so a stationary record also
shows some fringe activity at the plane.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional

import numpy as np

from brillouin_system.devices.ni.ni6008_multi import NI6008Multi, NIMultiReadResult, effective_range_v


class SimHomodyneDAQ:

    def __init__(
        self,
        zaber,
        *,
        plane_um: float,
        wavelength_nm: float = 780.24,
        lo_dc_v: tuple[float, float] = (0.80, 0.65),
        peak_amplitude_v: float = 0.05,
        peak_sigma_um: float = 3.8,
        noise_std_v: float = 0.0003,
        seed: Optional[int] = None,
    ):
        self.zaber = zaber
        self.plane_um = float(plane_um)
        self.wavelength_um = float(wavelength_nm) * 1e-3
        self.lo_dc_v = tuple(float(x) for x in lo_dc_v)
        self.peak_amplitude_v = float(peak_amplitude_v)
        self.peak_sigma_um = float(peak_sigma_um)
        self.noise_std_v = float(noise_std_v)
        self._rng = np.random.default_rng(seed)
        self._cfg = NI6008Multi(device="SimDev")  # holds + validates settings

    # ---- NI6008Multi-compatible settings API ----
    @property
    def channels(self):
        return self._cfg.channels

    @property
    def sample_rate_hz(self):
        return self._cfg.sample_rate_hz

    @property
    def terminal(self):
        return self._cfg.terminal

    @property
    def effective_range_v(self) -> float:
        return effective_range_v(self._cfg.terminal, self._cfg.v_range)

    def configure(self, **kw) -> None:
        self._cfg.configure(**kw)

    def validate(self) -> None:
        self._cfg.validate()

    # ---- signal model ----
    def _generate(self, t_perf: np.ndarray, phase_walk: np.ndarray) -> np.ndarray:
        n_ch = len(self.channels)
        z = self.zaber.z_at_array(t_perf)
        z = z + 0.03 * np.sin(2 * math.pi * 37.0 * t_perf)          # micro-vibration
        dz = z - self.plane_um
        env = self.peak_amplitude_v * np.exp(-dz ** 2 / (4.0 * self.peak_sigma_um ** 2))
        theta = 0.6 + 0.5 * np.sin(2 * math.pi * t_perf / 25.0)      # signal polarization angle
        split = (np.cos(theta), np.sin(theta))
        phase = 4.0 * math.pi * z / self.wavelength_um + phase_walk

        out = np.empty((n_ch, t_perf.size))
        for i in range(n_ch):
            dc = self.lo_dc_v[i % len(self.lo_dc_v)]
            amp = env * math.sqrt(dc / self.lo_dc_v[0]) * split[i % 2]
            out[i] = dc + amp * np.cos(phase + 1.3 * i) + self._rng.normal(0, self.noise_std_v, t_perf.size)

        rng = self.effective_range_v
        bits = 11 if self.terminal == "RSE" else 12                   # USB-6008 resolution
        q = 2.0 * rng / 2 ** bits
        return np.clip(np.round(out / q) * q, -rng, rng)

    def acquire(
        self,
        duration_s: float,
        *,
        on_chunk: Optional[Callable[[int], None]] = None,
        stop_evt: Optional[threading.Event] = None,
        chunk_size: int = 500,
        **_,
    ) -> NIMultiReadResult:
        self.validate()
        fs = self.sample_rate_hz
        n_ch = len(self.channels)
        n_target = max(1, int(round(float(duration_s) * fs)))
        buf = np.empty((n_ch, n_target))
        t0 = time.perf_counter()
        walk = 0.0
        n = 0
        while n < n_target:
            if stop_evt is not None and stop_evt.is_set():
                break
            avail = int((time.perf_counter() - t0) * fs) - n          # real-time pacing
            if avail <= 0:
                time.sleep(0.002)
                continue
            want = min(avail, chunk_size, n_target - n)
            t = t0 + np.arange(n, n + want) / fs
            steps = self._rng.normal(0.0, 0.002, want)               # phase random walk
            phase_walk = walk + np.cumsum(steps)
            walk = float(phase_walk[-1])
            buf[:, n:n + want] = self._generate(t, phase_walk)
            n += want
            if on_chunk is not None:
                on_chunk(n)
        return NIMultiReadResult(values=buf[:, :n].copy(), sample_rate_hz=fs,
                                 t0_perf=t0, channels=self.channels)
