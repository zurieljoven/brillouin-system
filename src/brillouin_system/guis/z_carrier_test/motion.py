"""
Lens Z and stage Z behind one interface, so a slew or an Andor sweep runs the same way on either axis.

A slew is an absolute move at a set velocity that the Zaber ends by itself, so a hung PC cannot carry an
axis past the commanded target. Every command is also checked against a window around the start position
(set_window), so nothing in this test moves more than a few hundred um from where the plane was parked.
Positions are logged by polling on the perf_counter timeline, like the eye-lens position log.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class WindowError(RuntimeError):
    pass


class AxisMover:
    """Base: window check and position log. Subclasses implement _get, _move_abs, _slew_to, _wait_idle, stop."""

    def __init__(self, name: str):
        self.name = name
        self.window: Optional[tuple[float, float]] = None
        self._log_on = False
        self._log_thread: Optional[threading.Thread] = None
        self._log_t: list[float] = []
        self._log_z: list[float] = []

    # ---- window -------------------------------------------------------------------------------------------

    def set_window(self, lo_um: float, hi_um: float) -> None:
        if not lo_um < hi_um:
            raise ValueError("The window needs lo < hi")
        self.window = (float(lo_um), float(hi_um))

    def _check(self, target_um: float) -> None:
        if self.window is None:
            raise WindowError(f"{self.name}: no motion window set")
        lo, hi = self.window
        if not lo - 1e-6 <= target_um <= hi + 1e-6:
            raise WindowError(f"{self.name}: target {target_um:.1f} um is outside {lo:.1f}..{hi:.1f} um")

    # ---- motion -------------------------------------------------------------------------------------------

    def get_position(self) -> float:
        return float(self._get())

    def move_abs(self, z_um: float) -> None:
        """Blocking move at the axis' default speed."""
        self._check(float(z_um))
        self._move_abs(float(z_um))

    def slew_to(self, z_um: float, speed_um_s: float) -> None:
        """Non-blocking move to z_um at speed_um_s; the controller stops there by itself."""
        self._check(float(z_um))
        if speed_um_s <= 0:
            raise ValueError("Speed must be positive")
        self._slew_to(float(z_um), float(speed_um_s))

    def wait_idle(self, timeout_s: float = 10.0) -> None:
        self._wait_idle(float(timeout_s))

    def stop(self) -> None:
        raise NotImplementedError

    # ---- position log -------------------------------------------------------------------------------------

    def start_log(self, poll_s: float = 0.016, alpha: float = 0.25) -> None:
        if self._log_on:
            raise RuntimeError("Position log already running")
        self._log_t, self._log_z = [], []
        self._log_on = True

        def loop():
            nxt = time.perf_counter()
            while self._log_on:
                now = time.perf_counter()
                if now < nxt:
                    time.sleep(nxt - now)
                    continue
                nxt += poll_s
                t0 = time.perf_counter()
                z = self._get()
                t1 = time.perf_counter()
                self._log_t.append(t0 + alpha * (t1 - t0))
                self._log_z.append(float(z))

        self._log_thread = threading.Thread(target=loop, name=f"{self.name}PosLog", daemon=True)
        self._log_thread.start()

    def stop_log(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._log_on:
            return np.empty(0), np.empty(0)
        self._log_on = False
        if self._log_thread is not None:
            self._log_thread.join(timeout=2.0)
        t, z = np.asarray(self._log_t, float), np.asarray(self._log_z, float)
        if t.size >= 2:
            keep = np.concatenate(([True], np.diff(t) > 0))
            t, z = t[keep], z[keep]
        return t, z

    # ---- subclass hooks -----------------------------------------------------------------------------------

    def _get(self) -> float:
        raise NotImplementedError

    def _move_abs(self, z_um: float) -> None:
        raise NotImplementedError

    def _slew_to(self, z_um: float, speed_um_s: float) -> None:
        raise NotImplementedError

    def _wait_idle(self, timeout_s: float) -> None:
        raise NotImplementedError


class ZaberAxisMover(AxisMover):
    """A zaber_motion Axis: ZaberEyeLens.axis for the lens, ZaberHumanInterface.z_axis for the stage."""

    def __init__(self, name: str, axis):
        super().__init__(name)
        self.axis = axis

    def _get(self) -> float:
        from zaber_motion import Units
        return self.axis.get_position(Units.LENGTH_MICROMETRES)

    def _move_abs(self, z_um: float) -> None:
        from zaber_motion import Units
        self.axis.move_absolute(z_um, Units.LENGTH_MICROMETRES)

    def _slew_to(self, z_um: float, speed_um_s: float) -> None:
        from zaber_motion import Units
        self.axis.move_absolute(z_um, Units.LENGTH_MICROMETRES, wait_until_idle=False, velocity=speed_um_s,
                                velocity_unit=Units.VELOCITY_MICROMETRES_PER_SECOND)

    def _wait_idle(self, timeout_s: float) -> None:
        self.axis.wait_until_idle()

    def stop(self) -> None:
        self.axis.stop()


class SimAxisMover(AxisMover):
    """SimZaberLens (patient_movement_analysis.simulated_devices) as an axis, for --dummy and the tests."""

    def __init__(self, name: str, sim):
        super().__init__(name)
        self.sim = sim

    def _get(self) -> float:
        return self.sim.get_position()

    def _move_abs(self, z_um: float) -> None:
        self.sim.move_abs(z_um)

    def _slew_to(self, z_um: float, speed_um_s: float) -> None:
        d = z_um - self.sim.z_at(time.perf_counter())
        if abs(d) > 1e-9:
            self.sim.start_slewing_guarded(speed_um_s if d > 0 else -speed_um_s, abs(d))

    def _wait_idle(self, timeout_s: float) -> None:
        t_end = time.perf_counter() + timeout_s
        while self.sim._slew_guard_active and time.perf_counter() < t_end:
            time.sleep(0.01)

    def stop(self) -> None:
        self.sim.stop_slewing()
