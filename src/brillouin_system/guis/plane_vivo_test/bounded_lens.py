"""
Eye lens with a hard z ceiling, for use near a person. +z moves toward the eye.

Nothing can be commanded past the ceiling. A slew is a move to an absolute target
that the Zaber stops by itself, instead of the driver's velocity move stopped by a
thread on the PC, so a hung PC cannot carry the lens further.
"""

from __future__ import annotations

from typing import Optional


class LensLimitError(RuntimeError):
    pass


class BoundedLens:
    def __init__(self, lens):
        self._lens = lens
        self.ceiling_um: Optional[float] = None
        self._restore = None

    def __getattr__(self, name):
        return getattr(self._lens, name)    # get_position, the position log, stop_slewing, close

    def set_ceiling(self, ceiling_um: float) -> None:
        self.ceiling_um = float(ceiling_um)

    def _check(self, target_um: float) -> None:
        if self.ceiling_um is None:
            raise LensLimitError("Set the lens limit first.")
        # above the ceiling only moves back out are allowed
        if target_um > self.ceiling_um and target_um > float(self._lens.get_position()):
            raise LensLimitError(f"Lens target {target_um:.1f} um is past the limit {self.ceiling_um:.1f} um.")

    def move_abs(self, position_um: float) -> None:
        self._check(float(position_um))
        self._restore_speed()
        self._lens.move_abs(float(position_um))

    def move_rel(self, delta_um: float) -> None:
        self.move_abs(float(self._lens.get_position()) + float(delta_um))

    def room_um(self, direction: float, wanted_um: float) -> float:
        """Slew distance available from here, at most wanted_um."""
        if direction <= 0:
            return float(wanted_um)
        if self.ceiling_um is None:
            raise LensLimitError("Set the lens limit first.")
        return min(float(wanted_um), self.ceiling_um - float(self._lens.get_position()))

    def start_slewing_guarded(self, speed_um_per_s: float, max_distance_um: float) -> None:
        direction = 1.0 if speed_um_per_s > 0 else -1.0
        distance = self.room_um(direction, abs(max_distance_um))
        if distance <= 0:
            return
        axis = getattr(self._lens, "axis", None)
        if axis is None:    # simulated lens
            self._lens.start_slewing_guarded(speed_um_per_s, distance)
            return
        from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_slew import move_abs_at_speed
        self._restore_speed()
        self._restore = move_abs_at_speed(axis, float(self._lens.get_position()) + direction * distance,
                                          speed_um_per_s)

    def _restore_speed(self) -> None:
        """Older firmware: the slew lowered maxspeed; put it back (only once the axis has stopped)."""
        restore = getattr(self, "_restore", None)
        if restore is not None:
            restore()
            self._restore = None

    def stop_slewing(self) -> None:
        self._lens.stop_slewing()       # stops and waits until idle
        self._restore_speed()
