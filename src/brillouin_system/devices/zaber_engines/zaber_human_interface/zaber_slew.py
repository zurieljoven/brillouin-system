"""
Absolute move at a set speed that the Zaber controller ends by itself (no PC thread has to stop it).

Newer firmware takes the speed in the move command ("move abs <pos> <speed>"). Older firmware rejects that
with NotSupportedException; then the axis maxspeed setting is lowered to the slew speed for the move and must
be put back afterwards with the returned restore(), only once the axis is idle (restoring mid-move would
speed up the rest of the move).
"""

from __future__ import annotations

from typing import Callable


def _noop() -> None:
    pass


def move_abs_at_speed(axis, target_um: float, speed_um_s: float) -> Callable[[], None]:
    """Start the move without waiting. Returns restore(): call it after the axis has stopped (idempotent)."""
    from zaber_motion import Units
    um, ums = Units.LENGTH_MICROMETRES, Units.VELOCITY_MICROMETRES_PER_SECOND
    try:
        axis.move_absolute(float(target_um), um, wait_until_idle=False, velocity=abs(float(speed_um_s)),
                           velocity_unit=ums)
        return _noop
    except Exception as e:
        if type(e).__name__ != "NotSupportedException":
            raise
    old = axis.settings.get("maxspeed", ums)
    axis.settings.set("maxspeed", abs(float(speed_um_s)), ums)
    done = {"v": False}

    def restore() -> None:
        if not done["v"]:
            axis.settings.set("maxspeed", old, ums)
            done["v"] = True

    try:
        axis.move_absolute(float(target_um), um, wait_until_idle=False)
    except Exception:
        restore()
        raise
    return restore
