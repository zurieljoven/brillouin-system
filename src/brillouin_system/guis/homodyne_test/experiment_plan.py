"""
Planning and move math for the homodyne test GUI (no Qt; unit-testable).

Eye-tracking moves use the same math, limits and signs as hi_frontend
(on_move_xy_polar_clicked / on_move_z_by_dc_clicked).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

MAX_XY_MOVE_UM = 3500.0
MAX_Z_MOVE_UM = 2000.0


def xy_stage_move_um(laser_xy_mm, r_mm: float, phi_deg: float) -> tuple[float, float, float, bool]:
    """
    Relative rig-stage move (dx, dy) [um] that brings the laser from its
    measured pupil-frame position to (R, phi). Returns (dx_stage, dy_stage,
    distance_to_target_mm, clamped). Stage +X moves the laser -X, hence the
    sign on dx. Moves are clamped to MAX_XY_MOVE_UM.
    """
    phi = np.deg2rad(phi_deg)
    dx_um = (r_mm * np.cos(phi) - float(laser_xy_mm[0])) * 1000.0
    dy_um = (r_mm * np.sin(phi) - float(laser_xy_mm[1])) * 1000.0
    mag = float(np.hypot(dx_um, dy_um))
    clamped = mag > MAX_XY_MOVE_UM
    if clamped:
        dx_um, dy_um = dx_um * MAX_XY_MOVE_UM / mag, dy_um * MAX_XY_MOVE_UM / mag
    return -dx_um, dy_um, mag / 1000.0, clamped


def z_lens_move_um(dc_current_mm: float, dc_target_mm: float) -> float:
    """Relative eye-lens move [um] taking delta_c to the target (clamped).
    The lens moves -dz: a +z lens move pushes the focus into the eye (delta_c down)."""
    dz_um = float(np.clip((dc_target_mm - float(dc_current_mm)) * 1000.0, -MAX_Z_MOVE_UM, MAX_Z_MOVE_UM))
    return -dz_um


@dataclass(frozen=True)
class ExperimentStep:
    loop: int                   # 1-based (loop, or pair for repeat runs)
    r_mm: Optional[float]       # target radius; None for repeat runs (no eye tracking)
    direction: str              # "fwd" | "bwd"


def build_experiment_steps(radii_mm, n_loops: int) -> list[ExperimentStep]:
    """Each loop: forward slews at radii in order, then backward slews at the
    radii in reverse order (the outermost radius gets fwd then bwd)."""
    radii = [float(r) for r in radii_mm]
    steps = []
    for k in range(1, int(n_loops) + 1):
        steps += [ExperimentStep(k, r, "fwd") for r in radii]
        steps += [ExperimentStep(k, r, "bwd") for r in reversed(radii)]
    return steps


def build_repeat_steps(n_pairs: int) -> list[ExperimentStep]:
    """Repeat runs without eye tracking: fwd, bwd, fwd, bwd, ... (n_pairs of
    each), alternating so slow drifts affect both directions equally."""
    steps = []
    for k in range(1, int(n_pairs) + 1):
        steps += [ExperimentStep(k, None, "fwd"), ExperimentStep(k, None, "bwd")]
    return steps


def radii_range(start_mm: float, stop_mm: float, step_mm: float) -> list[float]:
    """Inclusive radius list, e.g. (0, 4, 0.5) -> 0, 0.5, ..., 4.0."""
    if step_mm <= 0:
        raise ValueError("Radius step must be > 0")
    n = int(np.floor((stop_mm - start_mm) / step_mm + 1e-9)) + 1
    if n < 1:
        raise ValueError("Radius stop must be >= start")
    return [round(start_mm + i * step_mm, 6) for i in range(n)]


# ---------------------------------------------------------------- stage-referenced radial run
#
# No eye tracking: the user marks the apex ("center") and an outer point
# ("edge") by hand, each as the rig-stage XY plus the eye-lens z of the
# reflection plane there. Radii run along the straight stage line from center
# to edge, and the lens z at each radius comes from the sphere through the
# two planes (apex at the center, rotationally symmetric about it).

MAX_STAGE_STEP_UM = 10000.0     # refuse any single stage move larger than this


@dataclass(frozen=True)
class StageRef:
    stage_x_um: float
    stage_y_um: float
    stage_z_um: float
    lens_um: float              # eye-lens z of the reflection plane here


def stage_ref_distance_mm(center: StageRef, edge: StageRef) -> float:
    """In-plane stage distance from center to edge [mm]."""
    return float(np.hypot(edge.stage_x_um - center.stage_x_um, edge.stage_y_um - center.stage_y_um)) / 1000.0


def stage_ref_radii(center: StageRef, edge: StageRef, step_mm: float) -> list[float]:
    """0, step, 2*step, ... up to the edge; the edge itself is always the last radius."""
    d = stage_ref_distance_mm(center, edge)
    if d <= 0.0:
        raise ValueError("Center and edge are at the same stage XY")
    radii = radii_range(0.0, d, step_mm)
    if d - radii[-1] > 1e-6:
        radii.append(round(d, 6))
    return radii


def stage_ref_sphere_radius_mm(center: StageRef, edge: StageRef) -> Optional[float]:
    """Radius [mm] of the sphere whose apex is the center plane and that passes
    through the edge plane: R = (r^2 + s^2) / (2 s), s = sag. None if flat (|s| < 1 um)."""
    r = stage_ref_distance_mm(center, edge)
    s = abs(edge.lens_um - center.lens_um) / 1000.0
    if s < 1e-3:
        return None
    return (r * r + s * s) / (2.0 * s)


def stage_ref_lens_um(center: StageRef, edge: StageRef, r_mm: float) -> float:
    """Predicted eye-lens z of the reflection plane at radius r_mm (sphere model,
    same sign as edge - center; flat surface -> center z)."""
    R = stage_ref_sphere_radius_mm(center, edge)
    if R is None:
        return center.lens_um
    r = min(abs(float(r_mm)), R)
    sag_um = 1000.0 * (R - np.sqrt(R * R - r * r))
    return center.lens_um + float(np.sign(edge.lens_um - center.lens_um)) * sag_um


def stage_ref_xy_um(center: StageRef, edge: StageRef, r_mm: float) -> tuple[float, float]:
    """Absolute rig-stage XY [um] for radius r_mm on the center -> edge line."""
    f = float(r_mm) / stage_ref_distance_mm(center, edge)
    return (center.stage_x_um + f * (edge.stage_x_um - center.stage_x_um),
            center.stage_y_um + f * (edge.stage_y_um - center.stage_y_um))
