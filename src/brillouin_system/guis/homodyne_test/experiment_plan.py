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
    speed_um_s: Optional[float] = None   # None: the run's single speed


def build_experiment_steps(radii_mm, n_loops: int, speeds_um_s=None, paired: bool = False) -> list[ExperimentStep]:
    """Each loop: forward slews at radii in order, then backward slews at the
    radii in reverse order (the outermost radius gets fwd then bwd).
    With speeds, every radius and direction gets one slew per speed, back to back; the speed order
    flips every loop so slow drifts hit all speeds alike.
    paired (live-finder runs): every forward find is followed straight away by a backward find at the same radius and
    speed, which starts just past the plane the forward find located; the radius order flips every loop."""
    radii = [float(r) for r in radii_mm]
    steps = []
    for k in range(1, int(n_loops) + 1):
        vs = [None] if not speeds_um_s else [float(v) for v in speeds_um_s]
        if k % 2 == 0:
            vs = vs[::-1]
        if paired:
            rs = radii if k % 2 == 1 else radii[::-1]
            steps += [ExperimentStep(k, r, d, v) for r in rs for v in vs for d in ("fwd", "bwd")]
            continue
        steps += [ExperimentStep(k, r, "fwd", v) for r in radii for v in vs]
        steps += [ExperimentStep(k, r, "bwd", v) for r in reversed(radii) for v in vs]
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
    """0, step, 2*step, ... up to the edge; the edge itself is always the last radius.
    A grid radius within a quarter step of the edge is replaced by the edge (stage
    readback jitter makes the edge e.g. 2.75001 mm, which must not add a second edge)."""
    d = stage_ref_distance_mm(center, edge)
    if d <= 0.0:
        raise ValueError("Center and edge are at the same stage XY")
    radii = radii_range(0.0, d, step_mm)
    if len(radii) > 1 and d - radii[-1] < 0.25 * step_mm:
        radii.pop()
    if not radii or d - radii[-1] > 1e-6:
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


# ---- measured surface profile: center, intermediate points, edge --------------------------------------------------

MAX_POINT_OFF_LINE_UM = 100.0   # an intermediate point must sit on the center -> edge line (trials run along it)
MAX_POINT_DZ_UM = 5.0           # and at the center's stage Z (the runs never move stage Z)
MIN_POINT_GAP_MM = 0.05


def stage_ref_point_r_mm(center: StageRef, edge: StageRef, p: StageRef) -> tuple[float, float]:
    """(radius along the center -> edge line [mm], distance off that line [um]) of point p."""
    ux, uy = edge.stage_x_um - center.stage_x_um, edge.stage_y_um - center.stage_y_um
    n = float(np.hypot(ux, uy))
    if n <= 0.0:
        raise ValueError("Center and edge are at the same stage XY")
    dx, dy = p.stage_x_um - center.stage_x_um, p.stage_y_um - center.stage_y_um
    return (dx * ux + dy * uy) / n / 1000.0, abs(dx * uy - dy * ux) / n


def profile_nodes(center: StageRef, edge: StageRef, points=()) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (r [mm], lens z [um]) of center, every intermediate point and edge; checks the points."""
    d = stage_ref_distance_mm(center, edge)
    r, z = [0.0, d], [center.lens_um, edge.lens_um]
    for i, p in enumerate(points, 1):
        rp, off = stage_ref_point_r_mm(center, edge, p)
        if off > MAX_POINT_OFF_LINE_UM:
            raise ValueError(f"Point {i} is {off:.0f} um off the center -> edge line (limit {MAX_POINT_OFF_LINE_UM:.0f})")
        if abs(p.stage_z_um - center.stage_z_um) > MAX_POINT_DZ_UM:
            raise ValueError(f"Point {i}: stage Z differs from the center by {p.stage_z_um - center.stage_z_um:+.1f} um")
        if not 0.0 < rp < d:
            raise ValueError(f"Point {i} is at R = {rp:.2f} mm, outside the center -> edge range (0..{d:.2f} mm)")
        r.append(rp)
        z.append(p.lens_um)
    order = np.argsort(r)
    r, z = np.asarray(r)[order], np.asarray(z)[order]
    if np.any(np.diff(r) < MIN_POINT_GAP_MM):
        raise ValueError(f"Points closer than {MIN_POINT_GAP_MM} mm along the line")
    return r, z


def profile_lens_um(center: StageRef, edge: StageRef, points, r_mm: float) -> float:
    """Predicted lens z at r_mm: the 2-point sphere without intermediate points, else a shape-preserving (PCHIP)
    interpolation through center, points and edge, so it passes through every measured plane exactly."""
    if not points:
        return stage_ref_lens_um(center, edge, r_mm)
    from scipy.interpolate import PchipInterpolator
    r, z = profile_nodes(center, edge, points)
    return float(PchipInterpolator(r, z)(np.clip(float(r_mm), r[0], r[-1])))


def circle_fit_mm(r_mm: np.ndarray, z_um: np.ndarray) -> Optional[tuple[float, float, np.ndarray]]:
    """Least-squares circle through (r, z) in mm: (radius [mm], apex r [mm], residuals [um]). None if < 3 points."""
    if len(r_mm) < 3:
        return None
    x, y = np.asarray(r_mm, float), np.asarray(z_um, float) / 1000.0
    A = np.column_stack([x, y, np.ones_like(x)])
    a, b, c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    x0, y0 = a / 2.0, b / 2.0
    R = float(np.sqrt(c + x0 * x0 + y0 * y0))
    res = (np.hypot(x - x0, y - y0) - R) * 1000.0
    return R, float(x0), res
