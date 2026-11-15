"""
Sample geometry from points set by hand (lens on the plane at each): a sphere fit, then scan locations along the
center -> edge line with the lens z predicted by the fit.

Coordinates: t along the center -> edge line, d across it (stage um, center at 0). Lens z on a sphere:
    lens(t, d) = z0 + s * (R - sqrt(R^2 - (t - t0)^2 - (d - d0)^2)),   s = sign(edge lens - center lens)
Free parameters by the points available:
    center + edge only          two-point sphere, apex at the center (what the plane-accuracy run used)
    3+ points on the line       z0, R, t0      (apex free along the line)
    4+ points, some off the line z0, R, t0, d0
A sag under 1 um between center and edge is treated as flat (linear in t).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from brillouin_system.guis.z_carrier_test.runner import Location

OFF_LINE_UM = 100.0         # points further than this from the center -> edge line make the fit 2-D
MAX_STAGE_Z_SPREAD_UM = 1.0


@dataclass(frozen=True)
class SphereFit:
    method: str                 # "two-point" | "line" | "2d" | "flat"
    center: Location
    edge: Location
    z0_um: float
    r_um: float                 # sphere radius (inf when flat)
    t0_um: float                # apex along the line, from the center
    d0_um: float                # apex across the line
    sign: float
    rms_um: float               # residual over the points (nan when the fit is exact)
    n_points: int

    def _td(self, x: float, y: float) -> tuple[float, float]:
        u, v = _axes(self.center, self.edge)
        p = np.array([x - self.center.stage_x_um, y - self.center.stage_y_um])
        return float(p @ u), float(p @ v)

    def lens_at(self, x: float, y: float) -> float:
        t, d = self._td(x, y)
        if self.method == "flat":
            dist = _dist(self.center, self.edge)
            return self.center.lens_um + (self.edge.lens_um - self.center.lens_um) * t / dist
        rho2 = (t - self.t0_um) ** 2 + (d - self.d0_um) ** 2
        return float(self.z0_um + self.sign * (self.r_um - np.sqrt(max(self.r_um ** 2 - rho2, 0.0))))

    def tilt_deg(self, x: float, y: float) -> float:
        if self.method == "flat":
            return 0.0
        t, d = self._td(x, y)
        rho = np.hypot(t - self.t0_um, d - self.d0_um)
        return float(np.degrees(np.arcsin(min(rho / self.r_um, 1.0))))

    def summary(self) -> str:
        if self.method == "flat":
            return f"flat (sag < 1 um), {self.n_points} points"
        s = (f"sphere R {self.r_um / 1000:.2f} mm, apex {self.t0_um:+.0f} um along / {self.d0_um:+.0f} um across "
             f"the line from the center; {self.method} fit, {self.n_points} points")
        if np.isfinite(self.rms_um):
            s += f", rms {self.rms_um:.1f} um"
        e = self.edge
        return s + f"; edge tilt {self.tilt_deg(e.stage_x_um, e.stage_y_um):.0f} deg"


def _dist(c: Location, e: Location) -> float:
    return float(np.hypot(e.stage_x_um - c.stage_x_um, e.stage_y_um - c.stage_y_um))


def _axes(c: Location, e: Location) -> tuple[np.ndarray, np.ndarray]:
    u = np.array([e.stage_x_um - c.stage_x_um, e.stage_y_um - c.stage_y_um]) / _dist(c, e)
    return u, np.array([-u[1], u[0]])


def fit_sphere(center: Location, edge: Location, intermediates: list[Location] = ()) -> SphereFit:
    pts = [center, edge, *intermediates]
    dist = _dist(center, edge)
    if dist < 100.0:
        raise ValueError("The edge must be at least 0.1 mm from the center")
    zs = [p.stage_z_um for p in pts]
    if max(zs) - min(zs) > MAX_STAGE_Z_SPREAD_UM:
        raise ValueError(f"Stage Z differs by {max(zs) - min(zs):.1f} um between the points; keep stage Z fixed "
                         f"while setting them (only X/Y and the lens should move)")
    u, v = _axes(center, edge)
    P = np.array([[p.stage_x_um - center.stage_x_um, p.stage_y_um - center.stage_y_um] for p in pts])
    t, d, L = P @ u, P @ v, np.array([p.lens_um for p in pts])
    sag = edge.lens_um - center.lens_um
    n = len(pts)
    base = dict(center=center, edge=edge, n_points=n)
    if abs(sag) < 1.0:
        return SphereFit(method="flat", z0_um=center.lens_um, r_um=float("inf"), t0_um=0.0, d0_um=0.0, sign=0.0,
                         rms_um=float("nan"), **base)
    s = float(np.sign(sag))
    r2 = (dist ** 2 + sag ** 2) / (2.0 * abs(sag))          # two-point sphere, apex at the center
    if n == 2:
        return SphereFit(method="two-point", z0_um=center.lens_um, r_um=r2, t0_um=0.0, d0_um=0.0, sign=s,
                         rms_um=float("nan"), **base)
    two_d = n >= 4 and float(np.max(np.abs(d))) > OFF_LINE_UM

    def model(q):
        z0, R, t0 = q[:3]
        d0 = q[3] if two_d else 0.0
        rho2 = (t - t0) ** 2 + (d - d0) ** 2
        return z0 + s * (R - np.sqrt(np.clip(R ** 2 - rho2, 0.0, None)))

    q0 = [center.lens_um, r2, 0.0] + ([0.0] if two_d else [])
    reach = float(np.max(np.hypot(t, d)))
    lo = [-np.inf, 1.05 * reach + 1000.0, -0.5 * reach] + ([-0.5 * reach] if two_d else [])
    hi = [np.inf, 1e6, 0.5 * reach] + ([0.5 * reach] if two_d else [])
    q0 = list(np.clip(q0, np.array(lo) + 1e-6, np.array(hi) - 1e-6))
    res = least_squares(lambda q: model(q) - L, q0, bounds=(lo, hi))
    z0, R, t0 = res.x[:3]
    d0 = res.x[3] if two_d else 0.0
    n_par = 4 if two_d else 3
    rms = float(np.sqrt(np.sum(res.fun ** 2) / (n - n_par))) if n > n_par else float("nan")
    return SphereFit(method="2d" if two_d else "line", z0_um=float(z0), r_um=float(R), t0_um=float(t0),
                     d0_um=float(d0), sign=s, rms_um=rms, **base)


def scan_radii_mm(dist_um: float, step_mm: float) -> list[float]:
    """0, step, 2 step, ... and the edge; a grid radius within a quarter step of the edge is replaced by it."""
    if step_mm <= 0:
        raise ValueError("The radius step must be positive")
    edge = dist_um / 1000.0
    radii = [k * step_mm for k in range(int(np.floor(edge / step_mm + 1e-9)) + 1)]
    radii = [r for r in radii if edge - r > 0.25 * step_mm]
    return radii + [edge]


def scan_locations(fit: SphereFit, step_mm: float) -> list[Location]:
    """Locations along the center -> edge line, lens z from the fit, stage Z from the center."""
    c, e = fit.center, fit.edge
    u, _ = _axes(c, e)
    out = []
    for r in scan_radii_mm(_dist(c, e), step_mm):
        x, y = c.stage_x_um + 1000.0 * r * u[0], c.stage_y_um + 1000.0 * r * u[1]
        out.append(Location(name=f"r{r:.2f}", stage_x_um=x, stage_y_um=y, stage_z_um=c.stage_z_um,
                            lens_um=fit.lens_at(x, y)))
    return out


def shuffled(locations: list[Location], seed: int) -> list[Location]:
    """Random order (drift is then not aliased onto radius); the center stays first as the reference."""
    rng = np.random.default_rng(seed)
    rest = list(locations[1:])
    rng.shuffle(rest)
    return [locations[0], *rest]
