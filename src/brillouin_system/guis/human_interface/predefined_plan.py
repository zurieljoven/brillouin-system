"""Predefined measurement plans (TOML) and live quality-control metrics.

A plan is a TOML file: one ``[defaults]`` table whose values apply to every
step, plus an array of ``[[step]]`` tables that override only what differs.
Each step expands into ``replicates`` sweep scans, whose IDs are
``R<R_mm>_phi<phi_deg>_d<depth_um>_n<k>`` — the same identifier the acquisition code tags each
sweep scan with, so a saved scan can be matched back to the step that asked
for it.

Everything here is pure Python over duck-typed scans (``id``,
``sweep_cycles``, ``measurements``): no Qt, no hardware, no spectrum fitting.
That keeps the "Check Progress" pass fast and this module unit-testable.

The quality metric is the per-cycle MOTION delta already logged by the sweep
scan: |out-crossing z - in-crossing z| for a cycle that found both crossings.
A small delta means the eye held still while that frame was taken. A scan
"passes" its motion limit if at least one of its cycles has |delta| < limit;
a limit of 0 (or None) means no gate.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Plan schema
# ---------------------------------------------------------------------------

# Fields a [[step]] may carry (besides depth_um). Anything a step omits is
# inherited from [defaults]; anything both omit falls back to _DEFAULTS below.
_DEFAULTS: dict[str, Any] = {
    "mode": "timed",         # "timed" (cycles bounded by max_time_s) or "fixed"
    "max_time_s": 10.0,      # timed budget [s]; fixed-mode wall-clock cap (0 = uncapped)
    "cycles": 6,             # fixed mode only: number of in-out cycles
    "replicates": 5,         # sweep scans taken at this depth
    "R_mm": 0.0,             # laser radius  -> Move XY
    "phi_deg": 0.0,          # laser angle   -> Move XY
    "delta_c_mm": 0.0,       # Move Z
    "motion_limit_um": 0.0,  # |delta| gate [µm]; 0 = no gate
}


@dataclass(frozen=True)
class PlanEntry:
    """One depth line of a plan, before expansion into replicates."""
    depth_um: float
    replicates: int
    timed: bool
    cycles: int | None       # fixed mode only (None when timed)
    max_time_s: float
    R_mm: float
    phi_deg: float
    delta_c_mm: float
    motion_limit_um: float


@dataclass(frozen=True)
class PlanStep:
    """One expanded step: a single sweep scan the operator will take."""
    depth_um: float
    num: int                 # replicate number at this location and depth (1-based)
    timed: bool
    cycles: int | None
    max_time_s: float
    R_mm: float
    phi_deg: float
    delta_c_mm: float
    motion_limit_um: float

    @property
    def id(self) -> str:
        return (f"R{_fmt_depth(self.R_mm)}_phi{_fmt_depth(self.phi_deg)}"
                f"_d{_fmt_depth(self.depth_um)}_n{self.num}")

    def mode_str(self) -> str:
        budget = f"{self.max_time_s:g} s" if self.max_time_s > 0 else "no cap"
        if self.timed:
            return f"TIMED (budget {budget})"
        return f"cycles={self.cycles}, max time={budget}"


def _fmt_depth(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


# ---------------------------------------------------------------------------
# Parsing / expansion
# ---------------------------------------------------------------------------

def parse_plan_toml(path: str | Path) -> list[PlanEntry]:
    """Parse a TOML plan file into per-depth entries.

    Raises ValueError (naming the offending step) on bad input.
    """
    try:
        import tomllib as _toml  # Python 3.11+
    except ModuleNotFoundError:
        import tomli as _toml     # Python 3.10 (project dependency)

    with open(path, "rb") as f:
        data = _toml.load(f)

    defaults = dict(_DEFAULTS)
    raw_defaults = data.get("defaults", {})
    if not isinstance(raw_defaults, dict):
        raise ValueError("[defaults] must be a table.")
    _reject_unknown_keys(raw_defaults, where="[defaults]", allow_depth=False)
    defaults.update(raw_defaults)

    raw_steps = data.get("step", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(
            "Plan has no steps. Add at least one [[step]] table with a "
            "depth_um.")

    entries: list[PlanEntry] = []
    for i, raw in enumerate(raw_steps, start=1):
        where = f"[[step]] #{i}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a table.")
        _reject_unknown_keys(raw, where=where, allow_depth=True)
        if "depth_um" not in raw:
            raise ValueError(f"{where}: missing required 'depth_um'.")
        merged = {**defaults, **raw}
        try:
            entries.append(_entry_from_merged(merged))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}: {e}") from e
    return entries


def _reject_unknown_keys(table: dict, *, where: str, allow_depth: bool) -> None:
    allowed = set(_DEFAULTS) | ({"depth_um"} if allow_depth else set())
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}.")


def _entry_from_merged(m: dict) -> PlanEntry:
    depth_um = float(m["depth_um"])
    replicates = int(m["replicates"])
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")

    mode = str(m["mode"]).lower()
    if mode not in ("timed", "fixed"):
        raise ValueError(f"mode must be 'timed' or 'fixed', got {mode!r}")
    timed = mode == "timed"

    max_time_s = float(m["max_time_s"])
    if timed:
        cycles = None
        if max_time_s <= 0:
            raise ValueError("timed mode needs max_time_s > 0")
    else:
        cycles = int(m["cycles"])
        if cycles < 1:
            raise ValueError(f"fixed mode needs cycles >= 1, got {cycles}")
        if max_time_s < 0:
            raise ValueError(f"max_time_s must be >= 0, got {max_time_s}")

    motion_limit_um = float(m["motion_limit_um"])
    if motion_limit_um < 0:
        raise ValueError(
            f"motion_limit_um must be >= 0 (0 disables the gate), got "
            f"{motion_limit_um}")

    return PlanEntry(
        depth_um=depth_um,
        replicates=replicates,
        timed=timed,
        cycles=cycles,
        max_time_s=max_time_s,
        R_mm=float(m["R_mm"]),
        phi_deg=float(m["phi_deg"]),
        delta_c_mm=float(m["delta_c_mm"]),
        motion_limit_um=motion_limit_um,
    )


def expand_plan(entries: Iterable[PlanEntry],
                scramble: bool = False,
                rng: random.Random | None = None) -> list[PlanStep]:
    """Expand entries into individual steps.

    Steps are grouped by depth; replicate numbers count per (R, phi, depth), so
    an ID always means the same location and replicate. When scrambling, depths
    are interleaved randomly but each depth keeps its file order (and therefore
    every location's n1, n2, … numbering).
    """
    groups: "OrderedDict[float, list[PlanEntry]]" = OrderedDict()
    for e in entries:
        for _ in range(e.replicates):
            groups.setdefault(e.depth_um, []).append(e)

    counters: dict[tuple[float, float, float], int] = {}

    def _step(e: PlanEntry) -> PlanStep:
        key = (e.R_mm, e.phi_deg, e.depth_um)
        counters[key] = counters.get(key, 0) + 1
        return PlanStep(
            depth_um=e.depth_um, num=counters[key], timed=e.timed, cycles=e.cycles,
            max_time_s=e.max_time_s, R_mm=e.R_mm, phi_deg=e.phi_deg,
            delta_c_mm=e.delta_c_mm, motion_limit_um=e.motion_limit_um,
        )

    if not scramble:
        out: list[PlanStep] = []
        for reps in groups.values():
            for e in reps:
                out.append(_step(e))
        return out

    rng = rng or random.Random()
    remaining = OrderedDict((d, list(reps)) for d, reps in groups.items())
    sequence: list[PlanStep] = []
    while any(remaining.values()):
        avail = [d for d, reps in remaining.items() if reps]
        d = rng.choice(avail)
        sequence.append(_step(remaining[d].pop(0)))
    return sequence


# ---------------------------------------------------------------------------
# Quality metric: per-cycle motion delta
# ---------------------------------------------------------------------------

def cycle_motion_deltas(cycles: Iterable[Any] | None) -> list[float]:
    """|out-crossing z - in-crossing z| [µm] for each cycle that found both
    crossings. Cycles with a missing/rejected crossing contribute nothing —
    their motion is unknown, so they can never satisfy the limit."""
    out: list[float] = []
    for c in (cycles or []):
        ri = getattr(c, "reflection_in", None)
        ro = getattr(c, "reflection_out", None)
        if (ri is not None and ro is not None
                and getattr(ri, "found", False) and getattr(ro, "found", False)
                and ri.event_z_um is not None and ro.event_z_um is not None):
            out.append(abs(float(ro.event_z_um) - float(ri.event_z_um)))
    return out


def scan_motion_deltas(scan: Any) -> list[float]:
    return cycle_motion_deltas(getattr(scan, "sweep_cycles", None))


def passes_motion_limit(deltas: Iterable[float], limit_um: float | None) -> bool:
    """A scan passes if some cycle held below the limit. limit 0/None = no gate
    (always passes as long as there is data upstream)."""
    if not limit_um or limit_um <= 0:
        return True
    return any(d < limit_um for d in deltas)


def scan_passes_motion(scan: Any, limit_um: float | None) -> bool:
    return passes_motion_limit(scan_motion_deltas(scan), limit_um)


def cycle_frames(scan: Any) -> list[tuple[float, float]]:
    """Per-FRAME (actual_depth_um, |motion delta|_um) for the coverage plot.

    One entry per cycle that found BOTH crossings and took a frame:
        depth = frame lens position − (in-crossing + out-crossing) / 2
        |delta| = |out-crossing − in-crossing|
    depth is measured against the bias-free (forward/backward averaged) plane,
    so on a well-behaved frame it equals the prescribed target depth (lands on
    y = x). |delta| is that frame's own eye motion — the per-frame quantity the
    motion limit is judged against. No fitting."""
    out: list[tuple[float, float]] = []
    measurements = getattr(scan, "measurements", None) or []
    for c in (getattr(scan, "sweep_cycles", None) or []):
        ri = getattr(c, "reflection_in", None)
        ro = getattr(c, "reflection_out", None)
        mi = getattr(c, "measurement_index", None)
        if not (ri is not None and ro is not None
                and getattr(ri, "found", False) and getattr(ro, "found", False)
                and ri.event_z_um is not None and ro.event_z_um is not None
                and mi is not None and 0 <= mi < len(measurements)):
            continue
        frame_z = getattr(measurements[mi], "lens_zaber_position", None)
        if frame_z is None:
            continue
        in_z, out_z = float(ri.event_z_um), float(ro.event_z_um)
        depth = float(frame_z) - 0.5 * (in_z + out_z)
        out.append((depth, abs(out_z - in_z)))
    return out


def cycle_actual_depths(scan: Any) -> list[float]:
    """Per-frame measured depths past the plane (see cycle_frames)."""
    return [depth for depth, _ in cycle_frames(scan)]


def scan_mean_actual_depth_um(scan: Any) -> float | None:
    """Mean of a scan's per-frame measured depths (see cycle_frames)."""
    depths = cycle_actual_depths(scan)
    return sum(depths) / len(depths) if depths else None


def frame_passes_motion_limit(delta_um: float, limit_um: float | None) -> bool:
    """One frame passes if its own motion delta is under the limit.
    limit 0/None = no gate (every frame passes)."""
    if not limit_um or limit_um <= 0:
        return True
    return delta_um < limit_um


# ---------------------------------------------------------------------------
# Progress report (Check Progress)
# ---------------------------------------------------------------------------

@dataclass
class ProgressPoint:
    """One FRAME on the coverage plot: prescribed depth (x) vs measured depth
    past the bias-free plane (y). passed is THIS frame's own motion verdict
    (|delta| < the step's limit), not its scan's."""
    prescribed_depth_um: float
    actual_depth_um: float
    passed: bool


@dataclass
class ProgressRow:
    id: str
    depth_um: float
    R_mm: float
    phi_deg: float
    motion_limit_um: float
    taken: bool
    n_scans: int                       # saved scans matching this step's ID
    n_frames: int                      # frames with a measurable motion delta
    n_pass_frames: int                 # of those, frames within the limit
    avg_actual_depth_um: float | None  # mean measured depth past the plane
    min_delta_um: float | None


@dataclass
class ProgressReport:
    rows: list[ProgressRow] = field(default_factory=list)
    unmatched_scan_ids: list[str] = field(default_factory=list)
    # One point per measured frame across all matched scans, for the plot.
    points: list[ProgressPoint] = field(default_factory=list)

    @property
    def planned_total(self) -> int:
        return len(self.rows)

    @property
    def taken_count(self) -> int:
        return sum(1 for r in self.rows if r.taken)

    @property
    def frames_total(self) -> int:
        return sum(r.n_frames for r in self.rows)

    @property
    def frames_pass(self) -> int:
        return sum(r.n_pass_frames for r in self.rows)

    def remaining_ids(self) -> list[str]:
        return [r.id for r in self.rows if not r.taken]


def build_progress(planned: Iterable[PlanStep],
                   scans: Iterable[Any]) -> ProgressReport:
    """Join the planned steps against the currently-saved scans.

    A step is 'taken' if at least one saved scan carries its ID (so a scan the
    operator removed from the list stops counting), and 'passed' if some
    matching saved scan clears the step's motion limit. Averaged position and
    min delta are reported over all matching scans.
    """
    planned = list(planned)
    planned_ids = {s.id for s in planned}

    by_id: dict[str, list[Any]] = {}
    for scan in scans:
        by_id.setdefault(str(getattr(scan, "id", "")), []).append(scan)

    rows: list[ProgressRow] = []
    points: list[ProgressPoint] = []
    for step in planned:
        matches = by_id.get(step.id, [])
        depths_all: list[float] = []
        min_delta: float | None = None
        n_frames = 0
        n_pass_frames = 0
        for scan in matches:
            # Per-FRAME: each cycle's depth and its own motion delta, judged
            # against the limit independently.
            for depth, delta in cycle_frames(scan):
                n_frames += 1
                depths_all.append(depth)
                min_delta = delta if min_delta is None else min(min_delta, delta)
                frame_ok = frame_passes_motion_limit(delta, step.motion_limit_um)
                if frame_ok:
                    n_pass_frames += 1
                points.append(ProgressPoint(
                    prescribed_depth_um=step.depth_um,
                    actual_depth_um=depth,
                    passed=frame_ok,
                ))
        rows.append(ProgressRow(
            id=step.id,
            depth_um=step.depth_um,
            R_mm=step.R_mm,
            phi_deg=step.phi_deg,
            motion_limit_um=step.motion_limit_um,
            taken=bool(matches),
            n_scans=len(matches),
            n_frames=n_frames,
            n_pass_frames=n_pass_frames,
            avg_actual_depth_um=(sum(depths_all) / len(depths_all)) if depths_all else None,
            min_delta_um=min_delta,
        ))

    unmatched = sorted(
        {str(getattr(s, "id", "")) for s in scans} - planned_ids)
    return ProgressReport(rows=rows, unmatched_scan_ids=unmatched, points=points)
