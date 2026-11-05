"""
Step order for the lens-Z vs stage-Z carrier test, run from a lens parked on the plane.

    Andor fine sweeps (lens, then stage) at the start
    n_blocks x [ stationary record, then every (axis, direction, speed) slew once in random order ]
    Andor fine sweeps (lens, then stage) at the end

Every slew and sweep is centred on the start position of its axis, so the sign of stage Z does not matter.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

WAVELENGTH_UM = 0.78024
MAX_SLEW_HALF_UM = 400.0
MAX_ANDOR_HALF_UM = 100.0
MAX_LENS_SPEED_UM_S = 2000.0
MAX_STAGE_SPEED_UM_S = 1000.0
PRELOAD_UM = 30.0           # sweeps and returns approach from below, so backlash is always taken up the same way
PRE_ROLL_S = 0.3            # DAQ runs this long before a slew starts
POST_ROLL_S = 0.3           # and this long after it should have ended
AXES = ("lens", "stage")


@dataclass(frozen=True)
class Step:
    index: int              # 0-based
    block: int              # 0 = start sweeps, 1..n blocks, n+1 = end sweeps
    kind: str               # "stationary" | "slew" | "andor"
    axis: str = ""          # "lens" | "stage" ("" for stationary)
    direction: int = 0      # +1 slews toward +z of that axis, -1 toward -z
    speed_um_s: float = 0.0

    def name(self) -> str:
        if self.kind == "stationary":
            return f"{self.index + 1:03d}_b{self.block}_stationary"
        if self.kind == "andor":
            return f"{self.index + 1:03d}_b{self.block}_andor_{self.axis}"
        d = "fwd" if self.direction > 0 else "bwd"
        return f"{self.index + 1:03d}_b{self.block}_slew_{self.axis}_{d}_v{self.speed_um_s:.0f}"


@dataclass(frozen=True)
class ProtocolConfig:
    sample: str = "glass"
    n_blocks: int = 5
    seed: int = 0
    stationary_s: float = 10.0
    slew_half_um: float = 200.0
    speeds_um_s: tuple[float, ...] = (400.0,)
    sample_rate_hz: float = 5000.0
    andor_half_um: float = 40.0
    andor_step_um: float = 2.0
    exposure_s: float = 0.1
    gain: int = 0
    close_shutters_at_end: bool = True

    def max_fringe_speed_um_s(self) -> float:
        """Fastest speed whose 2v/lambda fringe stays below 0.9 x Nyquist, for a slew that changes the path."""
        return 0.9 * 0.5 * self.sample_rate_hz * WAVELENGTH_UM / 2.0

    def validate(self) -> None:
        if self.n_blocks < 1:
            raise ValueError("At least one block is required")
        if not self.speeds_um_s:
            raise ValueError("At least one speed is required")
        if not 0 < self.slew_half_um <= MAX_SLEW_HALF_UM:
            raise ValueError(f"Slew range must be 0..{MAX_SLEW_HALF_UM:.0f} um either side of the plane")
        if not 0 < self.andor_half_um <= MAX_ANDOR_HALF_UM:
            raise ValueError(f"Andor sweep must be 0..{MAX_ANDOR_HALF_UM:.0f} um either side of the plane")
        if not 0 < self.andor_step_um <= self.andor_half_um:
            raise ValueError("The Andor step must be positive and no larger than the sweep half-width")
        if self.stationary_s <= 0 or self.exposure_s <= 0:
            raise ValueError("Durations must be positive")
        top = min(MAX_STAGE_SPEED_UM_S, self.max_fringe_speed_um_s())
        for v in self.speeds_um_s:
            if not 0 < v <= top:
                raise ValueError(f"Speed {v:.0f} um/s: must be 0..{top:.0f} um/s (stage limit, and the stage-Z "
                                 f"fringe 2v/lambda must stay below Nyquist at {self.sample_rate_hz:.0f} S/s)")

    def window_half_um(self) -> float:
        return max(self.slew_half_um, self.andor_half_um + PRELOAD_UM) + 5.0

    def steps(self) -> list[Step]:
        rng = random.Random(self.seed)
        out: list[Step] = []

        def add(**kw):
            out.append(Step(index=len(out), **kw))

        for ax in AXES:
            add(block=0, kind="andor", axis=ax)
        for b in range(1, self.n_blocks + 1):
            add(block=b, kind="stationary")
            cells = [(ax, d, float(v)) for v in self.speeds_um_s for ax in AXES for d in (1, -1)]
            rng.shuffle(cells)
            for ax, d, v in cells:
                add(block=b, kind="slew", axis=ax, direction=d, speed_um_s=v)
        for ax in AXES:
            add(block=self.n_blocks + 1, kind="andor", axis=ax)
        return out

    def andor_frames(self) -> int:
        return int(round(2.0 * self.andor_half_um / self.andor_step_um)) + 1

    def estimated_s(self, stage_move_s: float = 1.0) -> float:
        t = 0.0
        for s in self.steps():
            if s.kind == "stationary":
                t += self.stationary_s + 0.5
            elif s.kind == "andor":
                t += self.andor_frames() * (self.exposure_s + 0.15) + 2.0
            else:
                t += PRE_ROLL_S + 2.0 * self.slew_half_um / s.speed_um_s + 0.2 + POST_ROLL_S + 2.0 * stage_move_s
        return t

    def as_dict(self) -> dict:
        d = asdict(self)
        d["speeds_um_s"] = list(self.speeds_um_s)
        return d
