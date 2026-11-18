"""
Step order for the lens-Z vs stage-Z carrier test, run from a lens parked on the plane.

    Andor fine sweeps (lens, then stage) at the start
    n_blocks x [ in random order: a stationary record at every DAQ sample rate, and every
                 (axis, direction, speed, sample rate) slew whose stage-Z fringe fits below Nyquist ]
    Andor fine sweeps (lens, then stage) at the end

Above 5 kS/s per channel only H (ai0) is recorded, since the USB-6008 samples 10 kS/s in total.
Sweeping the sample rate tests the aliasing: an aliased tone (e.g. the FP dither) moves with the rate, a real one
does not. Sweeping the speed tests the carrier: a path-changing slew puts its fringe at 2v/lambda.
Every slew and sweep is centred on the start position of its axis, so the sign of stage Z does not matter.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Optional

WAVELENGTH_UM = 0.78024
MAX_SLEW_HALF_UM = 400.0
MAX_ANDOR_HALF_UM = 100.0
MAX_LENS_SPEED_UM_S = 2000.0
MAX_STAGE_SPEED_UM_S = 2000.0
MAX_RATE_HZ = 10000.0       # USB-6008: 10 kS/s in total; above 5000 per channel only one channel fits
TWO_CHANNEL_MAX_HZ = 5000.0
MIN_RATE_HZ = 1000.0
CHANNELS = ("ai0", "ai1")   # H, V
PRELOAD_UM = 30.0           # sweeps and returns approach from below, so backlash is always taken up the same way
PRE_ROLL_S = 0.3            # DAQ runs this long before a slew starts
POST_ROLL_S = 0.3           # and this long after it should have ended
AXES = ("lens", "stage")


def max_fringe_speed_um_s(sample_rate_hz: float) -> float:
    """Fastest speed whose 2v/lambda fringe stays below 0.9 x Nyquist, for a slew that changes the path."""
    return 0.9 * 0.5 * sample_rate_hz * WAVELENGTH_UM / 2.0


def channels_for(sample_rate_hz: float) -> tuple[str, ...]:
    """Both channels up to 5 kS/s each; above that the USB-6008's 10 kS/s only fits H (ai0)."""
    return CHANNELS if sample_rate_hz <= TWO_CHANNEL_MAX_HZ else CHANNELS[:1]


def alias_hz(f: float, fs: float) -> float:
    """Where a tone at f appears when sampled at fs (folded into 0..fs/2)."""
    return float(abs(f - fs * round(f / fs)))


@dataclass(frozen=True)
class Step:
    index: int              # 0-based
    block: int              # 0 = start sweeps, 1..n blocks, n+1 = end sweeps
    kind: str               # "stationary" | "slew" | "andor"
    axis: str = ""          # "lens" | "stage" ("" for stationary)
    direction: int = 0      # +1 slews toward +z of that axis, -1 toward -z
    speed_um_s: float = 0.0
    fs_hz: float = 0.0      # DAQ sample rate per channel (0 for Andor sweeps)

    def name(self) -> str:
        if self.kind == "stationary":
            return f"{self.index + 1:03d}_b{self.block}_stationary_fs{self.fs_hz:.0f}"
        if self.kind == "andor":
            return f"{self.index + 1:03d}_b{self.block}_andor_{self.axis}"
        d = "fwd" if self.direction > 0 else "bwd"
        return f"{self.index + 1:03d}_b{self.block}_slew_{self.axis}_{d}_v{self.speed_um_s:.0f}_fs{self.fs_hz:.0f}"


@dataclass(frozen=True)
class ProtocolConfig:
    sample: str = "glass"
    n_blocks: int = 3
    seed: int = 0
    stationary_s: float = 10.0
    slew_half_um: float = 200.0
    speeds_um_s: tuple[float, ...] = (200.0, 400.0, 800.0, 1200.0, 1600.0)
    sample_rates_hz: tuple[float, ...] = (10000.0, 5000.0, 4000.0)   # 10000: H only
    andor_half_um: float = 40.0
    andor_step_um: float = 2.0
    exposure_s: float = 0.1
    gain: int = 0                   # Andor EM gain
    pd_gain_db: Optional[int] = 70  # photodiode gain switch, recorded only (single-gain GUIs)
    pd_gain_h_db: Optional[int] = None   # per-channel photodiode gains, recorded only (homodyne test GUI)
    pd_gain_v_db: Optional[int] = None
    fp_mod_hz: float = 3396.5       # FP lock modulation frequency, recorded only
    close_shutters_at_end: bool = True
    lens_above_nyquist: bool = False    # also slew the lens at speeds whose stage fringe would alias (speed sweep)

    def combos(self) -> list[tuple[float, float]]:
        """(speed, sample rate) pairs that are slewed: the stage-Z fringe must fit below 0.9 x Nyquist."""
        return [(float(v), float(fs)) for v in self.speeds_um_s for fs in self.sample_rates_hz
                if v <= min(MAX_STAGE_SPEED_UM_S, max_fringe_speed_um_s(fs))]

    def slews(self) -> list[tuple[str, float, float]]:
        """(axis, speed, sample rate) slewed. Stage only below its fringe Nyquist limit; with lens_above_nyquist the
        lens runs every speed up to MAX_LENS_SPEED_UM_S (its fringe is vibration, not 2v/lambda)."""
        ok = set(self.combos())
        out = []
        for v in self.speeds_um_s:
            for fs in self.sample_rates_hz:
                c = (float(v), float(fs))
                if c in ok or (self.lens_above_nyquist and v <= MAX_LENS_SPEED_UM_S):
                    out.append(("lens", *c))
                if c in ok:
                    out.append(("stage", *c))
        return out

    def skipped(self) -> list[tuple[float, float]]:
        """(speed, rate) pairs whose stage slews are skipped."""
        ok = set(self.combos())
        return [(float(v), float(fs)) for v in self.speeds_um_s for fs in self.sample_rates_hz
                if (float(v), float(fs)) not in ok]

    def validate(self) -> None:
        if self.n_blocks < 1:
            raise ValueError("At least one block is required")
        if not self.speeds_um_s:
            raise ValueError("At least one speed is required")
        if not self.sample_rates_hz:
            raise ValueError("At least one sample rate is required")
        if len(set(self.sample_rates_hz)) != len(self.sample_rates_hz) or len(set(self.speeds_um_s)) != len(self.speeds_um_s):
            raise ValueError("Speeds and sample rates must not repeat")
        for fs in self.sample_rates_hz:
            if not MIN_RATE_HZ <= fs <= MAX_RATE_HZ:
                raise ValueError(f"Sample rate {fs:.0f}: must be {MIN_RATE_HZ:.0f}..{MAX_RATE_HZ:.0f} S/s per channel "
                                 f"(USB-6008; above {TWO_CHANNEL_MAX_HZ:.0f} only H is recorded)")
        if not 0 < self.slew_half_um <= MAX_SLEW_HALF_UM:
            raise ValueError(f"Slew range must be 0..{MAX_SLEW_HALF_UM:.0f} um either side of the plane")
        if not 0 < self.andor_half_um <= MAX_ANDOR_HALF_UM:
            raise ValueError(f"Andor sweep must be 0..{MAX_ANDOR_HALF_UM:.0f} um either side of the plane")
        if not 0 < self.andor_step_um <= self.andor_half_um:
            raise ValueError("The Andor step must be positive and no larger than the sweep half-width")
        if self.stationary_s <= 0 or self.exposure_s <= 0:
            raise ValueError("Durations must be positive")
        good = {v for _, v, _ in self.slews()}
        top = min(MAX_STAGE_SPEED_UM_S, max_fringe_speed_um_s(max(self.sample_rates_hz)))
        if self.lens_above_nyquist:
            top = MAX_LENS_SPEED_UM_S
        for v in self.speeds_um_s:
            if v <= 0 or v not in good:
                raise ValueError(f"Speed {v:.0f} um/s: must be 0..{top:.0f} um/s at the highest sample rate (stage "
                                 f"limit, and the stage-Z fringe 2v/lambda must stay below Nyquist)")

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
            cells = [dict(kind="stationary", fs_hz=float(fs)) for fs in self.sample_rates_hz]
            cells += [dict(kind="slew", axis=ax, direction=d, speed_um_s=v, fs_hz=fs)
                      for ax, v, fs in self.slews() for d in (1, -1)]
            rng.shuffle(cells)
            for c in cells:
                add(block=b, **c)
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
        d["sample_rates_hz"] = list(self.sample_rates_hz)
        d["slewed_combos"] = [list(c) for c in self.combos()]
        d["slews"] = [list(c) for c in self.slews()]
        d["fp_alias_hz"] = {f"{fs:.0f}": alias_hz(self.fp_mod_hz, fs) for fs in self.sample_rates_hz}
        return d
