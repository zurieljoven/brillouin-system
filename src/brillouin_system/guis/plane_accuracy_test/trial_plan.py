"""
Trial order for the plane precision + accuracy run: randomized complete blocks.

Every block holds every (radius, kind) once, in its own random order, so slow
drift is spread evenly over radii and kinds instead of aliasing onto them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

KINDS = ("find_fwd", "find_bwd", "axial")


@dataclass(frozen=True)
class Trial:
    index: int          # 0-based position in the run
    block: int          # 1-based
    r_mm: float
    kind: str           # "find_fwd" | "find_bwd" | "axial"


def build_trials(radii_mm: list[float], n_blocks: int, seed: int) -> list[Trial]:
    if n_blocks < 1:
        raise ValueError("At least one block is required")
    if not radii_mm:
        raise ValueError("At least one radius is required")
    rng = random.Random(seed)
    trials: list[Trial] = []
    for block in range(1, n_blocks + 1):
        cells = [(float(r), k) for r in radii_mm for k in KINDS]
        rng.shuffle(cells)
        trials += [Trial(len(trials) + i, block, r, k) for i, (r, k) in enumerate(cells)]
    return trials


def axial_positions_um(z_pred_um: float, before_um: float, after_um: float, step_um: float) -> np.ndarray:
    """Lens positions from before the predicted plane to after it, always increasing."""
    if step_um <= 0:
        raise ValueError("The axial step must be positive")
    n = int(np.floor((before_um + after_um) / step_um + 1e-9)) + 1
    return z_pred_um - before_um + step_um * np.arange(n)
