from dataclasses import dataclass

from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults


@dataclass
class AdaptiveScanParams:
    """A one-direction sweep centred on the current lens position: n_fine
    frames at min_step_um in the middle, the remaining frames on either side
    with the step growing linearly outwards so the sweep ends at
    +-total_range_um/2. Forward (+z) by default; reverse=True sweeps -z."""
    total_range_um: float
    min_step_um: float
    n_fine: int
    reverse: bool = False


@dataclass
class RequestAxialStepScan:
    id: str
    n_measurements: int
    step_size_um: float
    find_reflection_plane: bool | None = None
    eye_tracker_results: EyeTrackerResults | None = None
    # Visit the same target positions as the ordinary step scan, in shuffled
    # order (each approached from below to take up backlash). random_seed
    # fixes the order; None draws a fresh seed, which is logged.
    randomize_order: bool = False
    random_seed: int | None = None
    # Adaptive sweep (n_measurements frames in total); step_size_um is unused.
    adaptive: AdaptiveScanParams | None = None
