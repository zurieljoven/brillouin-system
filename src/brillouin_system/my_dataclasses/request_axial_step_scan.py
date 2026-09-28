from dataclasses import dataclass

from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults


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
