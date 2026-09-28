"""The acquisition procedures, driven against a fake backend.

scan_procedures functions only touch a defined surface of the backend
(devices, display callbacks, scan registration), so a duck-typed fake
exercises the orchestration — step sequence, cancellation, lens return,
scan registration — without hardware or Qt.
"""
import numpy as np

from brillouin_system.devices.cameras.andor.andor_dataclasses import AndorCameraInfo
from brillouin_system.guis.human_interface.scan_procedures import (
    AXIAL_SCAN_BACKLASH_PRELOAD_UM,
    adaptive_scan_targets,
    perform_calibration,
    random_scan_targets,
    take_axial_step_scan,
)
import pytest

from brillouin_system.my_dataclasses.request_axial_step_scan import AdaptiveScanParams, RequestAxialStepScan
from brillouin_system.my_dataclasses.system_state import SystemState


class FakeLens:
    def __init__(self, start_um: float):
        self.position = start_um
        self.moves: list[tuple[str, float]] = []

    def get_position(self) -> float:
        return self.position

    def move_rel(self, delta_um: float):
        self.position += delta_um
        self.moves.append(("rel", delta_um))

    def move_abs(self, z_um: float):
        self.position = z_um
        self.moves.append(("abs", z_um))


class FakeMicrowave:
    def __init__(self):
        self.freq = 0.0

    def set_frequency(self, freq: float):
        self.freq = freq

    def get_frequency(self) -> float:
        return self.freq


class FakeCalculator:
    @staticmethod
    def get_str_all_models() -> str:
        return "fake models"


class FakeBackend:
    """Only the surface scan_procedures uses."""

    def __init__(self, is_reference_mode: bool, cancel_after: int | None = None):
        self.is_reference_mode = is_reference_mode
        self.zaber_eye_lens = FakeLens(start_um=9000.0)
        self.microwave = FakeMicrowave()
        self.calibration_poly_fit_params = None
        self.calibration_data = None
        self.calibration_calculator = FakeCalculator()

        self.displayed_frames: list[np.ndarray] = []
        self.emitted_lens_positions: list[float] = []
        self.returned_to: list[float] = []
        self.registered: list = []
        self.calculator_updates = 0

        self._i = 0
        self._snaps = 0
        self._cancel_after = cancel_after

    # --- primitives the procedures compose ---

    def f2b_cancel_callback(self) -> bool:
        return (self._cancel_after is not None
                and self._snaps >= self._cancel_after)

    def get_andor_frame(self) -> np.ndarray:
        self._snaps += 1
        return np.full((4, 6), float(self._snaps))

    def display_spectrum(self, frame):
        self.displayed_frames.append(frame)

    def b2f_emit_update_zaber_lens_position(self, z_um: float):
        self.emitted_lens_positions.append(z_um)

    def move_and_update_gui_zaber_eye_lens_abs(self, z_um: float):
        self.zaber_eye_lens.move_abs(z_um)
        self.returned_to.append(z_um)

    def get_current_system_state(self) -> SystemState:
        return SystemState(
            is_reference_mode=self.is_reference_mode,
            andor_camera_info=AndorCameraInfo(
                model="fake", serial="0", roi=(1, 6, 1, 4), binning=(1, 1),
                gain=0, exposure=0.1, amp_mode="Conventional",
                preamp_gain=1.0, temperature=-70.0,
                flip_image_horizontally=False, advanced_gain_option=False,
                vss_speed=1.0),
        )

    def calibration_data_to_store(self):
        return self.calibration_data

    def next_axial_scan_index(self) -> int:
        self._i += 1
        return self._i

    def register_axial_scan(self, scan):
        self.registered.append(scan)

    # --- perform_calibration extras ---

    def force_reference_mode(self):
        import contextlib

        @contextlib.contextmanager
        def cm():
            yield
        return cm()

    def update_calibration_calculator(self):
        self.calculator_updates += 1


def test_reference_scan_takes_n_frames_and_registers_the_scan():
    backend = FakeBackend(is_reference_mode=True)
    ok = take_axial_step_scan(backend, RequestAxialStepScan(
        id="ref", n_measurements=4, step_size_um=0.0))

    assert ok
    assert len(backend.registered) == 1
    scan = backend.registered[0]
    assert scan.i == 1 and scan.id == "ref"
    assert len(scan.measurements) == 4
    assert len(backend.displayed_frames) == 4
    # Reference frames are taken at the starting lens position.
    assert all(m.lens_zaber_position == 9000.0 for m in scan.measurements)


def test_sample_scan_steps_the_lens_and_returns_to_start():
    backend = FakeBackend(is_reference_mode=False)
    ok = take_axial_step_scan(backend, RequestAxialStepScan(
        id="sample", n_measurements=3, step_size_um=10.0))

    assert ok
    scan = backend.registered[0]
    positions = [m.lens_zaber_position for m in scan.measurements]
    assert positions == [9010.0, 9020.0, 9030.0]
    assert backend.emitted_lens_positions == positions
    # Lens went back to the starting position at the end.
    assert backend.returned_to == [9000.0]
    assert backend.zaber_eye_lens.position == 9000.0


def test_random_scan_visits_the_step_scan_positions_in_shuffled_order():
    backend = FakeBackend(is_reference_mode=False)
    ok = take_axial_step_scan(backend, RequestAxialStepScan(
        id="random", n_measurements=8, step_size_um=10.0,
        randomize_order=True, random_seed=3))

    assert ok
    positions = [m.lens_zaber_position for m in backend.registered[0].measurements]
    # Same set of positions as the ordinary step scan, not in ascending order.
    assert sorted(positions) == [9000.0 + 10.0 * (k + 1) for k in range(8)]
    assert positions != sorted(positions)
    assert positions == random_scan_targets(9000.0, 10.0, 8, seed=3)
    assert backend.emitted_lens_positions == positions
    assert backend.zaber_eye_lens.position == 9000.0


def test_random_scan_approaches_every_target_from_below():
    backend = FakeBackend(is_reference_mode=False)
    take_axial_step_scan(backend, RequestAxialStepScan(
        id="random", n_measurements=6, step_size_um=10.0,
        randomize_order=True, random_seed=1))

    moves = backend.zaber_eye_lens.moves[:-1]      # last move = return to start
    assert all(kind == "abs" for kind, _ in moves)
    # Moves come in pairs: preload position, then the target 100 µm above it.
    for (_, preload), (_, target) in zip(moves[0::2], moves[1::2]):
        assert target - preload == AXIAL_SCAN_BACKLASH_PRELOAD_UM


def test_adaptive_targets_fine_centre_and_growing_steps_to_the_range_ends():
    z = adaptive_scan_targets(1000.0, n_total=101, total_range_um=100.0, min_step_um=0.2, n_fine=41)
    steps = np.diff(z)
    assert len(z) == 101
    assert z[0] == pytest.approx(950.0) and z[-1] == pytest.approx(1050.0)
    assert np.all(steps > 0)
    # 41 fine frames at the smallest step, centred on 1000 µm.
    centre = int(np.argmin(np.abs(np.array(z) - 1000.0)))
    assert z[centre] == pytest.approx(1000.0)
    assert np.allclose(steps[centre - 20:centre + 20], 0.2)
    # Steps never shrink going outwards on either side, and the largest are at the ends.
    right, left = steps[centre + 20:], steps[:centre - 20][::-1]
    assert np.all(np.diff(right) >= -1e-12) and np.all(np.diff(left) >= -1e-12)
    assert steps.max() == pytest.approx(max(steps[0], steps[-1]))


@pytest.mark.parametrize("kwargs", [
    dict(n_total=100, total_range_um=10.0, min_step_um=0.2, n_fine=80),   # fine core wider than the range
    dict(n_total=600, total_range_um=20.0, min_step_um=0.1, n_fine=50),   # too many wing frames
    dict(n_total=10, total_range_um=20.0, min_step_um=0.1, n_fine=20),    # more fine than total frames
])
def test_adaptive_targets_refuse_impossible_inputs(kwargs):
    with pytest.raises(ValueError):
        adaptive_scan_targets(0.0, **kwargs)


@pytest.mark.parametrize("reverse", [False, True])
def test_adaptive_sweep_is_monotonic_with_one_preload_on_the_incoming_side(reverse):
    backend = FakeBackend(is_reference_mode=False)
    ok = take_axial_step_scan(backend, RequestAxialStepScan(
        id="adaptive", n_measurements=21, step_size_um=0.5,
        adaptive=AdaptiveScanParams(total_range_um=40.0, min_step_um=0.5, n_fine=9, reverse=reverse)))

    assert ok
    positions = [m.lens_zaber_position for m in backend.registered[0].measurements]
    expected = adaptive_scan_targets(9000.0, 21, 40.0, 0.5, 9)
    assert positions == (expected[::-1] if reverse else expected)
    # One preload move beyond the first target (below forward, above reverse),
    # then exactly one move per frame, then the return to the start.
    moves = [z for _, z in backend.zaber_eye_lens.moves]
    sign = 1.0 if reverse else -1.0
    assert moves[0] == pytest.approx(positions[0] + sign * AXIAL_SCAN_BACKLASH_PRELOAD_UM)
    assert moves[1:-1] == positions
    assert backend.zaber_eye_lens.position == 9000.0


def test_cancellation_registers_nothing_and_returns_the_lens():
    backend = FakeBackend(is_reference_mode=False, cancel_after=2)
    ok = take_axial_step_scan(backend, RequestAxialStepScan(
        id="cancelled", n_measurements=10, step_size_um=10.0))

    assert not ok
    assert backend.registered == []
    assert backend.returned_to == [9000.0]
    assert backend.zaber_eye_lens.position == 9000.0


def test_perform_calibration_stores_raw_frames_per_frequency():
    backend = FakeBackend(is_reference_mode=False)
    ok = perform_calibration(backend)

    assert ok
    assert backend.calculator_updates == 1
    data = backend.calibration_data
    assert data is not None

    from brillouin_system.calibration.config.calibration_config import calibration_config
    cfg = calibration_config.get()
    assert len(data.measured_freqs) == len(cfg.calibration_freqs)
    for block in data.measured_freqs:
        assert len(block.cali_meas_points) == cfg.n_per_freq
        # The set frequency was actually programmed on the synthesizer.
        assert all(p.microwave_freq == block.set_freq_ghz
                   for p in block.cali_meas_points)
