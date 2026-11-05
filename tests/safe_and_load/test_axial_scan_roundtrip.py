"""Schema regression test for the HDF5 data format.

Builds an AxialScan with EVERY field populated — nested calibration, eye
tracker, reflection results, sweep cycles, both configs — saves it to HDF5,
loads it back, and deep-compares. This is the guard for the two silent
failure modes of the format:

* a nested dataclass missing from known_classes (the registry is built by
  automatic discovery from the saved roots — this test fails if discovery
  ever misses one),
* a field change that saves fine but no longer loads.

Run it after ANY change to the my_dataclasses layer or the loader.
"""
import numpy as np
import pytest
from dataclasses import fields, is_dataclass

from brillouin_system.calibration.calibration import (
    CalibrationData,
    CalibrationMeasurementPoint,
    CalibrationPolyfitParameters,
    MeasurementsPerFreq,
)
from brillouin_system.devices.cameras.andor.andor_dataclasses import AndorCameraInfo
from brillouin_system.eye_tracker.eye_tracker_results import EyeTrackerResults
from brillouin_system.eye_tracker.pupil_fitting.pupil3D import Pupil3D
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.my_dataclasses.measurement_point import MeasurementPoint
from brillouin_system.my_dataclasses.sweep_cycle import SweepCycle
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    dataclass_to_hdf5_native_dict,
    dict_to_dataclass_tree,
    load_dict_from_hdf5,
    save_dict_to_hdf5,
)
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import SweepScanConfig


def make_reflection_result(seed: float) -> ReflectionResult:
    return ReflectionResult(
        found=True,
        event_index=10.5 + seed,
        event_time_perf=123.4 + seed,
        event_z_um=5000.0 + seed,
        event_z_um_interp=5001.0 + seed,
        event_z_um_fit=4999.0 + seed,
        z_offset_um=2.0,
        peak_value=3.3,
        background_mean=0.1,
        background_std=0.02,
        threshold_high=0.3,
        threshold_low=0.15,
        idx_first=100,
        idx_last=140,
        n_samples_above=41,
        n_rejected_intervals=1,
        daq_ts=np.linspace(0.0, 1.0, 5),
        daq_values=np.array([0.1, 0.2, 3.0, 0.2, 0.1]),
        zaber_lens_ts=np.linspace(0.0, 1.0, 4),
        zaber_lens_z_um=np.array([4900.0, 4950.0, 5000.0, 5050.0]),
    )


def make_full_scan() -> AxialScan:
    """Every field populated, every nested dataclass present."""
    frame = np.arange(15 * 20, dtype=float).reshape(15, 20)

    camera_info = AndorCameraInfo(
        model="DU897", serial="X-1234", roi=(1, 512, 1, 512),
        binning=(1, 1), gain=0, exposure=0.3, amp_mode="Conventional",
        preamp_gain=1.0, temperature=-70.0, flip_image_horizontally=True,
        advanced_gain_option=False, vss_speed=1.13,
        fixed_pre_amp_mode_index=2,
    )

    calibration_data = CalibrationData(measured_freqs=[
        MeasurementsPerFreq(set_freq_ghz=f, cali_meas_points=[
            CalibrationMeasurementPoint(frame=frame + f, microwave_freq=f),
        ])
        for f in (4.0, 6.0, 8.0)
    ])

    calibration_params = CalibrationPolyfitParameters(
        degree=2,
        freq_left_peak=np.array([0.001, 0.28, 0.0]),
        freq_right_peak=np.array([-0.001, -0.35, 30.0]),
        freq_peak_distance=np.array([0.0, -0.156, 10.0]),
        calibration_width_left_peak=np.array([0.0, 0.01, 1.2]),
        calibration_width_right_peak=np.array([0.0, 0.01, 1.1]),
        left_px_points=np.array([10.0, 20.0, 30.0]),
        left_freq_points=np.array([4.0, 6.0, 8.0]),
        right_px_points=np.array([70.0, 60.0, 50.0]),
        right_freq_points=np.array([4.0, 6.0, 8.0]),
        dist_px_points=np.array([60.0, 40.0, 20.0]),
        dist_freq_points=np.array([4.0, 6.0, 8.0]),
    )

    eye_tracker = EyeTrackerResults(
        left_img=np.zeros((8, 10, 3), dtype=np.uint8),
        right_img=np.ones((8, 10, 3), dtype=np.uint8),
        time_stamp=1000.5,
        laser_position=(0.1, -0.2, 0.3),
        delta_laser_corner=1.5,
        pupil3d=Pupil3D(
            center_left=np.array([1.0, 2.0, 3.0]),
            center_ref=np.array([1.1, 2.1, 3.1]),
            normal_left=np.array([0.0, 0.0, 1.0]),
            normal_ref=np.array([0.0, 0.1, 0.9]),
            radius=1.8,
        ),
    )

    return AxialScan(
        i=7,
        id="roundtrip-test-scan",
        measurements=[
            MeasurementPoint(frame_andor=frame * (k + 1),
                             lens_zaber_position=5000.0 + 10.0 * k,
                             time_stamp=float(k))
            for k in range(3)
        ],
        system_state=SystemState(is_reference_mode=False,
                                 andor_camera_info=camera_info),
        calibration_params=calibration_params,
        calibration_data=calibration_data,
        eye_tracker_results=eye_tracker,
        reflection_result_forwards=make_reflection_result(0.0),
        reflection_result_backwards=make_reflection_result(1.0),
        sweep_cycles=[
            SweepCycle(cycle_index=0,
                       reflection_in=make_reflection_result(2.0),
                       reflection_out=make_reflection_result(3.0),
                       measurement_index=0),
            SweepCycle(cycle_index=1, reflection_in=None,
                       reflection_out=make_reflection_result(4.0),
                       measurement_index=None),
        ],
        sweep_config=SweepScanConfig(),
        scanning_config=ScanningConfig(),
    )


def assert_deep_equal(a, b, path="scan"):
    """Recursive equality across dataclasses, arrays, containers, scalars."""
    if is_dataclass(a):
        assert is_dataclass(b) and type(a).__name__ == type(b).__name__, \
            f"{path}: {type(a).__name__} != {type(b).__name__}"
        for f in fields(a):
            assert_deep_equal(getattr(a, f.name), getattr(b, f.name),
                              f"{path}.{f.name}")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{path}: arrays differ"
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for k, (x, y) in enumerate(zip(a, b)):
            assert_deep_equal(x, y, f"{path}[{k}]")
    elif isinstance(a, float):
        assert b == pytest.approx(a), f"{path}: {a} != {b}"
    elif a is None:
        assert b is None, f"{path}: expected None, got {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_full_axial_scan_roundtrips_through_hdf5(tmp_path):
    scan = make_full_scan()
    path = str(tmp_path / "scan.h5")

    save_dict_to_hdf5(path, dataclass_to_hdf5_native_dict(scan))
    loaded = dict_to_dataclass_tree(load_dict_from_hdf5(path), known_classes)

    assert_deep_equal(scan, loaded)


def test_every_nested_dataclass_is_discovered_by_the_registry():
    """The registry is auto-built from the saved roots; this pins the full
    set the format has historically needed, so a discovery regression
    (e.g. an unresolvable type hint) cannot silently shrink it."""
    required = {
        "AxialScan", "MeasurementPoint", "SweepCycle",
        "CalibrationData", "MeasurementsPerFreq", "CalibrationMeasurementPoint",
        "CalibrationPolyfitParameters",
        "SystemState", "AndorCameraInfo",
        "ImageStatistics", "BackgroundImage",
        "ZaberPosition", "FittedSpectrum", "DisplayResults",
        "Intrinsics", "StereoExtrinsics", "StereoCalibration",
        "EyeTrackerConfig", "EyeTrackerResults", "Pupil3D",
        "ReflectionResult", "ScanningConfig", "SweepScanConfig",
    }
    missing = required - set(known_classes)
    assert not missing, f"registry discovery lost: {sorted(missing)}"


def test_two_channel_homodyne_reflection_result_roundtrips(tmp_path):
    from dataclasses import replace

    values = np.arange(12.0).reshape(2, 6)
    scan = replace(make_full_scan(), reflection_result_forwards=replace(
        make_reflection_result(0.0), daq_values=values, threshold_low=None))
    path = str(tmp_path / "scan.h5")
    save_dict_to_hdf5(path, dataclass_to_hdf5_native_dict(scan))
    loaded = dict_to_dataclass_tree(load_dict_from_hdf5(path), known_classes)
    assert np.array_equal(loaded.reflection_result_forwards.daq_values, values)
    assert loaded.reflection_result_forwards.threshold_low is None
