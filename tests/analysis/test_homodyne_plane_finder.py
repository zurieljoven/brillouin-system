import numpy as np
import pytest

from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
from brillouin_system.scan_managers.homodyne_plane_finder import (
    HomodyneFinderParams, PlaneDetector, StreamingS, find_reflection_homodyne, refine_plane, zero_phase_s,
)

FS = 5000.0
P = HomodyneFinderParams()


def background(n, seed=0, mean=0.007, std=0.0008):
    return np.random.default_rng(seed).normal(mean, std, n)


def feed(det, s, chunk=100):
    for i in range(0, s.size, chunk):
        if det.push(s[i:i + chunk]):
            return True
    return False


def test_streaming_s_does_not_depend_on_chunking():
    v = np.random.default_rng(1).normal(6.0, 0.02, (2, 3000))
    whole = StreamingS(2, FS, P).push(v)
    st = StreamingS(2, FS, P)
    parts = np.concatenate([st.push(v[:, i:i + 137]) for i in range(0, v.shape[1], 137)])
    assert parts == pytest.approx(whole)


def test_streaming_s_has_no_start_transient_on_a_large_lo_level():
    v = np.full((2, 2000), 6.2)
    assert StreamingS(2, FS, P).push(v).max() < 1e-6


def test_s_measures_fringe_amplitude_at_any_frequency():
    # S = A / sqrt(DC) on one channel; the band needs no knowledge of the fringe frequency
    t = np.arange(10000) / FS
    for f in (300.0, 1025.0, 2000.0):
        v = 4.0 + 0.2 * np.cos(2 * np.pi * f * t)
        s = zero_phase_s(v[None], FS, P)
        assert np.median(s[1000:-1000]) == pytest.approx(0.2 / np.sqrt(4.0), rel=0.02)


def test_detector_stops_after_the_set_travel_past_the_maximum():
    s = background(10000)
    s[4000:4040] = 0.2
    s[4020] = 0.3
    det = PlaneDetector(FS, P)
    assert feed(det, s)
    assert det.i_max == 4020
    assert det.n - 4020 >= 150.0 / 400.0 * FS
    assert det.n - 4020 < 150.0 / 400.0 * FS + 100 + 1   # at most one chunk later


def test_detector_follows_a_later_larger_maximum():
    s = background(10000)
    s[4000] = 0.2
    s[4500] = 0.5     # within stop_after_um of the first
    det = PlaneDetector(FS, P)
    assert feed(det, s)
    assert det.i_max == 4500


def test_detector_latches_a_weak_first_peak_when_the_real_one_is_too_far():
    s = background(10000)
    s[4000] = 0.2
    s[7000] = 0.5
    det = PlaneDetector(FS, P)
    assert feed(det, s)
    assert det.i_max == 4000


def test_detector_does_not_arm_on_background():
    det = PlaneDetector(FS, P)
    assert not feed(det, background(20000))
    assert det.i_arm is None


def test_refine_finds_the_centre_of_a_symmetric_burst():
    t = np.arange(10000) / FS
    env = 0.2 * np.exp(-((np.arange(10000) - 4000.0) / 40.0) ** 2)
    v = 6.0 + env * np.cos(2 * np.pi * 800.0 * t) + np.random.default_rng(2).normal(0, 0.002, 10000)
    assert refine_plane(v[None], FS, P).index == pytest.approx(4000.0, abs=3.0)


@pytest.mark.parametrize("direction, start", [(1.0, 7500.0), (-1.0, 8500.0)])
def test_finds_the_simulated_plane_in_real_time(direction, start):
    zaber = SimZaberLens(start_um=start)
    daq = SimHomodyneDAQ(zaber, plane_um=8000.0, peak_amplitude_v=0.2, noise_std_v=0.006, seed=3)
    r = find_reflection_homodyne(daq, zaber, HomodyneFinderParams(max_distance_um=1000.0), direction=direction,
                                 z_offset_um=200.0)
    assert r.found
    assert r.event_z_um == pytest.approx(8000.0, abs=6.0)
    assert r.z_offset_um == 200.0
    assert r.daq_values.shape[0] == 2
    # stopped ~150 um past the plane, not at the end of the search range
    assert abs(zaber.get_position() - 8000.0) == pytest.approx(150.0, abs=40.0)


def test_no_surface_in_range_is_not_found():
    zaber = SimZaberLens(start_um=7000.0)
    daq = SimHomodyneDAQ(zaber, plane_um=9000.0, peak_amplitude_v=0.2, noise_std_v=0.006, seed=4)
    assert not find_reflection_homodyne(daq, zaber, HomodyneFinderParams(max_distance_um=200.0)).found


def test_blocked_lo_is_not_found():
    zaber = SimZaberLens(start_um=7900.0)
    daq = SimHomodyneDAQ(zaber, plane_um=8000.0, lo_dc_v=(0.01, 0.01), noise_std_v=0.006, seed=5)
    assert not find_reflection_homodyne(daq, zaber, HomodyneFinderParams(max_distance_um=200.0)).found


def test_a_miss_still_returns_the_raw_record():
    zaber = SimZaberLens(start_um=7000.0)
    daq = SimHomodyneDAQ(zaber, plane_um=9000.0, peak_amplitude_v=0.2, noise_std_v=0.006, seed=6)
    r = find_reflection_homodyne(daq, zaber, HomodyneFinderParams(max_distance_um=200.0))
    assert not r.found
    assert r.daq_values.shape[0] == 2 and r.daq_values.shape[1] > 100
    assert r.zaber_lens_z_um.size >= 2
