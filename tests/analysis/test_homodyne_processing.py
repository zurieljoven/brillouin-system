import numpy as np
import pytest

from brillouin_system.devices.ni.ni6008_multi import NI6008Multi, effective_range_v
from brillouin_system.scan_managers.homodyne_processing import (
    combine_polarizations,
    fringe_band_hz,
    fringe_envelope,
    fringe_frequency_hz,
    locate_peak,
    process_channels,
)
from brillouin_system.scan_managers.homodyne_recording import (
    HomodyneRecording,
    load_recording,
    save_recording,
)

FS = 5000.0
LAMBDA_NM = 780.24


def _slew(
    *,
    split: float,
    lo_dc=(0.8, 0.8),
    speed_um_s=400.0,
    plane_um=150.0,
    amp_v=0.01,
    sigma_um=3.8,
    noise_v=2e-4,
    seed=0,
):
    """Two-channel fringe scan through a Gaussian surface; `split` is the
    fraction of signal POWER in channel H."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.75 * FS)) / FS
    z = speed_um_s * t
    env = amp_v * np.exp(-((z - plane_um) ** 2) / (4 * sigma_um**2))
    phase = 4 * np.pi * z / (LAMBDA_NM * 1e-3) + rng.uniform(0, 2 * np.pi)
    v = np.empty((2, t.size))
    for i, (dc, frac) in enumerate(zip(lo_dc, (split, 1 - split))):
        a = env * np.sqrt(dc / 0.8) * np.sqrt(frac)  # fringe amp ~ sqrt(P_LO * P_s)
        v[i] = dc + a * np.cos(phase + 1.1 * i) + rng.normal(0, noise_v, t.size)
    return z, v, speed_um_s


def test_fringe_frequency_780nm():
    assert fringe_frequency_hz(400.0, 780.0) == pytest.approx(1025.6, rel=1e-3)
    assert fringe_frequency_hz(-400.0, 780.0) == pytest.approx(1025.6, rel=1e-3)


def test_fringe_band_rejects_speed_above_nyquist():
    lo, hi = fringe_band_hz(400.0, LAMBDA_NM, FS)
    assert lo < fringe_frequency_hz(400.0, LAMBDA_NM) < hi < FS / 2
    with pytest.raises(ValueError):
        fringe_band_hz(2000.0, LAMBDA_NM, FS)  # 5.1 kHz fringe, 2.5 kHz Nyquist


def test_envelope_recovers_constant_amplitude():
    t = np.arange(int(FS)) / FS
    v = 0.8 + 0.005 * np.cos(2 * np.pi * 1000.0 * t + 0.3)
    env = fringe_envelope(v, FS, (600.0, 1400.0))
    mid = env[500:-500]  # skip filter edges
    assert np.median(mid) == pytest.approx(0.005, rel=0.02)


@pytest.mark.parametrize("split", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_combined_s_is_polarization_independent(split):
    z, v, speed = _slew(split=split, lo_dc=(0.8, 0.5), noise_v=0.0)
    trace = process_channels(v, FS, fringe_band_hz(speed, LAMBDA_NM, FS))
    z_ref, v_ref, _ = _slew(split=0.5, lo_dc=(0.8, 0.5), noise_v=0.0)
    ref = process_channels(v_ref, FS, fringe_band_hz(speed, LAMBDA_NM, FS))
    assert np.max(trace.s) == pytest.approx(np.max(ref.s), rel=0.03)


def test_combine_is_invariant_to_lo_imbalance():
    env = np.array([[0.01], [0.01]])
    # same signal powers with 4x LO in H: A_H doubles, DC_H quadruples
    balanced = combine_polarizations(env, np.array([[0.8], [0.8]]))
    unbalanced = combine_polarizations(env * np.array([[2.0], [1.0]]), np.array([[3.2], [0.8]]))
    assert unbalanced[0] == pytest.approx(balanced[0])


@pytest.mark.parametrize("split", [0.0, 0.3, 1.0])
def test_peak_located_within_tolerance(split):
    z, v, speed = _slew(split=split, seed=3)
    trace = process_channels(v, FS, fringe_band_hz(speed, LAMBDA_NM, FS))
    res = locate_peak(z, trace.s, edge_samples=20)
    assert res.found
    assert res.x_peak == pytest.approx(150.0, abs=1.0)


def test_no_surface_reports_not_found():
    z, v, speed = _slew(split=0.5, amp_v=0.0, seed=4)
    trace = process_channels(v, FS, fringe_band_hz(speed, LAMBDA_NM, FS))
    assert not locate_peak(z, trace.s, edge_samples=20).found


def _example_recording():
    return HomodyneRecording(
        mode="slew",
        values=np.arange(10.0).reshape(2, 5),
        sample_rate_hz=5000.0,
        t0_perf=100.0,
        channels=("ai0", "ai1"),
        meta={"speed_um_s": 400.0, "dummy": True},
        zlog_t_perf=np.array([100.0, 100.001]),
        zlog_z_um=np.array([0.0, 0.4]),
    )


def test_quantization_toggling_is_not_a_surface():
    """A DC level sitting on an LSB boundary of a coarse ADC range toggles
    throughout the record; the resulting envelope bursts must not be
    reported as a surface (this produced false peaks before)."""
    rng = np.random.default_rng(7)
    n = int(1.5 * FS)
    lsb = 20.0 / 2048  # USB-6008 RSE
    v = np.empty((2, n))
    v[0] = 0.8
    v[1] = np.round((0.65 + 0.5 * lsb + rng.normal(0, 3e-4, n)) / lsb) * lsb
    z = 400.0 * np.arange(n) / FS
    trace = process_channels(v, FS, fringe_band_hz(400.0, LAMBDA_NM, FS))
    assert not locate_peak(z, trace.s, edge_samples=20).found


def test_flat_record_is_not_a_surface():
    z = np.arange(1000.0)
    assert not locate_peak(z, np.zeros(1000)).found


def test_surface_found_above_quantization_bursts():
    z, v, speed = _slew(split=0.5, amp_v=0.05, seed=5)
    lsb = 20.0 / 2048
    v = np.round(v / lsb) * lsb
    trace = process_channels(v, FS, fringe_band_hz(speed, LAMBDA_NM, FS))
    res = locate_peak(z, trace.s, edge_samples=20)
    assert res.found
    assert res.x_peak == pytest.approx(150.0, abs=1.5)


@pytest.mark.parametrize("name", ["r.h5", "r.npz", "r"])
def test_recording_roundtrip(tmp_path, name):
    rec = _example_recording()
    path = save_recording(rec, tmp_path / name)
    assert path.suffix == (".npz" if name.endswith(".npz") else ".h5")
    out = load_recording(path)
    np.testing.assert_array_equal(out.values, rec.values)
    assert out.channels == ("ai0", "ai1")
    assert out.mode == "slew"
    assert out.meta == rec.meta
    np.testing.assert_allclose(out.z_um(), [0.0, 0.08, 0.16, 0.24, 0.32])


def test_h5_layout_is_plain_and_self_describing(tmp_path):
    import h5py

    path = save_recording(_example_recording(), tmp_path / "r.h5")
    with h5py.File(path, "r") as f:
        assert f["daq/values"].shape == (2, 5)
        assert f["daq/values"].attrs["units"] == "V"
        assert list(f["daq/values"].attrs["channels"]) == ["ai0", "ai1"]
        np.testing.assert_allclose(f["daq/t_s"][:], np.arange(5) / 5000.0)
        np.testing.assert_allclose(f["daq/z_um"][:], [0.0, 0.08, 0.16, 0.24, 0.32])
        assert f.attrs["sample_rate_hz"] == 5000.0
        assert f.attrs["speed_um_s"] == 400.0  # scalar meta exposed as attrs


def test_daq_validation_enforces_aggregate_rate():
    NI6008Multi(channels=("ai0", "ai1"), sample_rate_hz=5000).validate()
    with pytest.raises(ValueError):
        NI6008Multi(channels=("ai0", "ai1"), sample_rate_hz=6000).validate()
    assert effective_range_v("RSE", 1.0) == 10.0
    assert effective_range_v("DIFF", 1.0) == 1.0
