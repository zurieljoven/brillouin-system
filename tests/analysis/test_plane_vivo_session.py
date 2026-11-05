import csv
import threading

import pytest

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterfaceDummy
from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
from brillouin_system.guis.plane_accuracy_test.axial_recording import load_axial
from brillouin_system.guis.plane_accuracy_test.runner import Devices
from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
from brillouin_system.guis.plane_vivo_test.bounded_lens import BoundedLens, LensLimitError
from brillouin_system.guis.plane_vivo_test.session import Session, Settings, xy_move_for_target
from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
from brillouin_system.scan_managers.homodyne_recording import load_recording

PLANE = 8000.0


class FakeAxis:
    def __init__(self):
        self.calls = []

    def move_absolute(self, position, unit, wait_until_idle=True, velocity=0, velocity_unit=None):
        self.calls.append((position, wait_until_idle, velocity))


class FakeLens:
    def __init__(self, z=1000.0):
        self.z, self.axis, self.moves = z, FakeAxis(), []

    def get_position(self):
        return self.z

    def move_abs(self, z):
        self.moves.append(z)
        self.z = z


def rig(peak=0.2, lens_start=PLANE - 500.0, ceiling=PLANE + 1000.0):
    inner = SimZaberLens(start_um=lens_start)
    lens = BoundedLens(inner)
    lens.set_ceiling(ceiling)
    stage = ZaberHumanInterfaceDummy(home_on_connect=False)
    stage.move_abs(12000.0, 10000.0, 12000.0)
    daq = SimHomodyneDAQ(inner, plane_um=PLANE, peak_amplitude_v=peak, noise_std_v=0.006, seed=1)
    daq.configure(sample_rate_hz=5000.0)
    return Devices(daq=daq, lens=lens, stage=stage, camera=SimAndor(inner, plane_um=PLANE, seed=2))


def settings(**kw):
    return Settings(**{**dict(ceiling_um=PLANE + 1000.0, retract_um=PLANE - 500.0, n_cycles=2, exposure_s=0.0), **kw})


def rows(run):
    with open(run / "slews.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_lens_refuses_to_move_without_a_limit_and_past_it():
    lens = BoundedLens(FakeLens())
    with pytest.raises(LensLimitError):
        lens.move_abs(900.0)
    lens.set_ceiling(1100.0)
    lens.move_abs(1100.0)
    with pytest.raises(LensLimitError):
        lens.move_abs(1100.5)
    with pytest.raises(LensLimitError):
        lens.move_rel(1.0)


def test_lens_above_the_limit_can_only_move_back():
    inner = FakeLens(1200.0)
    lens = BoundedLens(inner)
    lens.set_ceiling(1100.0)
    with pytest.raises(LensLimitError):
        lens.move_abs(1300.0)
    lens.move_abs(1150.0)
    assert inner.z == 1150.0


def test_slew_is_an_absolute_move_stopped_by_the_device_at_the_limit():
    inner = FakeLens(1000.0)
    lens = BoundedLens(inner)
    lens.set_ceiling(1100.0)
    lens.start_slewing_guarded(400.0, 500.0)
    assert inner.axis.calls == [(1100.0, False, 400.0)]
    inner.axis.calls.clear()
    lens.start_slewing_guarded(-400.0, 500.0)
    assert inner.axis.calls == [(500.0, False, 400.0)]


def test_move_xy_matches_the_human_interface_signs_and_limit():
    dx, dy, clamped = xy_move_for_target(1.0, 90.0, (0.5, 0.0))
    assert dx == pytest.approx(500.0) and dy == pytest.approx(1000.0) and not clamped
    dx, dy, clamped = xy_move_for_target(5.0, 0.0, (0.0, 0.0))
    assert dx == pytest.approx(-3500.0) and clamped


def test_retract_must_be_below_the_limit_and_approach_clear_the_background_window():
    with pytest.raises(ValueError, match="retract"):
        settings(retract_um=PLANE + 2000.0).validate()
    with pytest.raises(ValueError, match="Approach"):
        settings(approach_um=60.0).validate()


def test_single_find_saves_the_slew_and_returns_to_the_start(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "run", "S01")
    slew = s.single_find(settings(), 1.0, threading.Event())
    assert slew.status == "found" and slew.z_um == pytest.approx(PLANE, abs=15.0)
    assert s.plane_um == slew.z_um
    assert dev.lens.get_position() == pytest.approx(PLANE - 500.0, abs=2.0)
    row = rows(tmp_path / "run")[0]
    assert float(row["peak_snr"]) > 20 and row["status"] == "found"
    rec = load_recording(tmp_path / "run" / row["file"])
    assert rec.meta["subject"] == "S01" and rec.meta["settings"]["n_sigma"] == 20.0


def test_round_trips_save_every_slew_and_frame_and_retract(tmp_path):
    dev = rig()
    seen = []
    s = Session(dev, tmp_path / "run", on_slew=seen.append)
    s.plane_um = PLANE
    c = s.round_trips(settings(frame_depth_um=50.0), threading.Event())
    assert c["cycles"] == 2 and c["found"] == 4 and c["ended"] == "done"
    assert [r["direction"] for r in rows(tmp_path / "run")] == ["1.0", "-1.0"] * 2
    frames = sorted((tmp_path / "run").glob("*_frame.h5"))
    assert len(frames) == 2 and load_axial(frames[0]).z_um[0] == pytest.approx(PLANE + 50.0, abs=20.0)
    assert dev.lens.get_position() == pytest.approx(PLANE - 500.0, abs=2.0)
    assert all(z.status == "found" for z in seen)


def test_misses_are_saved_and_end_the_round_trips(tmp_path):
    dev = rig(peak=0.0)
    s = Session(dev, tmp_path / "run")
    s.plane_um = PLANE
    c = s.round_trips(settings(n_cycles=10, max_misses=2), threading.Event())
    assert c["ended"].startswith("2 forward finds") and c["found"] == 0
    r = rows(tmp_path / "run")
    assert {x["status"] for x in r} == {"missed"} and all((tmp_path / "run" / x["file"]).exists() for x in r)
    assert dev.lens.get_position() == pytest.approx(PLANE - 500.0, abs=2.0)


def test_far_crossing_is_saved_as_rejected_and_keeps_the_estimate(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "run")
    s.plane_um = PLANE + 400.0
    st = settings(gate_um=50.0, approach_um=500.0)
    slew = s.find(st, 1.0, PLANE - 100.0, 400.0, threading.Event())
    assert slew.status == "rejected" and s.plane_um == PLANE + 400.0


def test_single_find_at_a_new_spot_is_not_gated(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "run")
    s.plane_um = PLANE + 700.0
    slew = s.single_find(settings(gate_um=50.0), 1.0, threading.Event())
    assert slew.status == "found" and s.plane_um == slew.z_um


def test_the_lens_limit_holds_during_a_miss(tmp_path):
    dev = rig(peak=0.0, ceiling=PLANE + 100.0)
    s = Session(dev, tmp_path / "run")
    s.plane_um = PLANE
    st = settings(ceiling_um=PLANE + 100.0, retract_um=PLANE - 500.0, n_cycles=1, max_misses=5, approach_um=100.0)
    slew = s.find(st, 1.0, PLANE - 100.0, 1000.0, threading.Event())
    assert slew.row["truncated"] and slew.row["search_um"] == pytest.approx(200.0, abs=2.0)
    # the simulated lens stops itself with a 30 ms software guard, so it overshoots by 12 um or more under load;
    # the real path stops in the device, see test_slew_is_an_absolute_move_stopped_by_the_device_at_the_limit
    assert max(slew.rec.zlog_z_um) <= PLANE + 150.0


def test_stop_ends_the_round_trips_and_retracts(tmp_path):
    dev = rig()
    stop = threading.Event()
    s = Session(dev, tmp_path / "run", on_slew=lambda _: stop.set())
    s.plane_um = PLANE
    c = s.round_trips(settings(n_cycles=50), stop)
    assert c["ended"] == "stopped" and c["cycles"] == 1 and c["missed"] == 0
    assert dev.lens.get_position() == pytest.approx(PLANE - 500.0, abs=2.0)
