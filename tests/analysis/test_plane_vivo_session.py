import csv
import math
import threading

import pytest

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterfaceDummy
from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
from brillouin_system.guis.plane_accuracy_test.axial_recording import load_axial
from brillouin_system.guis.plane_accuracy_test.runner import Devices
from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
from brillouin_system.guis.plane_vivo_test.bounded_lens import BoundedLens, LensLimitError
from brillouin_system.guis.plane_vivo_test.session import Session, Settings, parse_locations, xy_move_for_target
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
    return Settings(**{**dict(ceiling_um=PLANE + 1000.0, retract_um=PLANE - 500.0, n_cycles=2, exposure_s=0.0,
                              settle_s=0.0), **kw})


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


def test_auto_test_searches_from_the_retract_position_then_runs_round_trips(tmp_path):
    dev = rig(lens_start=PLANE - 200.0)
    s = Session(dev, tmp_path / "run")
    s.plane_um = PLANE + 700.0      # stale: the search must replace it
    c = s.auto_test(settings(), threading.Event())
    r = rows(tmp_path / "run")
    assert r[0]["label"] == "search" and float(r[0]["start_um"]) == pytest.approx(PLANE - 500.0, abs=2.0)
    assert c["ended"] == "done" and c["cycles"] == 2 and c["found"] == 4
    assert dev.lens.get_position() == pytest.approx(PLANE - 500.0, abs=2.0)


def test_auto_test_without_a_plane_retracts_and_runs_nothing_more(tmp_path):
    dev = rig(peak=0.0)
    s = Session(dev, tmp_path / "run")
    c = s.auto_test(settings(search_um=800.0), threading.Event())
    assert c["ended"] == "no plane found" and c["cycles"] == 0
    assert len(rows(tmp_path / "run")) == 1
    assert dev.lens.get_position() == pytest.approx(PLANE - 500.0, abs=2.0)


def test_auto_test_counts_down_with_shutters_closed_then_reports_by_sound(tmp_path):
    class Shutters:
        def __init__(self):
            self.log = []
            self.sample = self

        def change_to_objective(self):
            self.log.append("objective")

        def open(self):
            self.log.append("open")

        def close_all(self):
            self.log.append("closed")

    dev = rig()
    dev.shutters = Shutters()
    heard = []
    s = Session(dev, tmp_path / "run", cue=lambda n: (heard.append(n), dev.shutters.log.append(n)))
    said = []
    s.say = said.append
    s.auto_test(settings(settle_s=2.0, n_cycles=1), threading.Event())
    assert heard == ["count", "count", "go", "success"]
    assert said == ["2 of 2 found"]
    assert dev.shutters.log[:4] == ["count", "count", "go", "objective"]
    assert dev.shutters.log[-2:] == ["closed", "success"]


def test_stop_during_the_countdown_opens_nothing(tmp_path):
    dev = rig()
    stop = threading.Event()
    heard = []
    s = Session(dev, tmp_path / "run", cue=lambda n: (heard.append(n), stop.set()))
    c = s.auto_test(settings(settle_s=5.0), stop)
    assert c["ended"] == "stopped" and heard == ["count", "stopped"]
    assert not (tmp_path / "run" / "slews.csv").exists()


def test_failed_search_sounds_as_a_failure(tmp_path):
    heard = []
    Session(rig(peak=0.0), tmp_path / "a", cue=heard.append).auto_test(settings(search_um=800.0), threading.Event())
    assert heard == ["go", "miss", "failed"]


def test_spoken_result_names_the_failure(tmp_path):
    said = []
    Session(rig(peak=0.0), tmp_path / "a", say=said.append).auto_test(settings(search_um=800.0), threading.Event())
    assert said == ["no plane found"]
    said.clear()
    s = Session(rig(), tmp_path / "b", say=said.append)
    s.auto_test(settings(n_cycles=3), threading.Event())
    assert said[0].startswith("6 of 6 found, short term sigma ") and said[0].endswith(" microns")


def test_speech_commands_quote_safely():
    from brillouin_system.guis.plane_vivo_test import cues
    original = cues.sys.platform
    try:
        cues.sys.platform = "win32"
        cmd = cues.speech_command("it's 6 of 6; found")
    finally:
        cues.sys.platform = original
    assert cmd[0] == "powershell" and "'its 6 of 6 found'" in cmd[-1]


class EyeOnStage:
    """Pupil reading that responds to the stage with the human interface's signs: stage +x moves the reading -x
    (dx_stage = -dx_laser), stage +y moves it +y. wrong=True flips the response. Every call is a new frame, and a
    frame shows the stage as it was late_frames ago, like the real cameras' 0.2 s exposure and delivery."""

    def __init__(self, dev, x, y, wrong=False, plane=PLANE, late_frames=0):
        self.dev, self.x, self.y, self.sign, self.plane = dev, x, y, -1.0 if wrong else 1.0, plane
        self.x0, self.y0, _ = dev.stage.get_position()
        self.late, self.history, self.n = late_frames, [], 0

    def __call__(self):
        self.n += 1
        self.history.append(self.dev.stage.get_position())
        sx, sy, _ = self.history[max(0, len(self.history) - 1 - self.late)]
        return dict(eye_x_mm=self.x - self.sign * (sx - self.x0) / 1000.0, eye_y_mm=self.y + self.sign * (sy - self.y0) / 1000.0,
                    eye_age_s=0.0, eye_plane_um=self.plane, eye_n=self.n)


class FixedEye:
    """A reading that does not respond to the stage. new_frames=False: the cameras have stopped."""

    def __init__(self, new_frames=True, **state):
        self.state, self.new_frames, self.n = state, new_frames, 0

    def __call__(self):
        self.n += int(self.new_frames)
        return dict(self.state, eye_n=self.n)


NO_READING = dict(eye_x_mm=None, eye_y_mm=None, eye_plane_um=None)


def test_centering_reaches_the_target_after_the_countdown(tmp_path):
    dev = rig()
    eye = EyeOnStage(dev, -0.5, 0.4)
    heard = []
    s = Session(dev, tmp_path / "a", eye_state=eye, cue=heard.append)
    c = s.auto_test(settings(center_first=True, target_r_mm=0.0, n_cycles=1, settle_s=1.0), threading.Event())
    assert c["ended"] == "done" and heard[:2] == ["count", "go"]
    now = eye()
    assert abs(now["eye_x_mm"]) < 0.01 and abs(now["eye_y_mm"]) < 0.01


def test_frames_from_before_the_move_do_not_stop_the_test(tmp_path):
    # the failure seen on the rig: stage moved, no plane search at any location
    dev = rig()
    eye = EyeOnStage(dev, 0.3, -0.2, late_frames=2)
    s = Session(dev, tmp_path / "a", eye_state=eye)
    locs = ((1.0, 0.0), (1.0, 90.0), (2.0, 270.0))
    c = s.auto_test(settings(center_first=True, locations=locs, n_cycles=1), threading.Event())
    assert c["found"] == 6 and c["n_failed"] == 0 and c["n_off_target"] == 0


@pytest.mark.parametrize("name, eye, note", [
    ("no_reading", FixedEye(**NO_READING), "no pupil found"),
    ("cameras_stopped", FixedEye(new_frames=False, eye_x_mm=0.1, eye_y_mm=0.0, eye_plane_um=PLANE), "no pupil found"),
    ("far", FixedEye(eye_x_mm=5.0, eye_y_mm=0.0, eye_plane_um=PLANE), "too far"),
    ("stage_does_nothing", FixedEye(eye_x_mm=0.8, eye_y_mm=0.0, eye_plane_um=PLANE), "off target"),
])
def test_eye_tracker_trouble_still_tests_the_plane(tmp_path, name, eye, note):
    dev = rig()
    said = []
    s = Session(dev, tmp_path / name, eye_state=eye, say=said.append)
    c = s.auto_test(settings(center_first=True, target_r_mm=0.0, n_cycles=1), threading.Event())
    assert c["found"] == 2 and c["n_off_target"] == 1 and c["n_failed"] == 0
    assert note in locations_csv(tmp_path / name)[0]["center_note"]
    assert said == ["2 of 2 found, 1 off target"]


@pytest.mark.parametrize("name, eye", [("no_reading", FixedEye(**NO_READING)),
                                       ("far", FixedEye(eye_x_mm=5.0, eye_y_mm=0.0, eye_plane_um=PLANE))])
def test_the_stage_does_not_move_on_a_bad_reading(tmp_path, name, eye):
    dev = rig()
    before = dev.stage.get_position()
    Session(dev, tmp_path / name, eye_state=eye).auto_test(settings(center_first=True, target_r_mm=0.0, n_cycles=1),
                                                           threading.Event())
    assert dev.stage.get_position() == before


def test_a_move_the_wrong_way_ends_the_run_without_opening_the_shutters(tmp_path):
    dev = rig()
    x0 = dev.stage.get_position()[0]
    said = []
    s = Session(dev, tmp_path / "a", eye_state=EyeOnStage(dev, 0.5, 0.0, wrong=True), say=said.append)
    locs = ((0.0, 0.0), (0.0, 0.0))
    c = s.auto_test(settings(center_first=True, locations=locs), threading.Event())
    assert c["cycles"] == 0 and said == ["No planes. centering moved the wrong way, stopping"]
    assert abs(dev.stage.get_position()[0] - x0) <= 501.0   # one move, not a runaway
    assert not (tmp_path / "a" / "slews.csv").exists()


def locations_csv(run):
    with open(run / "locations.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_parse_locations():
    assert parse_locations("0,0; 1,90;2,-45 ;") == ((0.0, 0.0), (1.0, 90.0), (2.0, -45.0))
    for bad in ("", "1", "1,2,3", "a,b", "9,0"):
        with pytest.raises(ValueError):
            parse_locations(bad)


def test_one_click_runs_every_location_in_random_order_and_saves_the_depth_estimate(tmp_path):
    dev = rig()
    eye = EyeOnStage(dev, 0.3, -0.2, plane=PLANE + 40.0)
    said = []
    s = Session(dev, tmp_path / "a", eye_state=eye, say=said.append)
    locs = ((0.0, 0.0), (1.0, 0.0), (1.0, 90.0), (1.0, 180.0))
    c = s.auto_test(settings(center_first=True, locations=locs, n_cycles=1), threading.Event())
    assert c["ended"] == "done" and c["found"] == 8 and c["n_failed"] == 0
    r = locations_csv(tmp_path / "a")
    assert {(float(x["r_mm"]), float(x["phi_deg"])) for x in r} == set(locs)
    assert all(float(x["z_tracker_um"]) == PLANE + 40.0 and abs(float(x["z_search_um"]) - PLANE) < 15 for x in r)
    last = eye()
    target = (float(r[-1]["r_mm"]) * math.cos(math.radians(float(r[-1]["phi_deg"]))),
              float(r[-1]["r_mm"]) * math.sin(math.radians(float(r[-1]["phi_deg"]))))
    assert math.dist((last["eye_x_mm"], last["eye_y_mm"]), target) < 0.21
    assert said == ["8 of 8 found"]
    assert dev.lens.get_position() == pytest.approx(PLANE - 500.0, abs=2.0)


def test_an_outer_location_can_come_first(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "a", eye_state=EyeOnStage(dev, 0.5, 0.0))
    c = s.auto_test(settings(center_first=True, locations=((2.5, 180.0),), n_cycles=1), threading.Event())
    assert c["ended"] == "done" and c["n_failed"] == 0


def test_a_second_click_can_start_far_from_where_the_first_ended(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "a", eye_state=EyeOnStage(dev, 0.0, 0.0))
    for phi in (90.0, 270.0):
        c = s.auto_test(settings(center_first=True, locations=((2.0, phi),), n_cycles=1), threading.Event())
        assert c["n_off_target"] == 0 and c["found"] == 2


def test_search_starts_in_front_of_the_tracker_plane(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "a", eye_state=EyeOnStage(dev, 0.0, 0.0, plane=PLANE + 20.0))
    s.auto_test(settings(center_first=True, target_r_mm=0.0, z_margin_um=300.0, n_cycles=1), threading.Event())
    first = rows(tmp_path / "a")[0]
    assert first["label"] == "search" and float(first["start_um"]) == pytest.approx(PLANE - 280.0, abs=3.0)


def test_tracker_past_the_cornea_costs_a_second_search_from_the_retract_position(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "a", eye_state=EyeOnStage(dev, 0.0, 0.0, plane=PLANE + 600.0))
    c = s.auto_test(settings(center_first=True, target_r_mm=0.0, z_margin_um=300.0, search_um=300.0, n_cycles=1),
                    threading.Event())
    r = rows(tmp_path / "a")
    assert [x["label"] for x in r[:2]] == ["search", "search_fallback"] and r[0]["status"] == "missed"
    assert locations_csv(tmp_path / "a")[0]["fell_back"] == "True"
    assert c["ended"] == "no plane found"       # 300 um of search from 500 um out cannot reach it


def test_unbelievable_depth_is_ignored_and_never_passes_the_limit(tmp_path):
    dev = rig()
    s = Session(dev, tmp_path / "a", eye_state=EyeOnStage(dev, 0.0, 0.0, plane=PLANE + 5000.0))
    c = s.auto_test(settings(center_first=True, target_r_mm=0.0, n_cycles=1), threading.Event())
    assert float(rows(tmp_path / "a")[0]["start_um"]) == pytest.approx(PLANE - 500.0, abs=3.0) and c["found"] == 2


def test_a_failed_location_is_skipped_and_two_without_pupil_or_plane_end_the_run(tmp_path):
    dev = rig(peak=0.0)
    said = []
    s = Session(dev, tmp_path / "a", eye_state=EyeOnStage(dev, 0.0, 0.0), say=said.append)
    locs = ((0.0, 0.0), (0.1, 0.0))
    c = s.auto_test(settings(center_first=True, locations=locs, search_um=800.0), threading.Event())
    assert c["n_failed"] == 2 and len(locations_csv(tmp_path / "a")) == 2 and "no plane" in said[0]
    dev = rig(peak=0.0)
    s = Session(dev, tmp_path / "b", eye_state=FixedEye(**NO_READING))
    locs = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 90.0))
    c = s.auto_test(settings(center_first=True, locations=locs, search_um=800.0), threading.Event())
    assert c["ended"] == "no pupil found" and len(locations_csv(tmp_path / "b")) == 2


def test_every_cue_exists():
    from brillouin_system.guis.plane_vivo_test.cues import PATTERNS
    assert {"count", "go", "miss", "success", "partial", "failed", "stopped"} <= set(PATTERNS)


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
