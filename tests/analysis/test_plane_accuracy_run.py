import csv
import json
import threading

import numpy as np
import pytest

from brillouin_system.guis.homodyne_test.experiment_plan import StageRef
from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
from brillouin_system.guis.plane_accuracy_test.axial_recording import load_axial
from brillouin_system.guis.plane_accuracy_test.runner import Devices, RunConfig, run_experiment
from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
from brillouin_system.guis.plane_accuracy_test.trial_plan import KINDS, axial_positions_um, build_trials
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterfaceDummy
from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
from brillouin_system.scan_managers.homodyne_recording import load_recording


def test_every_block_holds_every_radius_and_kind_once():
    radii = [0.0, 0.5, 1.0]
    trials = build_trials(radii, 4, seed=7)
    assert len(trials) == 4 * 3 * 3
    assert [t.index for t in trials] == list(range(len(trials)))
    for b in range(1, 5):
        cells = {(t.r_mm, t.kind) for t in trials if t.block == b}
        assert cells == {(r, k) for r in radii for k in KINDS}


def test_order_is_reproducible_from_the_seed_and_differs_between_blocks():
    a = build_trials([0.0, 0.5, 1.0, 1.5], 2, seed=7)
    assert a == build_trials([0.0, 0.5, 1.0, 1.5], 2, seed=7)
    assert a != build_trials([0.0, 0.5, 1.0, 1.5], 2, seed=8)
    assert [(t.r_mm, t.kind) for t in a[:12]] != [(t.r_mm, t.kind) for t in a[12:]]


def test_axial_positions_span_the_plane():
    z = axial_positions_um(1000.0, 150.0, 200.0, 10.0)
    assert z[0] == 850.0 and z[-1] == 1200.0 and z.size == 36


def rig(plane=8000.0):
    lens = SimZaberLens(start_um=plane)
    stage = ZaberHumanInterfaceDummy(home_on_connect=False)
    stage.move_abs(12000.0, 10000.0, 12000.0)
    daq = SimHomodyneDAQ(lens, plane_um=plane, peak_amplitude_v=0.2, noise_std_v=0.006, seed=1)
    return Devices(daq=daq, lens=lens, stage=stage, camera=SimAndor(lens, plane_um=plane, seed=2)), plane


def config(plane, **kw):
    return RunConfig(center=StageRef(12000.0, 10000.0, 12000.0, plane), edge=StageRef(12500.0, 10000.0, 12000.0, plane),
                     r_step_mm=0.5, n_blocks=1, seed=3, half_window_um=200.0, exposure_s=0.0,
                     axial_before_um=60.0, axial_after_um=60.0, axial_step_um=20.0, **kw)


def test_full_run_saves_every_trial_and_finds_the_plane(tmp_path):
    dev, plane = rig()
    run = tmp_path / "run"
    shown = []
    counts = run_experiment(dev, config(plane), run, threading.Event(), on_frame=lambda f, z: shown.append((f, z)))
    assert counts == dict(n_done=6, n_found=4, n_missed=0, n_axial=2)
    assert len(shown) == 2 * 7     # every axial frame reaches the display, with its z
    assert shown[0][0].ndim == 2 and shown[0][1] == pytest.approx(plane - 60.0, abs=2.0)

    rows = list(csv.DictReader(open(run / "run_log.csv")))
    assert len(rows) == 6
    plan = json.load(open(run / "plan.json"))
    assert [t["kind"] for t in plan["trials"]] == [r["kind"] for r in rows]
    for r in rows:
        assert (run / r["file"]).exists()
        if r["kind"] == "axial":
            a = load_axial(run / r["file"])
            assert a.frames.shape[0] == 7
            assert a.z_um[np.argmax(a.frame_sums())] == pytest.approx(plane, abs=25.0)
        else:
            assert float(r["finder_z_um"]) == pytest.approx(plane, abs=8.0)
            rec = load_recording(run / r["file"])
            assert rec.values.shape[0] == 2 and rec.meta["kind"] == r["kind"]

    # only X/Y moved, and the run ends back at the center
    assert dev.stage.get_position() == pytest.approx((12000.0, 10000.0, 12000.0))
    assert dev.lens.get_position() == pytest.approx(plane, abs=2.0)


def test_stop_ends_the_run_and_still_returns_to_the_center(tmp_path):
    dev, plane = rig()
    stop = threading.Event()
    seen = []

    def on_trial(row, rec):
        seen.append(row)
        stop.set()

    counts = run_experiment(dev, config(plane), tmp_path / "run", stop, on_trial=on_trial)
    assert counts["n_done"] == 1 and len(seen) == 1
    assert dev.stage.get_position()[0] == pytest.approx(12000.0)


def test_run_folder_is_never_overwritten(tmp_path):
    dev, plane = rig()
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        run_experiment(dev, config(plane), tmp_path / "run", threading.Event())


def test_scan_window_too_short_for_the_background_is_refused(tmp_path):
    dev, plane = rig()
    cfg = RunConfig(center=StageRef(0.0, 0.0, 0.0, plane), edge=StageRef(500.0, 0.0, 0.0, plane), half_window_um=100.0)
    with pytest.raises(ValueError, match="at least"):
        run_experiment(dev, cfg, tmp_path / "run", threading.Event())
