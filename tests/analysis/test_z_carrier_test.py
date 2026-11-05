import csv
import json
import threading
from collections import Counter

import numpy as np
import pytest

from brillouin_system.devices.shutter_device import ShutterManagerDummy
from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
from brillouin_system.guis.plane_accuracy_test.axial_recording import load_axial
from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
from brillouin_system.guis.z_carrier_test.motion import SimAxisMover, WindowError
from brillouin_system.guis.z_carrier_test.protocol import ProtocolConfig
from brillouin_system.guis.z_carrier_test.runner import Rig, park_lens_on_plane, run_protocol
from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens
from brillouin_system.scan_managers.homodyne_recording import load_recording

PLANE = 8000.0


def rig(lens_start=PLANE):
    lens, stage = SimZaberLens(start_um=lens_start, seed=1), SimZaberLens(start_um=12000.0, seed=2)
    daq = SimHomodyneDAQ(lens, plane_um=PLANE, peak_amplitude_v=0.2, noise_std_v=0.006, seed=3)
    return Rig(daq, SimAxisMover("lens", lens), SimAxisMover("stage", stage),
               SimAndor(lens, plane_um=PLANE, seed=4), ShutterManagerDummy("human_interface"))


def small(**kw):
    base = dict(n_blocks=1, stationary_s=0.3, slew_half_um=40.0, speeds_um_s=(400.0,), andor_half_um=4.0,
                andor_step_um=2.0, exposure_s=0.01)
    base.update(kw)
    return ProtocolConfig(**base)


def test_steps_every_slew_once_per_block_and_sweeps_at_both_ends():
    cfg = ProtocolConfig(n_blocks=3, speeds_um_s=(200.0, 400.0), seed=7)
    steps = cfg.steps()
    assert [(s.kind, s.axis) for s in steps[:2]] == [("andor", "lens"), ("andor", "stage")]
    assert [(s.kind, s.axis) for s in steps[-2:]] == [("andor", "lens"), ("andor", "stage")]
    for b in (1, 2, 3):
        blk = [s for s in steps if s.block == b]
        assert blk[0].kind == "stationary"
        cells = Counter((s.axis, s.direction, s.speed_um_s) for s in blk[1:])
        assert len(cells) == 8 and set(cells.values()) == {1}
    assert [s.index for s in steps] == list(range(len(steps)))
    assert steps == ProtocolConfig(n_blocks=3, speeds_um_s=(200.0, 400.0), seed=7).steps()
    assert len({s.name() for s in steps}) == len(steps)


def test_validate_limits():
    ProtocolConfig().validate()
    with pytest.raises(ValueError):
        ProtocolConfig(speeds_um_s=(1000.0,)).validate()       # stage-Z fringe above Nyquist at 5 kS/s
    with pytest.raises(ValueError):
        ProtocolConfig(slew_half_um=500.0).validate()
    with pytest.raises(ValueError):
        ProtocolConfig(andor_step_um=0.0).validate()


def test_mover_refuses_targets_outside_window():
    m = SimAxisMover("stage", SimZaberLens(start_um=100.0))
    with pytest.raises(WindowError):
        m.move_abs(110.0)                                     # no window yet
    m.set_window(50.0, 150.0)
    m.move_abs(120.0)
    with pytest.raises(WindowError):
        m.slew_to(151.0, 400.0)


def test_run_protocol_writes_every_step_and_returns_axes(tmp_path):
    r = rig()
    lens0, stage0 = r.lens.get_position(), r.stage.get_position()
    cfg = small()
    counts = run_protocol(r, cfg, tmp_path / "run", threading.Event())
    steps = cfg.steps()
    assert counts == dict(n_done=len(steps), n_slew=4, n_andor=4, n_stationary=1)
    rows = list(csv.DictReader(open(tmp_path / "run" / "run_log.csv")))
    assert [row["file"] for row in rows] == [s.name() + ".h5" for s in steps]
    plan = json.load(open(tmp_path / "run" / "plan.json"))
    assert plan["mode"] == "z_carrier_test" and len(plan["steps"]) == len(steps)
    assert abs(r.lens.get_position() - lens0) < 2.0 and abs(r.stage.get_position() - stage0) < 2.0

    for s in steps:
        path = tmp_path / "run" / (s.name() + ".h5")
        if s.kind == "andor":
            a = load_axial(path)
            assert a.meta["axis"] == s.axis and a.frames.shape[0] == cfg.andor_frames()
            assert np.all(np.diff(a.z_um) > 0)
        elif s.kind == "slew":
            rec = load_recording(path)
            z = rec.z_um() - rec.meta["z0_um"]
            assert rec.meta["axis"] == s.axis and rec.meta["direction"] == s.direction
            assert z.min() < -30.0 and z.max() > 30.0           # crossed the start position on the moving axis
            assert np.sign(z[-1] - z[0]) == s.direction


def test_lens_slews_find_the_plane_and_stage_slews_leave_the_lens_alone(tmp_path):
    r = rig()
    run_protocol(r, small(), tmp_path / "run", threading.Event())
    rows = list(csv.DictReader(open(tmp_path / "run" / "run_log.csv")))
    lens_slews = [row for row in rows if row["kind"] == "slew" and row["axis"] == "lens"]
    assert all(abs(float(row["peak_rel_um"])) < 5.0 for row in lens_slews)
    for row in rows:
        if row["kind"] == "slew" and row["axis"] == "stage":
            assert abs(float(row["lens_um"]) - PLANE) < 2.0


def test_park_moves_the_lens_onto_the_plane():
    r = rig(lens_start=PLANE + 30.0)
    z, rec = park_lens_on_plane(r, 150.0, 400.0, threading.Event())
    assert z is not None and abs(z - PLANE) < 5.0
    assert abs(r.lens.get_position() - z) < 2.0


def test_stop_ends_the_run_and_returns_axes(tmp_path):
    r = rig()
    lens0, stage0 = r.lens.get_position(), r.stage.get_position()
    stop = threading.Event()
    seen = []

    def on_result(row, rec):
        seen.append(row["file"])
        if len(seen) == 3:
            stop.set()

    counts = run_protocol(r, small(), tmp_path / "run", stop, on_result=on_result)
    assert counts["n_done"] == 3
    assert abs(r.lens.get_position() - lens0) < 2.0 and abs(r.stage.get_position() - stage0) < 2.0
