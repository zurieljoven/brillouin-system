"""Pure-Python tests for the predefined-measurement plan + QC metrics.

No Qt, no hardware, no spectrum fitting: the plan parsing/expansion and the
motion-delta quality metric are all duck-typed, so fakes exercise them.
"""
import random
from dataclasses import dataclass

import pytest

from brillouin_system.guis.human_interface.predefined_plan import (
    build_progress,
    cycle_actual_depths,
    cycle_motion_deltas,
    expand_plan,
    parse_plan_toml,
    passes_motion_limit,
    scan_mean_actual_depth_um,
    scan_passes_motion,
)


# --- fakes matching the scan surface these helpers read -------------------

@dataclass
class FakeReflection:
    found: bool
    event_z_um: float | None


@dataclass
class FakeCycle:
    reflection_in: FakeReflection | None
    reflection_out: FakeReflection | None
    measurement_index: int | None = None


@dataclass
class FakeMeasurement:
    lens_zaber_position: float | None


@dataclass
class FakeScan:
    id: str
    sweep_cycles: list
    measurements: list


def _scan(scan_id, deltas_from=None, positions=None):
    """Build a scan whose cycles realise the given in->out crossing deltas.

    deltas_from: list of (in_z, out_z) or None entries (None = a cycle missing
    a crossing). positions: frame lens positions; entry k is the frame for
    cycle k (so cycle_actual_depths = position - (in+out)/2).
    """
    cycles = []
    meas = []
    positions = positions or []
    for k, pair in enumerate(deltas_from or []):
        if pair is None:
            cycles.append(FakeCycle(FakeReflection(True, 100.0), None))
            continue
        zin, zout = pair
        mi = None
        if k < len(positions):
            mi = len(meas)
            meas.append(FakeMeasurement(positions[k]))
        cycles.append(FakeCycle(
            FakeReflection(True, zin), FakeReflection(True, zout),
            measurement_index=mi))
    return FakeScan(id=scan_id, sweep_cycles=cycles, measurements=meas)


# --- TOML parsing / expansion ---------------------------------------------

TEMPLATE = (
    "src/brillouin_system/measurement_templates/example_measurement_plan.toml")


def test_example_template_parses_and_expands():
    entries = parse_plan_toml(TEMPLATE)
    # 5 depths in the example, 5 replicates each except overrides are per-depth.
    steps = expand_plan(entries)
    ids = [s.id for s in steps]
    assert "R0_phi0_d50_n1" in ids
    assert "R0_phi0_d300_n5" in ids
    # depth 400 is fixed-mode with 8 cycles.
    d400 = [s for s in steps if s.depth_um == 400][0]
    assert d400.timed is False
    assert d400.cycles == 8
    # depth 300 overrides the motion limit to 40.
    d300 = [s for s in steps if s.depth_um == 300][0]
    assert d300.motion_limit_um == 40
    # inherited default limit elsewhere.
    d50 = [s for s in steps if s.depth_um == 50][0]
    assert d50.motion_limit_um == 30
    assert d50.timed is True


def _write(tmp_path, text):
    p = tmp_path / "plan.toml"
    p.write_text(text)
    return str(p)


def test_defaults_inheritance_and_override(tmp_path):
    path = _write(tmp_path, """
[defaults]
mode = "timed"
max_time_s = 12
replicates = 2
motion_limit_um = 25

[[step]]
depth_um = 60

[[step]]
depth_um = 120
motion_limit_um = 50
replicates = 1
""")
    steps = expand_plan(parse_plan_toml(path))
    assert [s.id for s in steps] == ["R0_phi0_d60_n1", "R0_phi0_d60_n2", "R0_phi0_d120_n1"]
    assert all(s.max_time_s == 12 for s in steps)
    assert steps[0].motion_limit_um == 25
    assert steps[2].motion_limit_um == 50


def test_ids_name_location_and_count_per_location(tmp_path):
    path = _write(tmp_path, """
[defaults]
mode = "fixed"
replicates = 1

[[step]]
depth_um = 150
R_mm = 2
phi_deg = 90

[[step]]
depth_um = 150
R_mm = 1.5
phi_deg = -75

[[step]]
depth_um = 150
R_mm = 2
phi_deg = 90
""")
    entries = parse_plan_toml(path)
    assert [s.id for s in expand_plan(entries)] == [
        "R2_phi90_d150_n1", "R1.5_phi-75_d150_n1", "R2_phi90_d150_n2"]
    scrambled = expand_plan(entries, scramble=True, rng=random.Random(0))
    assert {s.id: s.R_mm for s in scrambled} == {
        "R2_phi90_d150_n1": 2.0, "R1.5_phi-75_d150_n1": 1.5, "R2_phi90_d150_n2": 2.0}


def test_missing_depth_is_error(tmp_path):
    path = _write(tmp_path, "[[step]]\nmax_time_s = 5\n")
    with pytest.raises(ValueError, match="depth_um"):
        parse_plan_toml(path)


def test_unknown_key_is_error(tmp_path):
    path = _write(tmp_path, "[[step]]\ndepth_um = 10\nwidth = 3\n")
    with pytest.raises(ValueError, match="unknown key"):
        parse_plan_toml(path)


def test_timed_needs_positive_time(tmp_path):
    path = _write(tmp_path, '[[step]]\ndepth_um = 10\nmode = "timed"\nmax_time_s = 0\n')
    with pytest.raises(ValueError, match="max_time_s"):
        parse_plan_toml(path)


def test_no_steps_is_error(tmp_path):
    path = _write(tmp_path, "[defaults]\nmode = \"timed\"\n")
    with pytest.raises(ValueError, match="no steps"):
        parse_plan_toml(path)


def test_scramble_preserves_per_depth_order():
    path = None
    from brillouin_system.guis.human_interface.predefined_plan import PlanEntry
    entries = [
        PlanEntry(50, 3, True, None, 10, 0, 0, 0, 30),
        PlanEntry(100, 2, True, None, 10, 0, 0, 0, 30),
    ]
    steps = expand_plan(entries, scramble=True, rng=random.Random(1))
    # Within each depth, num must ascend in the order they appear.
    for depth in (50, 100):
        nums = [s.num for s in steps if s.depth_um == depth]
        assert nums == sorted(nums)
    assert len(steps) == 5


# --- motion metric --------------------------------------------------------

def test_cycle_motion_deltas_ignores_missing_crossings():
    scan = _scan("x", deltas_from=[(100, 110), None, (200, 180)])
    assert cycle_motion_deltas(scan.sweep_cycles) == [10.0, 20.0]


def test_passes_motion_limit_any_cycle_under():
    assert passes_motion_limit([50.0, 10.0], 30) is True     # one under 30
    assert passes_motion_limit([50.0, 40.0], 30) is False    # none under 30
    assert passes_motion_limit([], 30) is False              # no data -> fail
    assert passes_motion_limit([50.0], 0) is True            # gate off


def test_scan_passes_motion_and_actual_depth():
    # cycle 0: in100/out110 (plane 105), frame 155 -> depth 50
    # cycle 1: in200/out180 (plane 190), frame 245 -> depth 55
    scan = _scan("d", deltas_from=[(100, 110), (200, 180)],
                 positions=[155.0, 245.0])
    assert scan_passes_motion(scan, 30) is True   # deltas 10, 20 both < 30
    assert scan_passes_motion(scan, 15) is True   # cycle 0 delta 10 < 15
    assert scan_passes_motion(scan, 5) is False   # neither delta < 5
    assert cycle_actual_depths(scan) == pytest.approx([50.0, 55.0])
    assert scan_mean_actual_depth_um(scan) == pytest.approx(52.5)


def test_cycle_actual_depths_skips_frameless_and_single_crossing():
    # cycle 0 has a frame + both crossings; cycle 1 has no out-crossing.
    scan = _scan("d", deltas_from=[(100, 110), None], positions=[155.0])
    assert cycle_actual_depths(scan) == pytest.approx([50.0])


# --- progress join --------------------------------------------------------

def test_build_progress_counts_and_matching():
    from brillouin_system.guis.human_interface.predefined_plan import PlanEntry
    steps = expand_plan([PlanEntry(50, 2, True, None, 10, 0, 0, 0, 30)])
    # step R0_phi0_d50_n1 taken with one within-limit frame; R0_phi0_d50_n2 not taken.
    scans = [
        _scan("R0_phi0_d50_n1", deltas_from=[(100, 115)], positions=[155.0]),
        _scan("R0_phi0_d999_n1", deltas_from=[(100, 105)], positions=[155.0]),
    ]
    report = build_progress(steps, scans)
    assert report.planned_total == 2
    assert report.taken_count == 1
    assert report.frames_total == 1
    assert report.frames_pass == 1
    assert report.remaining_ids() == ["R0_phi0_d50_n2"]
    assert report.unmatched_scan_ids == ["R0_phi0_d999_n1"]


def test_build_progress_per_frame_pass_fail_within_one_scan():
    from brillouin_system.guis.human_interface.predefined_plan import PlanEntry
    steps = expand_plan([PlanEntry(50, 1, True, None, 10, 0, 0, 0, 30)])
    # One scan, three frames: deltas 10 (pass), 20 (pass), 40 (fail) vs limit 30.
    scan = _scan("R0_phi0_d50_n1",
                 deltas_from=[(100, 110), (200, 180), (300, 340)],
                 positions=[155.0, 245.0, 350.0])
    report = build_progress(steps, [scan])
    row = report.rows[0]
    assert row.taken is True
    assert row.n_frames == 3
    assert row.n_pass_frames == 2          # per-FRAME, not the whole scan
    assert report.frames_pass == 2 and report.frames_total == 3
    # points carry each frame's own verdict
    assert [p.passed for p in report.points] == [True, True, False]
    assert row.min_delta_um == 10.0
