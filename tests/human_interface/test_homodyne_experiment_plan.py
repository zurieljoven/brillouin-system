import pytest

from brillouin_system.guis.homodyne_test.experiment_plan import (
    MAX_XY_MOVE_UM,
    MAX_Z_MOVE_UM,
    build_experiment_steps,
    radii_range,
    xy_stage_move_um,
    z_lens_move_um,
)


def test_radii_inclusive_half_mm_steps():
    assert radii_range(0.0, 4.0, 0.5) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    with pytest.raises(ValueError):
        radii_range(0.0, 4.0, 0.0)


def test_experiment_order_out_and_back_five_loops():
    radii = radii_range(0.0, 4.0, 0.5)
    steps = build_experiment_steps(radii, 5)
    assert len(steps) == 90
    first_loop = [(s.r_mm, s.direction) for s in steps[:18]]
    assert first_loop[:9] == [(r, "fwd") for r in radii]
    assert first_loop[8:10] == [(4.0, "fwd"), (4.0, "bwd")]      # 4.0 fwd then 4.0 bwd
    assert first_loop[9:] == [(r, "bwd") for r in reversed(radii)]
    assert steps[18].loop == 2 and (steps[18].r_mm, steps[18].direction) == (0.0, "fwd")
    for r in radii:                                                # 5 fwd + 5 bwd each
        assert sum(s.r_mm == r and s.direction == "fwd" for s in steps) == 5
        assert sum(s.r_mm == r and s.direction == "bwd" for s in steps) == 5


def test_xy_move_matches_main_gui_signs():
    # laser at the pupil center, target R=1 mm at phi=0: laser must go +1 mm in x,
    # so the stage moves -1000 um in x (stage +X moves the laser -X)
    dx, dy, dist, clamped = xy_stage_move_um((0.0, 0.0), 1.0, 0.0)
    assert dx == pytest.approx(-1000.0)
    assert dy == pytest.approx(0.0, abs=1e-9)
    assert dist == pytest.approx(1.0)
    assert not clamped
    dx, dy, _, _ = xy_stage_move_um((0.0, 0.0), 1.0, 90.0)
    assert dy == pytest.approx(1000.0)


def test_xy_move_clamped():
    dx, dy, dist, clamped = xy_stage_move_um((-5.0, 0.0), 0.0, 0.0)
    assert clamped and dist == pytest.approx(5.0)
    assert abs(dx) == pytest.approx(MAX_XY_MOVE_UM)


def test_z_move_sign_and_clamp():
    # delta_c is +0.5 mm, target -1 mm: focus must go 1.5 mm deeper -> lens +1500 um
    assert z_lens_move_um(0.5, -1.0) == pytest.approx(1500.0)
    assert z_lens_move_um(10.0, -1.0) == pytest.approx(MAX_Z_MOVE_UM)


def test_repeat_steps_alternate_fwd_bwd():
    from brillouin_system.guis.homodyne_test.experiment_plan import build_repeat_steps

    steps = build_repeat_steps(50)
    assert len(steps) == 100
    assert [s.direction for s in steps[:4]] == ["fwd", "bwd", "fwd", "bwd"]
    assert sum(s.direction == "fwd" for s in steps) == 50
    assert all(s.r_mm is None for s in steps)
    assert (steps[-1].loop, steps[-1].direction) == (50, "bwd")
