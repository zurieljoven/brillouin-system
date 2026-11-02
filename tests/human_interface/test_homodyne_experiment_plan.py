import numpy as np
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


def _refs(r_edge_mm=6.5, R_mm=12.7, angle_deg=30.0):
    from brillouin_system.guis.homodyne_test.experiment_plan import StageRef

    a = np.deg2rad(angle_deg)
    center = StageRef(12000.0, 10000.0, 12000.0, 14000.0)
    sag_um = 1000.0 * (R_mm - np.sqrt(R_mm ** 2 - r_edge_mm ** 2))
    edge = StageRef(12000.0 + 1000.0 * r_edge_mm * np.cos(a), 10000.0 + 1000.0 * r_edge_mm * np.sin(a),
                    12000.0, 14000.0 + sag_um)
    return center, edge


def test_stage_ref_radii_end_at_edge():
    from brillouin_system.guis.homodyne_test.experiment_plan import stage_ref_radii

    center, edge = _refs(r_edge_mm=6.5)
    radii = stage_ref_radii(center, edge, 0.5)
    assert radii[0] == 0.0 and radii[-1] == pytest.approx(6.5) and len(radii) == 14
    center, edge = _refs(r_edge_mm=6.3)
    radii = stage_ref_radii(center, edge, 0.5)
    assert radii[-2] == pytest.approx(6.0) and radii[-1] == pytest.approx(6.3)
    with pytest.raises(ValueError):
        stage_ref_radii(center, center, 0.5)


def test_stage_ref_sphere_recovers_radius_and_sag():
    from brillouin_system.guis.homodyne_test.experiment_plan import (
        stage_ref_lens_um, stage_ref_sphere_radius_mm, stage_ref_xy_um,
    )

    center, edge = _refs(r_edge_mm=6.5, R_mm=12.7)
    assert stage_ref_sphere_radius_mm(center, edge) == pytest.approx(12.7)
    assert stage_ref_lens_um(center, edge, 0.0) == pytest.approx(center.lens_um)
    assert stage_ref_lens_um(center, edge, 6.5) == pytest.approx(edge.lens_um)
    assert stage_ref_lens_um(center, edge, 4.0) - center.lens_um == pytest.approx(1000 * (12.7 - np.sqrt(12.7 ** 2 - 16)))
    x, y = stage_ref_xy_um(center, edge, 3.25)                      # halfway along the line
    assert x == pytest.approx(0.5 * (center.stage_x_um + edge.stage_x_um))
    assert y == pytest.approx(0.5 * (center.stage_y_um + edge.stage_y_um))


def test_stage_ref_flat_and_negative_sag():
    from brillouin_system.guis.homodyne_test.experiment_plan import StageRef, stage_ref_lens_um, stage_ref_sphere_radius_mm

    center = StageRef(0.0, 0.0, 0.0, 14000.0)
    flat = StageRef(5000.0, 0.0, 0.0, 14000.0)
    assert stage_ref_sphere_radius_mm(center, flat) is None
    assert stage_ref_lens_um(center, flat, 2.0) == 14000.0
    down = StageRef(5000.0, 0.0, 0.0, 13000.0)                      # edge plane at lower lens z
    assert stage_ref_lens_um(center, down, 5.0) == pytest.approx(13000.0)
    assert stage_ref_lens_um(center, down, 2.5) < 14000.0
