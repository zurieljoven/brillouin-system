import csv
import json
import threading

import numpy as np
import pytest

from brillouin_system.devices.shutter_device import ShutterManagerDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterfaceDummy
from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
from brillouin_system.guis.plane_accuracy_test.sim_camera import SimAndor
from brillouin_system.guis.z_carrier_test.geometry import fit_sphere, scan_locations, scan_radii_mm, shuffled
from brillouin_system.guis.z_carrier_test.motion import SimAxisMover
from brillouin_system.guis.z_carrier_test.protocol import ProtocolConfig
from brillouin_system.guis.z_carrier_test.runner import Location, Rig, check_locations, run_locations
from brillouin_system.patient_movement_analysis.simulated_devices import SimZaberLens

CX, CY, Z = 10000.0, 9900.0, 0.1


def sphere_point(name, t_um, d_um=0.0, R=9600.0, t0=-220.0, d0=0.0, z0=1090.0, ux=-1.0):
    """Points along stage -X from the center (like the Nov 5 runs), lens z on a sphere with an offset apex."""
    lens = z0 + R - np.sqrt(R ** 2 - (t_um - t0) ** 2 - (d_um - d0) ** 2)
    return Location(name, CX + ux * t_um, CY + d_um, Z, lens)


def test_two_points_give_the_old_two_point_sphere():
    c, e = sphere_point("center", 0.0), sphere_point("edge", 3000.0)
    f = fit_sphere(c, e)
    sag = e.lens_um - c.lens_um
    assert f.method == "two-point" and f.t0_um == 0.0
    assert f.r_um == pytest.approx((3000.0 ** 2 + sag ** 2) / (2 * sag))
    assert f.lens_at(e.stage_x_um, e.stage_y_um) == pytest.approx(e.lens_um)


def test_points_on_the_line_recover_the_offset_apex():
    pts = [sphere_point(n, t) for n, t in (("center", 0), ("p1", 1000), ("p2", 2000), ("edge", 3000))]
    f = fit_sphere(pts[0], pts[-1], pts[1:-1])
    assert f.method == "line"
    assert f.r_um == pytest.approx(9600.0, rel=1e-3) and f.t0_um == pytest.approx(-220.0, abs=2.0)
    assert f.rms_um < 0.1
    for t in (500.0, 1500.0, 2500.0):           # interpolation between the set points is exact on a sphere
        p = sphere_point("x", t)
        assert f.lens_at(p.stage_x_um, p.stage_y_um) == pytest.approx(p.lens_um, abs=0.1)


def test_two_point_guess_is_what_the_fit_fixes():
    c, e, mid = sphere_point("center", 0), sphere_point("edge", 3000), sphere_point("p1", 1500)
    two, fit = fit_sphere(c, e), fit_sphere(c, e, [mid])
    q = sphere_point("q", 1500)
    err_two = two.lens_at(q.stage_x_um, q.stage_y_um) - q.lens_um
    err_fit = fit.lens_at(q.stage_x_um, q.stage_y_um) - q.lens_um
    assert abs(err_two) > 5.0 and abs(err_fit) < 0.5    # the apex offset biases the two-point guess mid-way


def test_off_line_points_fit_the_apex_across_the_line():
    kw = dict(d0=150.0)
    pts = [sphere_point("center", 0, **kw), sphere_point("edge", 3000, **kw), sphere_point("p1", 1000, 600, **kw),
           sphere_point("p2", 2000, -500, **kw), sphere_point("p3", 1500, 0, **kw)]
    f = fit_sphere(pts[0], pts[1], pts[2:])
    assert f.method == "2d"
    assert abs(f.d0_um) == pytest.approx(150.0, abs=5.0) and f.t0_um == pytest.approx(-220.0, abs=5.0)
    q = sphere_point("q", 2500, 300, **kw)        # a new point off the line
    assert f.lens_at(q.stage_x_um, q.stage_y_um) == pytest.approx(q.lens_um, abs=0.5)


def test_stage_z_must_not_change_between_points():
    c, e = sphere_point("center", 0), sphere_point("edge", 3000)
    e = Location(e.name, e.stage_x_um, e.stage_y_um, Z + 20.0, e.lens_um)
    with pytest.raises(ValueError):
        fit_sphere(c, e)


def test_flat_sample_interpolates_linearly():
    c = Location("center", CX, CY, Z, 1000.0)
    e = Location("edge", CX - 3000, CY, Z, 1000.4)
    f = fit_sphere(c, e)
    assert f.method == "flat" and f.lens_at(CX - 1500, CY) == pytest.approx(1000.2)


def test_scan_grid_and_order():
    assert scan_radii_mm(3000.0, 0.5) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert scan_radii_mm(3000.1, 0.5)[-1] == pytest.approx(3.0001) and len(scan_radii_mm(3000.1, 0.5)) == 7
    pts = [sphere_point("center", 0), sphere_point("p1", 1500), sphere_point("edge", 3000)]
    f = fit_sphere(pts[0], pts[2], [pts[1]])
    locs = scan_locations(f, 1.0)
    assert [l.name for l in locs] == ["r0.00", "r1.00", "r2.00", "r3.00"]
    assert locs[-1].stage_x_um == pytest.approx(CX - 3000.0) and all(l.stage_z_um == Z for l in locs)
    check_locations(locs)
    s = shuffled(locs, 3)
    assert s[0] == locs[0] and sorted(l.name for l in s) == sorted(l.name for l in locs)


def test_run_locations_records_every_location_and_returns_to_the_center(tmp_path):
    lens, stage = SimZaberLens(start_um=8000.0, seed=1), SimZaberLens(start_um=12000.0, seed=2)
    xy = ZaberHumanInterfaceDummy(home_on_connect=False)
    xy.move_abs(x=CX, y=CY)
    daq = SimHomodyneDAQ(lens, plane_um=8000.0, peak_amplitude_v=0.2, noise_std_v=0.006, seed=3)
    rig = Rig(daq, SimAxisMover("lens", lens), SimAxisMover("stage", stage), SimAndor(lens, plane_um=8000.0, seed=4),
              ShutterManagerDummy("human_interface"), xy_get=lambda: xy.get_position()[:2],
              xy_move=lambda x, y: xy.move_abs(x=x, y=y))
    sz = stage.get_position()
    locs = [Location("r0.00", CX, CY, sz, 8000.0), Location("r1.00", CX - 1000, CY, sz, 8003.0)]
    cfg = ProtocolConfig(n_blocks=1, stationary_s=0.2, slew_half_um=40.0, andor_half_um=6.0, andor_step_um=2.0,
                         exposure_s=0.01, speeds_um_s=(400.0,), sample_rates_hz=(5000.0,))
    total = run_locations(rig, cfg, locs, tmp_path / "run", threading.Event())
    assert total["n_locations"] == 2 and total["n_slew"] == 8 and total["n_andor"] == 8
    info = json.load(open(tmp_path / "run" / "locations.json"))
    assert [l["name"] for l in info["locations"]] == ["r0.00", "r1.00"]
    for i, l in enumerate(locs):
        rows = list(csv.DictReader(open(tmp_path / "run" / f"L{i}_{l.name}" / "run_log.csv")))
        assert len(rows) == len(cfg.steps())
        assert (tmp_path / "run" / f"L{i}_{l.name}_recentre.h5").exists()
    # the simulated plane is at 8000 everywhere, so re-centring pulls the second location back to it
    assert abs(float(rows[0]["lens_um"]) - 8000.0) < 2.5
    x, y = xy.get_position()[:2]
    assert (x, y) == (CX, CY) and abs(lens.get_position() - 8000.0) < 2.0
