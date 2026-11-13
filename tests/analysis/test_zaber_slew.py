import pytest

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_slew import move_abs_at_speed
from brillouin_system.guis.plane_vivo_test.bounded_lens import BoundedLens
from brillouin_system.guis.z_carrier_test.motion import ZaberAxisMover


class NotSupportedException(Exception):     # same class name zaber_motion raises
    pass


class Settings:
    def __init__(self, maxspeed):
        self.v = {"maxspeed": maxspeed}
        self.sets = []

    def get(self, name, unit=None):
        return self.v[name]

    def set(self, name, value, unit=None):
        self.sets.append(value)
        self.v[name] = value


class Axis:
    """Records moves with the speed they would run at. old=True rejects 'move abs <pos> <speed>'."""

    def __init__(self, old: bool, maxspeed=5000.0, z=1000.0):
        self.old, self.settings, self.z, self.moves, self.stops = old, Settings(maxspeed), z, [], 0

    def move_absolute(self, position, unit=None, wait_until_idle=True, velocity=0, velocity_unit=None):
        if velocity and self.old:
            raise NotSupportedException("Command with template 'move abs ? ?' not supported")
        self.moves.append((position, velocity or self.settings.v["maxspeed"], wait_until_idle))
        self.z = position

    def get_position(self, unit=None):
        return self.z

    def wait_until_idle(self):
        pass

    def stop(self):
        self.stops += 1


def test_new_firmware_puts_the_speed_in_the_command():
    ax = Axis(old=False)
    restore = move_abs_at_speed(ax, 1200.0, 400.0)
    assert ax.moves == [(1200.0, 400.0, False)] and ax.settings.sets == []
    restore()
    assert ax.settings.sets == []


def test_old_firmware_lowers_maxspeed_for_the_move_and_restores_it():
    ax = Axis(old=True, maxspeed=5000.0)
    restore = move_abs_at_speed(ax, 1200.0, 400.0)
    assert ax.moves == [(1200.0, 400.0, False)]
    assert ax.settings.v["maxspeed"] == 400.0
    restore()
    restore()                                   # idempotent
    assert ax.settings.v["maxspeed"] == 5000.0 and ax.settings.sets == [400.0, 5000.0]


def test_other_errors_are_not_swallowed():
    class Broken(Axis):
        def move_absolute(self, *a, **k):
            raise RuntimeError("serial port gone")
    with pytest.raises(RuntimeError):
        move_abs_at_speed(Broken(old=True), 1200.0, 400.0)


@pytest.mark.parametrize("end", ["wait_idle", "stop"])
def test_carrier_mover_restores_speed_when_the_slew_ends(end):
    ax = Axis(old=True, maxspeed=5000.0)
    m = ZaberAxisMover("lens", ax)
    m.set_window(800.0, 1400.0)
    m.slew_to(1200.0, 400.0)
    assert ax.settings.v["maxspeed"] == 400.0
    m.wait_idle() if end == "wait_idle" else m.stop()
    assert ax.settings.v["maxspeed"] == 5000.0
    m.move_abs(1000.0)                          # plain moves run at the restored speed
    assert ax.moves[-1] == (1000.0, 5000.0, True)


def test_bounded_lens_slews_on_old_firmware_and_restores_on_stop():
    class Lens:
        def __init__(self):
            self.axis = Axis(old=True, maxspeed=5000.0)

        def get_position(self):
            return self.axis.z

        def move_abs(self, z):
            self.axis.move_absolute(z)

        def stop_slewing(self):
            self.axis.stop()

    lens = Lens()
    b = BoundedLens(lens)
    b.set_ceiling(2000.0)
    b.start_slewing_guarded(400.0, 300.0)
    assert lens.axis.moves[-1] == (1300.0, 400.0, False) and lens.axis.settings.v["maxspeed"] == 400.0
    b.stop_slewing()
    assert lens.axis.settings.v["maxspeed"] == 5000.0
    b.start_slewing_guarded(-400.0, 300.0)
    b.move_abs(1100.0)                          # a plain move also puts the speed back first
    assert lens.axis.settings.v["maxspeed"] == 5000.0 and lens.axis.moves[-1] == (1100.0, 5000.0, True)
