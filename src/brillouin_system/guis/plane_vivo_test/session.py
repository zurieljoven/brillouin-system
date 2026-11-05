"""
In vivo plane finder testing, without Qt. Every slew is saved raw as it finishes,
misses and rejected crossings included, with the settings and the eye tracker
reading, so failures can be replayed offline.

A round trip: forward find from plane - approach, optional Andor frame at
plane + depth, backward find from plane + approach. Nothing here gates the data;
the gate only decides whether a crossing may update the plane estimate.
"""

from __future__ import annotations

import csv
import math
import random
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from brillouin_system.guis.plane_accuracy_test.axial_recording import AxialRecording, save_axial
from brillouin_system.guis.plane_accuracy_test.runner import Devices, open_sample_path, recording_from_find
from brillouin_system.scan_managers.homodyne_plane_finder import (
    PRE_ROLL_S, HomodyneFinderParams, find_reflection_homodyne, zero_phase_s,
)
from brillouin_system.scan_managers.homodyne_recording import HomodyneRecording, save_recording

LENS_SETTLE_S = 0.05
SHUTTER_SETTLE_S = 0.1
MAX_XY_MOVE_UM = 3500.0     # as Move XY in the human interface
MAX_AUTO_XY_MOVE_UM = 1500.0    # automatic centering, nobody watching: a bad pupil fit must not move far
EYE_WAIT_S = 3.0            # rides out a blink
EYE_FRAMES = 3              # median over this many frames
EYE_SKIP_AFTER_MOVE = 2     # frames to throw away after a stage move: the eye cameras expose for about 0.2 s
Z_READ_S = 0.4
MAX_AUTO_Z_MOVE_UM = 2000.0     # as Move Z in the human interface; further forward than this is not believed
MAX_NO_PUPIL = 2            # locations in a row: the subject has moved away, stop
NO_PUPIL = "no pupil found"
WRONG_WAY = "centering moved the wrong way, stopping"
NO_PLANE = "no plane found"
CENTER_MOVES = 3
CENTER_TOL_MM = 0.2
STAGE_SETTLE_S = 0.2

LOG_FIELDS = ["n", "time", "t_perf", "label", "run", "cycle", "direction", "status", "z_um", "plane_before_um", "start_um",
              "search_um", "truncated", "s_peak", "threshold", "background_mean", "background_std", "peak_snr",
              "z_at_max_um", "lo_v", "stage_x_um", "stage_y_um", "target_r_mm", "target_phi_deg", "eye_x_mm", "eye_y_mm",
              "eye_dc_mm", "eye_plane_um", "eye_lens_um", "eye_age_s", "test", "location", "z_tracker_um", "seed", "center_note", "file"]
LOCATION_FIELDS = ["test", "order", "seed", "r_mm", "phi_deg", "ended", "center_note", "z_tracker_um", "z_start_um",
                   "fell_back", "z_search_um", "cycles", "found", "rejected", "missed", "sigma_um"]


@dataclass(frozen=True)
class Settings:
    ceiling_um: float
    retract_um: float
    speed_um_s: float = 400.0
    n_sigma: float = 20.0
    stop_after_um: float = 150.0
    search_um: float = 1500.0       # single finds from the current z
    approach_um: float = 300.0
    gate_um: float = 200.0          # a crossing further than this from the estimate does not update it
    n_cycles: int = 10
    max_misses: int = 3             # consecutive missed forward finds end the round trips
    frame_depth_um: float = 0.0     # 0 = no Andor frames
    exposure_s: float = 0.2
    gain: int = 0
    target_r_mm: Optional[float] = None
    target_phi_deg: Optional[float] = None
    settle_s: float = 3.0           # countdown after the click, to settle and fixate
    center_first: bool = False      # XYZ from the eye tracker at every location, after the countdown
    z_margin_um: float = 1000.0     # the search starts this far in front of the tracker's plane
    locations: tuple = ()           # (R mm, phi deg) of a click, in random order; empty: the target above

    def validate(self) -> None:
        if self.retract_um >= self.ceiling_um:
            raise ValueError("The retract position must be below the lens limit.")
        p = HomodyneFinderParams()
        travel_bg = (p.bg_start_s + p.bg_duration_s) * self.speed_um_s
        if self.approach_um < travel_bg + 50.0:
            raise ValueError(f"Approach must be at least {travel_bg + 50.0:.0f} um at this speed: "
                             f"the finder measures its background over the first {travel_bg:.0f} um.")
        if self.gate_um <= 0 or self.n_cycles < 1 or self.max_misses < 1:
            raise ValueError("Gate, round trips and misses must be positive.")
        if len(self.locations) > 1 and not self.center_first:
            raise ValueError("Several locations need the eye tracker centering.")

    def params(self, search_um: float) -> HomodyneFinderParams:
        return HomodyneFinderParams(speed_um_s=self.speed_um_s, n_sigma=self.n_sigma, stop_after_um=self.stop_after_um,
                                    max_distance_um=search_um)


@dataclass
class Slew:
    status: str     # "found" | "rejected" | "missed" | "stopped" (cut short by STOP) | "no data"
    z_um: Optional[float]
    row: dict
    rec: Optional[HomodyneRecording]


def parse_locations(text: str) -> tuple:
    """ "0,0; 1,90; 2,180" -> ((0.0, 0.0), (1.0, 90.0), (2.0, 180.0)) """
    out = []
    for part in text.split(";"):
        if part.strip():
            try:
                r, phi = (float(v) for v in part.split(","))
            except ValueError:
                raise ValueError(f"Locations are written R,phi; R,phi. Cannot read '{part.strip()}'.") from None
            if not 0.0 <= r <= 5.0:
                raise ValueError(f"Location R must be 0 to 5 mm, got {r:g}.")
            out.append((r, phi))
    if not out:
        raise ValueError("No locations given.")
    return tuple(out)


def _target_xy(r_mm: float, phi_deg: float) -> tuple[float, float]:
    return r_mm * math.cos(math.radians(phi_deg)), r_mm * math.sin(math.radians(phi_deg))


def xy_move_for_target(r_mm: float, phi_deg: float, laser_xy_mm: tuple[float, float],
                       max_um: float = MAX_XY_MOVE_UM) -> tuple[float, float, bool]:
    """Stage move [um] that puts the laser at (R, phi) in pupil coordinates; same signs and limit as
    Move XY in the human interface. Returns dx, dy, clamped."""
    dx = (r_mm * math.cos(math.radians(phi_deg)) - laser_xy_mm[0]) * 1000.0
    dy = (r_mm * math.sin(math.radians(phi_deg)) - laser_xy_mm[1]) * 1000.0
    size = math.hypot(dx, dy)
    scale = min(1.0, max_um / size) if size > 0 else 1.0
    return -dx * scale, dy * scale, scale < 1.0


def peak_diagnostics(rec: HomodyneRecording, params: HomodyneFinderParams) -> tuple[Optional[float], Optional[float]]:
    """Peak SNR after the background window and its z, also for a miss: how close was it?"""
    z = rec.z_um()
    fs = rec.sample_rate_hz
    i0 = int(round((PRE_ROLL_S + params.bg_start_s + params.bg_duration_s) * fs))
    mean, std = rec.meta.get("finder_background_mean"), rec.meta.get("finder_background_std")
    if z is None or mean is None or not std or rec.n_samples <= i0 + 2:
        return None, None
    s = zero_phase_s(rec.values, fs, params)
    i = i0 + int(np.argmax(s[i0:]))
    return float((s[i] - mean) / std), float(z[i])


class Session:
    def __init__(self, dev: Devices, run_dir: Path, subject: str = "",
                 eye_state: Callable[[], dict] = dict,
                 on_status: Callable[[str], None] = lambda s: None,
                 on_slew: Callable[[Slew], None] = lambda s: None,
                 on_frame: Callable[[np.ndarray, float], None] = lambda f, z: None,
                 cue: Callable[[str], None] = lambda name: None, say: Callable[[str], None] = lambda text: None):
        self.dev, self.run_dir, self.subject = dev, Path(run_dir), subject
        self.cue, self.say = cue, say
        self.eye_state, self.on_status, self.on_slew, self.on_frame = eye_state, on_status, on_slew, on_frame
        self.n = 0
        self.run = 0
        self.test_id = 0
        self.last_target = (0.0, 0.0)   # where the stage was last centered; until then the head is expected near R = 0
        self.context: dict = {}     # test, location, tracker depth: saved with every slew
        self._said_go = False
        self.plane_um: Optional[float] = None

    def _next_name(self, kind: str) -> str:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.n += 1
        return f"{self.n:04d}_{kind}.h5"

    def _log(self, row: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "slews.csv"
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def _base_meta(self, st: Settings) -> dict:
        sx, sy, _ = self.dev.stage.get_position()
        eye = self.eye_state() or {}
        return dict(experiment=self.run_dir.name, subject=self.subject, settings=asdict(st), stage_x_um=sx, stage_y_um=sy,
                    target_r_mm=st.target_r_mm, target_phi_deg=st.target_phi_deg, **eye, **self.context)

    def find(self, st: Settings, direction: float, start_um: Optional[float], search_um: float, stop: threading.Event,
             *, label: str = "single", cycle: Optional[int] = None, gated: bool = True) -> Slew:
        """start_um None: from the current z. Leaves the lens wherever the find ended. gated False: any
        crossing updates the estimate."""
        lens = self.dev.lens
        if start_um is not None:
            lens.move_abs(start_um)
            time.sleep(LENS_SETTLE_S)
        z_start = float(lens.get_position())
        room = lens.room_um(direction, search_um)
        if room < st.stop_after_um:
            raise ValueError(f"Only {room:.0f} um of room before the lens limit, from z = {z_start:.1f} um.")
        p = st.params(room)
        plane_before = self.plane_um
        meta = dict(self._base_meta(st), label=label, run=self.run, cycle=cycle, plane_before_um=plane_before,
                    kind="find_fwd" if direction > 0 else "find_bwd", truncated=room < search_um)
        r = find_reflection_homodyne(self.dev.daq, lens, p, direction=direction, stop_evt=stop)
        rec = None
        if r.daq_values is not None:
            rec = recording_from_find(r, self.dev.daq, p=p, z0=z_start, z_start=z_start, direction=direction, meta=meta)
            rec.meta["stopped_early"] = stop.is_set()

        status = "no data" if rec is None else "stopped" if stop.is_set() else "missed"
        z = None
        if r.found:
            z = float(r.event_z_um)
            accepted = not gated or plane_before is None or abs(z - plane_before) <= st.gate_um
            status = "found" if accepted else "rejected"
            if accepted:
                self.plane_um = z
        name = None
        snr, z_max, lo = None, None, None
        if rec is not None:
            rec.meta["status"] = status
            name = self._next_name(rec.meta["kind"])
            save_recording(rec, self.run_dir / name)
            snr, z_max = peak_diagnostics(rec, p)
            lo = float(np.mean(rec.values))
        row = dict(rec.meta if rec is not None else meta, n=self.n, t_perf=time.perf_counter(),
                   time=datetime.now().isoformat(timespec="milliseconds"), direction=direction, status=status, z_um=z,
                   start_um=z_start, search_um=room, s_peak=r.peak_value, threshold=r.threshold_high,
                   background_mean=r.background_mean, background_std=r.background_std, peak_snr=snr, z_at_max_um=z_max,
                   lo_v=lo, file=name)
        self._log(row)
        if status in ("missed", "rejected", "no data"):
            self.cue("miss")
        slew = Slew(status, z, row, rec)
        self.on_slew(slew)
        return slew

    def _frame(self, st: Settings, z_um: float, cycle: int) -> None:
        lens, cam = self.dev.lens, self.dev.camera
        lens.move_abs(z_um)
        time.sleep(LENS_SETTLE_S)
        z = float(lens.get_position())
        t = time.perf_counter()
        frame = np.asarray(cam.snap())
        meta = dict(self._base_meta(st), kind="frame", cycle=cycle, plane_um=self.plane_um,
                    depth_um=st.frame_depth_um, exposure_s=float(cam.get_exposure_time()),
                    gain=float(cam.get_emccd_gain()))
        save_axial(AxialRecording(frames=frame[None], z_um=np.array([z]), t_perf=np.array([t]), meta=meta),
                   self.run_dir / self._next_name("frame"))
        self.on_frame(frame, z)

    def round_trips(self, st: Settings, stop: threading.Event) -> dict:
        """Needs a plane estimate. Always retracts and closes the shutters at the end, also on an error."""
        st.validate()
        if self.plane_um is None:
            raise ValueError("No plane estimate yet: run a single find first.")
        self.run += 1
        counts = dict(cycles=0, found=0, rejected=0, missed=0, ended="done")
        if st.frame_depth_um > 0:
            self.dev.camera.set_exposure_time(st.exposure_s)
            self.dev.camera.set_emccd_gain(st.gain)
        self.open_shutters()
        misses = 0
        deltas = []
        try:
            for k in range(st.n_cycles):
                if stop.is_set():
                    counts["ended"] = "stopped"
                    break
                self.on_status(f"Round trip {k + 1}/{st.n_cycles}, plane estimate {self.plane_um:.1f} um")
                s_in = self.find(st, 1.0, self.plane_um - st.approach_um, 2.0 * st.approach_um, stop,
                                 label="round_trip", cycle=k)
                if s_in.status in counts:
                    counts[s_in.status] += 1
                counts["cycles"] += 1
                misses = misses + 1 if s_in.status in ("missed", "no data", "rejected") else 0
                if misses >= st.max_misses:
                    counts["ended"] = f"{misses} forward finds in a row without a plane"
                    break
                if stop.is_set():
                    counts["ended"] = "stopped"
                    break
                if s_in.status == "found" and st.frame_depth_um > 0:
                    self._frame(st, self.plane_um + st.frame_depth_um, k)
                s_out = self.find(st, -1.0, self.plane_um + st.approach_um, 2.0 * st.approach_um, stop,
                                  label="round_trip", cycle=k)
                if s_out.status in counts:
                    counts[s_out.status] += 1
                if s_in.status == "found" and s_out.status == "found":
                    deltas.append(s_in.z_um - s_out.z_um)
        finally:
            self.retract(st)
        counts["sigma_um"] = float(np.std(deltas, ddof=1) / math.sqrt(2.0)) if len(deltas) > 1 else None
        return counts

    def _read_eye(self, stop: threading.Event, key: str, skip: int, n: int) -> list:
        """Values of a key from the next new eye frames, after skipping some. The cameras expose for about 0.2 s and
        frames are delivered late, so the frames right after a stage move still show the old position. Rides out a
        blink. Frames are counted by eye_n; without new frames this returns nothing."""
        values = []
        last = (self.eye_state() or {}).get("eye_n")
        seen = 0
        t_end = time.monotonic() + EYE_WAIT_S
        while time.monotonic() < t_end and not stop.is_set() and len(values) < n:
            eye = self.eye_state() or {}
            k = eye.get("eye_n")
            if k is not None and k != last:
                last = k
                seen += 1
                if seen > skip and eye.get(key) is not None:
                    values.append(eye if key == "eye_x_mm" else float(eye[key]))
            time.sleep(0.01)
        return values

    def _read_xy(self, stop: threading.Event, skip: int) -> Optional[tuple[float, float]]:
        frames = self._read_eye(stop, "eye_x_mm", skip, EYE_FRAMES)
        if not frames:
            return None
        return float(np.median([f["eye_x_mm"] for f in frames])), float(np.median([f["eye_y_mm"] for f in frames]))

    def _center(self, st: Settings, stop: threading.Event, allowance_mm: float = 0.0) -> str:
        """Closed loop, as nobody is watching: re-reads the pupil after every move. Only a move that makes it clearly
        worse is an error (WRONG_WAY); anything else is a note, and the plane test goes ahead wherever the laser is.
        allowance_mm: how far the target itself moved from the last location."""
        r, phi = st.target_r_mm or 0.0, st.target_phi_deg or 0.0
        tx, ty = _target_xy(r, phi)
        limit_mm = MAX_AUTO_XY_MOVE_UM / 1000.0 + allowance_mm
        previous = None
        skip = 0
        for attempt in range(CENTER_MOVES + 1):
            xy = self._read_xy(stop, skip)
            if stop.is_set():
                return ""
            if xy is None:
                return NO_PUPIL
            err_mm = math.hypot(xy[0] - tx, xy[1] - ty)
            if err_mm <= CENTER_TOL_MM:
                return ""
            if previous is not None and err_mm > 1.5 * previous + 0.1:
                return WRONG_WAY
            if err_mm > limit_mm:
                return f"centering needs {err_mm:.1f} millimeters, too far"
            if attempt == CENTER_MOVES or (previous is not None and err_mm > 0.8 * previous):
                return f"off target by {err_mm:.2f} millimeters"
            dx, dy, _ = xy_move_for_target(r, phi, xy, limit_mm * 1000.0)
            self.dev.stage.move_rel(dx=dx, dy=dy)
            time.sleep(STAGE_SETTLE_S)
            previous, skip = err_mm, EYE_SKIP_AFTER_MOVE
        return ""

    def _tracker_plane(self, stop: threading.Event) -> Optional[float]:
        """Median over a few frames of where the eye tracker puts the cornea, in lens z."""
        values = self._read_eye(stop, "eye_plane_um", 1, EYE_FRAMES)
        return float(np.median(values)) if values else None

    def _search_start(self, st: Settings, z_tracker: Optional[float]) -> float:
        """Margin in front of the tracker's plane, or the retract position if that is not a sensible place."""
        if z_tracker is None:
            return st.retract_um
        start = z_tracker - st.z_margin_um
        if start > st.retract_um + MAX_AUTO_Z_MOVE_UM or start + st.stop_after_um >= st.ceiling_um:
            return st.retract_um
        return start

    def _countdown(self, st: Settings, stop: threading.Event) -> None:
        n = int(round(st.settle_s))
        for _ in range(n):
            self.cue("count")
            if stop.wait(1.0):
                return
        if st.settle_s > n:
            stop.wait(st.settle_s - n)

    def auto_test(self, st: Settings, stop: threading.Event) -> dict:
        """One click, for a subject who cannot see the screen: lens back, countdown to settle and fixate, then at
        every location in random order: XYZ centering, shutters open, search for the plane, round trips, lens back,
        shutters closed. Says the result out loud."""
        c = self._auto_test(st, stop)
        ended = c["ended"]
        total = c["found"] + c["rejected"] + c["missed"]
        all_found = c["found"] > 0 and c["rejected"] == 0 and c["missed"] == 0 and not c["n_failed"] \
            and not c["n_off_target"]
        self.cue("stopped" if ended == "stopped" else "success" if ended == "done" and all_found
                 else "partial" if c["found"] > 0 else "failed")
        if ended == "stopped":
            self.say("Stopped")
        elif total == 0 and c["n_locations"] <= 1:
            self.say(ended)
        elif total == 0:
            self.say(f"No planes. {ended}")
        else:
            sigma = f", short term sigma {c['sigma_um']:.0f} microns" if c.get("sigma_um") is not None else ""
            failed = f", {c['n_failed']} of {c['n_locations']} locations failed" if c["n_failed"] else ""
            failed += f", {c['n_off_target']} off target" if c["n_off_target"] else ""
            early = "" if ended == "done" else ", ended early"
            self.say(f"{c['found']} of {total} found{sigma}{failed}{early}")
        return c

    def _test_location(self, st: Settings, stop: threading.Event, allowance_mm: float) -> dict:
        """Returns the counts of this location; ended says why it stopped, "done" if it ran. Eye tracker trouble does
        not stop the plane test: it searches from wherever the laser is, and center_note says so."""
        none = dict(cycles=0, found=0, rejected=0, missed=0, sigma_um=None)
        self.plane_um = None
        self.dev.lens.move_abs(st.retract_um)
        note = ""
        z_tracker = None
        if st.center_first:
            note = self._center(st, stop, allowance_mm)
            if note == WRONG_WAY:
                return dict(none, ended=WRONG_WAY, center_note=note)
            if stop.is_set():
                return dict(none, ended="stopped", center_note=note)
            if note != NO_PUPIL:
                z_tracker = self._tracker_plane(stop)
        start = self._search_start(st, z_tracker)
        self.context.update(z_tracker_um=z_tracker, center_note=note)
        if stop.is_set():
            return dict(none, ended="stopped", center_note=note)
        if not self._said_go:
            self._said_go = True
            self.cue("go")
        self.open_shutters()
        try:
            s = self.find(st, 1.0, start, st.search_um, stop, label="search", gated=False)
            fell_back = False
            if s.status == "missed" and start > st.retract_um + 50.0:   # the tracker may have put it past the cornea
                fell_back = True
                s = self.find(st, 1.0, st.retract_um, st.search_um, stop, label="search_fallback", gated=False)
        except Exception:
            self.retract(st)
            raise
        info = dict(center_note=note, z_tracker_um=z_tracker, z_start_um=start, fell_back=fell_back,
                    z_search_um=s.z_um)
        if s.status != "found":
            self.retract(st)
            ended = "stopped" if s.status == "stopped" else NO_PUPIL if note == NO_PUPIL else NO_PLANE
            return dict(none, ended=ended, **info)
        return dict(self.round_trips(st, stop), **info)

    def _log_location(self, row: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "locations.csv"
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOCATION_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def _auto_test(self, st: Settings, stop: threading.Event) -> dict:
        st.validate()
        locations = list(st.locations) or [(st.target_r_mm or 0.0, st.target_phi_deg or 0.0)]
        seed = random.randrange(1 << 31)
        random.Random(seed).shuffle(locations)
        self.test_id += 1
        self.dev.lens.move_abs(st.retract_um)
        self._said_go = False
        self._countdown(st, stop)
        total = dict(cycles=0, found=0, rejected=0, missed=0, n_locations=len(locations), n_failed=0, n_ran=0, n_off_target=0,
                     ended="done")
        variances = []
        no_pupil = 0
        for i, (r, phi) in enumerate(locations):
            if stop.is_set():
                total["ended"] = "stopped"
                break
            one = replace(st, target_r_mm=r, target_phi_deg=phi)
            self.on_status(f"Location {i + 1}/{len(locations)}: R {r:g} mm, phi {phi:g} deg")
            self.context = dict(test=self.test_id, location=i, seed=seed)
            c = self._test_location(one, stop, math.dist(_target_xy(r, phi), _target_xy(*self.last_target)))
            for k in ("cycles", "found", "rejected", "missed"):
                total[k] += c[k]
            if c.get("sigma_um") is not None:
                variances.append(c["sigma_um"] ** 2)
            if c.get("center_note") == "":
                self.last_target = (r, phi)
            if c.get("center_note"):
                total["n_off_target"] += 1
            self._log_location(dict(c, test=self.test_id, order=i, seed=seed, r_mm=r, phi_deg=phi))
            self.on_status(f"R {r:g} mm, phi {phi:g} deg: {c['ended']}, {c['found']} found, {c['rejected']} rejected, "
                           f"{c['missed']} missed")
            if c["ended"] in ("stopped", WRONG_WAY):
                total["ended"] = c["ended"]
                break
            if c["ended"] == "done":
                total["n_ran"] += 1
            else:
                total["n_failed"] += 1
                total["last_problem"] = c["ended"]
            no_pupil = no_pupil + 1 if c["ended"] == NO_PUPIL else 0
            if no_pupil >= MAX_NO_PUPIL:
                total["ended"] = NO_PUPIL
                break
        self.retract(st)
        if total["ended"] == "done" and total["n_failed"] == len(locations):
            total["ended"] = total["last_problem"]
        total["sigma_um"] = math.sqrt(sum(variances) / len(variances)) if variances else None
        return total

    def single_find(self, st: Settings, direction: float, stop: threading.Event) -> Slew:
        """From the current z, then back to where it started."""
        z0 = float(self.dev.lens.get_position())
        self.open_shutters()
        try:
            return self.find(st, direction, None, st.search_um, stop, gated=False)
        finally:
            try:
                self.dev.lens.move_abs(z0)
            finally:
                self.close_shutters()

    def retract(self, st: Settings) -> None:
        try:
            self.dev.lens.move_abs(st.retract_um)
        finally:
            self.close_shutters()

    def open_shutters(self) -> None:
        if self.dev.shutters is not None:
            open_sample_path(self.dev.shutters)
            time.sleep(SHUTTER_SETTLE_S)

    def close_shutters(self) -> None:
        if self.dev.shutters is not None:
            self.dev.shutters.close_all()
