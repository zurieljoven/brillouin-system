"""
Real-time reflection-plane finder on the two-channel homodyne signal.

Replaces the single-channel threshold finder for the human interface GUI. The
fringes at the plane are not a clean line at 2v/lambda (vibration dominates,
see the 2026-10-09 FFTs), so the envelope uses band-limited power, which needs
no knowledge of the fringe frequency:

    x_i  = band-pass(V_i)                      causal IIR during the slew
    A_i² = 2 * mean(x_i²) over a short window  trailing during the slew
    S    = sqrt( sum_i A_i² / DC_i )           S² ∝ reflected power, any polarization

Stop rule (PlaneDetector): the background is taken just after the slew starts;
the search arms once S exceeds mean + n_sigma * std, then stops when S has made
no new maximum for stop_after_um of travel.

Plane (refine_plane): zero-phase S of the recorded buffer, S²-weighted centroid
within +-centroid_window_um of the maximum. The lens overshoots the plane by
stop_after_um; callers move it back, as with the old finder.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_position_log import interp_z_positions
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult, fit_z_at_time

log = logging.getLogger(__name__)

PRE_ROLL_S = 0.05       # DAQ runs this long before the slew starts
MIN_BACKGROUND_STD = 2e-4   # floor for a dead-quiet (quantized) background; measured std is 0.5-1.3 mV


@dataclass(frozen=True)
class HomodyneFinderParams:
    speed_um_s: float = 400.0
    max_distance_um: float = 1500.0
    band_hz: tuple[float, float] = (100.0, 2400.0)
    rms_window_s: float = 0.005
    dc_cutoff_hz: float = 20.0
    order: int = 4
    bg_start_s: float = 0.1             # after the slew starts (acceleration, filter settling)
    bg_duration_s: float = 0.1
    n_sigma: float = 20.0
    stop_after_um: float = 150.0
    centroid_window_um: float = 30.0
    min_lo_v: float = 0.05              # below this on every channel the LO is off / blocked


class StreamingS:
    """Causal S, chunk by chunk. Filter state is carried across push() calls."""

    def __init__(self, n_ch: int, fs: float, p: HomodyneFinderParams):
        self._bp = butter(p.order, p.band_hz, btype="bandpass", fs=fs, output="sos")
        self._lp = butter(2, p.dc_cutoff_hz, btype="lowpass", fs=fs, output="sos")
        self._n_ch = int(n_ch)
        self._win = max(1, int(round(p.rms_window_s * fs)))
        self._zi_bp: Optional[np.ndarray] = None
        self._zi_lp: Optional[np.ndarray] = None
        self._tail = np.zeros((self._n_ch, self._win))  # last x² samples, for the trailing mean
        self.dc = np.zeros(self._n_ch)                  # latest LO level per channel

    def push(self, values: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if self._zi_bp is None:     # start from steady state at the first sample: no step transient
            v0 = v[:, 0][None, :, None]
            self._zi_bp = sosfilt_zi(self._bp)[:, None, :] * v0
            self._zi_lp = sosfilt_zi(self._lp)[:, None, :] * v0
        x, self._zi_bp = sosfilt(self._bp, v, axis=-1, zi=self._zi_bp)
        dc, self._zi_lp = sosfilt(self._lp, v, axis=-1, zi=self._zi_lp)
        self.dc = dc[:, -1].copy()

        x2 = np.concatenate((self._tail, x * x), axis=-1)
        c = np.cumsum(x2, axis=-1)
        mean_x2 = (c[:, self._win:] - c[:, :-self._win]) / self._win
        self._tail = x2[:, -self._win:]
        return np.sqrt(np.sum(2.0 * np.clip(mean_x2, 0.0, None) / np.clip(dc, 1e-3, None), axis=0))


class PlaneDetector:
    """Arm above the background, stop after stop_after_um with no new maximum."""

    def __init__(self, fs: float, p: HomodyneFinderParams, *, slew_start_index: int = 0):
        self._i0 = slew_start_index + int(round(p.bg_start_s * fs))
        self._i1 = self._i0 + max(2, int(round(p.bg_duration_s * fs)))
        self._stop_n = max(1, int(round(p.stop_after_um / p.speed_um_s * fs)))
        self._n_sigma = float(p.n_sigma)
        self._bg: list[np.ndarray] = []
        self.n = 0                              # samples seen
        self.background_mean: Optional[float] = None
        self.background_std: Optional[float] = None
        self.threshold: Optional[float] = None
        self.i_arm: Optional[int] = None
        self.i_max: Optional[int] = None
        self.s_max = -np.inf
        self.stopped = False

    def push(self, s: np.ndarray) -> bool:
        """Feed causal S; True once the search should stop."""
        s = np.asarray(s, dtype=np.float64)
        i_first = self.n
        self.n += s.size
        if self.stopped or s.size == 0:
            return self.stopped

        if self.threshold is None:
            a, b = max(self._i0 - i_first, 0), max(self._i1 - i_first, 0)
            if b > a:
                self._bg.append(s[a:b])
            if self.n < self._i1:
                return False
            bg = np.concatenate(self._bg)
            self.background_mean = float(np.mean(bg))
            self.background_std = max(float(np.std(bg)), MIN_BACKGROUND_STD)
            self.threshold = self.background_mean + self._n_sigma * self.background_std
            s, i_first = s[b:], i_first + b     # the search starts after the background window
            if s.size == 0:
                return False

        idx = i_first + np.arange(s.size)
        if self.i_arm is None:
            above = np.flatnonzero(s > self.threshold)
            if above.size == 0:
                return False
            self.i_arm = int(idx[above[0]])
            s, idx = s[above[0]:], idx[above[0]:]

        running = np.maximum.accumulate(np.concatenate(([self.s_max], s)))
        is_new = s > running[:-1]
        last_new = np.maximum.accumulate(np.where(is_new, idx, -1))
        if self.i_max is not None:
            last_new = np.maximum(last_new, self.i_max)
        hit = np.flatnonzero(idx - last_new >= self._stop_n)
        end = hit[0] if hit.size else s.size - 1
        self.s_max, self.i_max = float(running[end + 1]), int(last_new[end])
        self.stopped = bool(hit.size)
        return self.stopped


@dataclass(frozen=True)
class RefinedPlane:
    index: float        # fractional sample index of the S² centroid
    s_peak: float       # max of the zero-phase S
    i_peak: int
    s: np.ndarray       # zero-phase S of the buffer


def zero_phase_s(values: np.ndarray, fs: float, p: HomodyneFinderParams) -> np.ndarray:
    """Zero-phase S of a recorded buffer (same band and window as StreamingS)."""
    v = np.atleast_2d(np.asarray(values, dtype=np.float64))
    bp = butter(p.order, p.band_hz, btype="bandpass", fs=fs, output="sos")
    lp = butter(2, p.dc_cutoff_hz, btype="lowpass", fs=fs, output="sos")
    x = sosfiltfilt(bp, v - v.mean(axis=-1, keepdims=True), axis=-1)
    mean_x2 = uniform_filter1d(x * x, max(1, int(round(p.rms_window_s * fs))), axis=-1, mode="nearest")
    dc = sosfiltfilt(lp, v, axis=-1)
    return np.sqrt(np.sum(2.0 * np.clip(mean_x2, 0.0, None) / np.clip(dc, 1e-3, None), axis=0))


def refine_plane(values: np.ndarray, fs: float, p: HomodyneFinderParams, *, i_min: int = 0,
                 i_max: Optional[int] = None) -> RefinedPlane:
    """S²-weighted centroid within +-centroid_window_um of the zero-phase maximum in [i_min, i_max)."""
    s = zero_phase_s(values, fs, p)
    edge = int(2.0 * fs / p.band_hz[0])
    a = max(int(i_min), edge)
    b = min(s.size if i_max is None else int(i_max), s.size - edge)
    if b <= a:
        a, b = 0, s.size
    k = a + int(np.argmax(s[a:b]))
    half = max(1, int(round(p.centroid_window_um / p.speed_um_s * fs)))
    i = np.arange(max(0, k - half), min(s.size, k + half + 1))
    q = s[i] ** 2
    return RefinedPlane(index=float(np.sum(q * i) / np.sum(q)), s_peak=float(s[k]), i_peak=k, s=s)


def find_reflection_homodyne(
    daq,
    zaber,
    params: HomodyneFinderParams,
    *,
    direction: float = 1.0,
    z_offset_um: float = 0.0,
    z_poll_s: float = 0.016,
    chunk_size: int = 100,
    stop_evt: Optional[threading.Event] = None,
) -> ReflectionResult:
    """
    Slew from the current position and stop just past the plane. daq is an
    NI6008Multi (or its simulator). The lens is NOT moved back: as with the old
    finder the caller moves it to event_z_um + z_offset_um, or home if not found.
    """
    p = params
    fs = float(daq.sample_rate_hz)
    n_pre = int(round(PRE_ROLL_S * fs))
    max_time_s = PRE_ROLL_S + abs(p.max_distance_um / p.speed_um_s) + 0.3
    stream = StreamingS(len(daq.channels), fs, p)
    detector = PlaneDetector(fs, p, slew_start_index=n_pre)
    stop = threading.Event()
    state = {"log": False, "slew": False, "n": 0}

    def on_data(block: np.ndarray) -> None:
        if not state["log"]:
            zaber.start_position_log(poll_s=z_poll_s, alpha=0.25)
            state["log"] = True
        state["n"] += block.shape[1]
        if not state["slew"] and state["n"] >= n_pre:
            zaber.start_slewing_guarded(direction * p.speed_um_s, p.max_distance_um)
            state["slew"] = True
        if detector.push(stream.push(block)) or (stop_evt is not None and stop_evt.is_set()):
            zaber.stop_slewing()
            stop.set()

    zlog = None
    res = None
    try:
        res = daq.acquire(max_time_s, on_data=on_data, stop_evt=stop, chunk_size=chunk_size)
    finally:
        try:
            zaber.stop_slewing()
        except Exception:
            pass
        if state["log"]:
            try:
                zlog = zaber.stop_position_log()
            except Exception:
                zlog = None

    # raw data is kept on a miss too, so a failed search can be analysed offline
    raw = dict(daq_ts=None if res is None else res.t0_perf + np.arange(res.values.shape[1]) / fs,
               daq_values=None if res is None else res.values,
               zaber_lens_ts=None if zlog is None else np.asarray(zlog.t_perf),
               zaber_lens_z_um=None if zlog is None else np.asarray(zlog.z_um))
    stats = dict(background_mean=detector.background_mean, background_std=detector.background_std,
                 threshold_high=detector.threshold)
    if np.all(stream.dc < p.min_lo_v):
        log.warning(f"[Homodyne finder] LO level {np.round(stream.dc, 3)} V on all channels: LO blocked or off?")
        return ReflectionResult(found=False, **stats, **raw)
    if detector.i_arm is None or res is None or zlog is None or zlog.t_perf.size < 2:
        return ReflectionResult(found=False, **stats, **raw)

    plane = refine_plane(res.values, fs, p, i_min=detector.i_arm - int(round(p.centroid_window_um / p.speed_um_s * fs)))
    t_event = res.time_of(plane.index)
    zt, zz = np.asarray(zlog.t_perf), np.asarray(zlog.z_um)
    z_interp = interp_z_positions(t_event, zt, zz)
    z_fit = fit_z_at_time(t_event, zt, zz)
    z_event = z_fit if np.isfinite(z_fit) else z_interp
    return ReflectionResult(
        found=True,
        event_index=plane.index,
        event_time_perf=t_event,
        event_z_um=float(z_event),
        event_z_um_interp=float(z_interp) if np.isfinite(z_interp) else None,
        event_z_um_fit=float(z_fit) if np.isfinite(z_fit) else None,
        z_offset_um=z_offset_um,
        peak_value=plane.s_peak,
        **stats,
        idx_first=detector.i_arm,
        idx_last=detector.i_max,
        n_samples_above=int(np.sum(plane.s > detector.threshold)),
        **raw,
    )
