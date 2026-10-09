"""
Signal processing for homodyne reflection-plane detection (pure functions).

Signal model on photodiode i (one output of the polarizing beam splitter):

    V_i(t) = DC_i + A_i(z) * cos(phi_i(t)) + noise

DC_i is dominated by the local oscillator (LO). The interference phase
advances by 2*pi for every lambda/2 of axial path change, so during a slew
at speed v the fringes sit at

    f_fringe = 2 * v / lambda          (400 um/s at 780 nm -> ~1.03 kHz)

The fringe amplitude A_i = 2 G R sqrt(P_LO,i * P_s,i) is recovered as the
envelope of the band-passed signal (phase independent). The two
polarization channels are combined as

    S = sqrt( A_H^2 / DC_H + A_V^2 / DC_V )        [units: sqrt(V)]

Dividing each A_i^2 by its own LO level removes the LO split between the
channels, so S^2 is proportional to the total signal power regardless of
the signal's polarization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt


def fringe_frequency_hz(speed_um_s: float, wavelength_nm: float) -> float:
    """Homodyne fringe frequency 2 v / lambda for an axial speed in um/s."""
    return 2.0 * abs(float(speed_um_s)) * 1e-6 / (float(wavelength_nm) * 1e-9)


def fringe_band_hz(
    speed_um_s: float,
    wavelength_nm: float,
    sample_rate_hz: float,
    *,
    rel_halfwidth: float = 0.4,
    min_lo_hz: float = 20.0,
) -> tuple[float, float]:
    """
    Band-pass edges around the expected fringe frequency.

    The band is f_fringe * (1 -/+ rel_halfwidth), so it tolerates stage
    velocity ripple and axial eye motion (Doppler). The upper edge is
    clipped to 0.95 * Nyquist. Raises ValueError if the fringe does not
    fit below Nyquist (lower the speed or raise the sample rate).
    """
    f0 = fringe_frequency_hz(speed_um_s, wavelength_nm)
    nyq = 0.5 * float(sample_rate_hz)
    lo = max(float(min_lo_hz), f0 * (1.0 - rel_halfwidth))
    hi = min(0.95 * nyq, f0 * (1.0 + rel_halfwidth))
    if not lo < hi:
        raise ValueError(
            f"Fringe frequency {f0:.0f} Hz does not fit below Nyquist "
            f"({nyq:.0f} Hz). Lower the speed or raise the sample rate.")
    return lo, hi


def dc_level(values: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 20.0) -> np.ndarray:
    """
    Slowly varying DC level of each channel (zero-phase low-pass).

    values: shape (n,) or (n_ch, n). Returns the same shape. Falls back to
    the per-channel mean when the record is too short to filter.
    """
    v = np.asarray(values, dtype=np.float64)
    nyq = 0.5 * float(sample_rate_hz)
    if not 0.0 < cutoff_hz < nyq:
        raise ValueError(f"DC cutoff {cutoff_hz} Hz must be within (0, {nyq:.0f}) Hz")
    sos = butter(2, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    try:
        return sosfiltfilt(sos, v, axis=-1)
    except ValueError:  # too short for the filter's edge padding
        return np.broadcast_to(np.mean(v, axis=-1, keepdims=True), v.shape).copy()


def fringe_envelope(
    values: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    *,
    order: int = 4,
) -> np.ndarray:
    """
    Fringe amplitude A(t): zero-phase band-pass around the fringe band,
    then the magnitude of the analytic signal (Hilbert envelope).

    values: shape (n,) or (n_ch, n). Returns the same shape, in volts.
    The first and last ~1/f_lo of the record carry filter edge effects.
    """
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(band_hz[0]), float(band_hz[1])
    nyq = 0.5 * float(sample_rate_hz)
    if not 0.0 < lo < hi < nyq:
        raise ValueError(f"Band {lo:.0f}-{hi:.0f} Hz must satisfy 0 < lo < hi < Nyquist ({nyq:.0f} Hz)")

    sos = butter(order, [lo, hi], btype="bandpass", fs=sample_rate_hz, output="sos")
    x = v - np.mean(v, axis=-1, keepdims=True)
    try:
        x = sosfiltfilt(sos, x, axis=-1)
    except ValueError:  # too short for the filter's edge padding
        return np.zeros_like(v)

    n = x.shape[-1]
    analytic = hilbert(x, N=next_fast_len(n), axis=-1)[..., :n]
    return np.abs(analytic)


def combine_polarizations(
    envelope: np.ndarray,
    dc: np.ndarray,
    dark_v: Optional[np.ndarray] = None,
    *,
    dc_floor_v: float = 1e-3,
) -> np.ndarray:
    """
    S = sqrt( sum_i A_i^2 / (DC_i - dark_i) ), shape (n,), units sqrt(V).

    envelope, dc: shape (n_ch, n). dark_v: per-channel offset with the LO
    and signal blocked (PD offset); defaults to 0. The LO level is floored
    at dc_floor_v so a channel without LO cannot blow up the sum.
    """
    env = np.atleast_2d(np.asarray(envelope, dtype=np.float64))
    lo = np.atleast_2d(np.asarray(dc, dtype=np.float64))
    if dark_v is not None:
        lo = lo - np.asarray(dark_v, dtype=np.float64).reshape(-1, 1)
    lo = np.clip(lo, dc_floor_v, None)
    return np.sqrt(np.sum(env ** 2 / lo, axis=0))


@dataclass(frozen=True)
class HomodyneTrace:
    """Processed two-channel record."""
    dc: np.ndarray          # (n_ch, n) LO level per channel [V]
    envelope: np.ndarray    # (n_ch, n) fringe amplitude per channel [V]
    s: np.ndarray           # (n,) combined reflectivity estimate [sqrt(V)]
    band_hz: tuple[float, float]


def process_channels(
    values: np.ndarray,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    *,
    dc_cutoff_hz: float = 20.0,
    dark_v: Optional[np.ndarray] = None,
    order: int = 4,
) -> HomodyneTrace:
    """DC level, fringe envelope and combined S for a (n_ch, n) record."""
    v = np.atleast_2d(np.asarray(values, dtype=np.float64))
    dc = dc_level(v, sample_rate_hz, dc_cutoff_hz)
    env = fringe_envelope(v, sample_rate_hz, band_hz, order=order)
    s = combine_polarizations(env, dc, dark_v)
    return HomodyneTrace(dc=dc, envelope=env, s=s, band_hz=(float(band_hz[0]), float(band_hz[1])))


@dataclass(frozen=True)
class PeakResult:
    found: bool
    x_peak: Optional[float]     # centroid position (same units as x)
    s_peak: float               # max of S
    background: float           # median of S (noise floor)
    noise_std: float            # robust std of S (1.4826 * MAD)
    snr: float                  # (s_peak - background) / noise_std


def locate_peak(
    x: np.ndarray,
    s: np.ndarray,
    *,
    min_snr: float = 8.0,
    centroid_fraction: float = 0.5,
    edge_samples: int = 0,
) -> PeakResult:
    """
    Locate the surface peak in S(x).

    The noise floor is estimated robustly (median / MAD of S), so the peak
    itself barely biases it. The position is the S-weighted centroid of the
    contiguous region around the maximum where S exceeds
    background + centroid_fraction * (peak - background). Returns
    found=False when the peak SNR is below min_snr (no surface) instead of
    a random noise maximum. edge_samples are ignored at both ends (filter
    edge effects).
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if x.shape != s.shape or s.size < 3:
        raise ValueError("x and s must be 1-D arrays of equal length (>= 3)")

    lo_i = int(edge_samples)
    hi_i = s.size - int(edge_samples)
    if hi_i - lo_i < 3:
        lo_i, hi_i = 0, s.size
    xs, ss = x[lo_i:hi_i], s[lo_i:hi_i]

    background = float(np.median(ss))
    noise_std = float(1.4826 * np.median(np.abs(ss - background)))
    k = int(np.argmax(ss))
    s_peak = float(ss[k])
    snr = (s_peak - background) / noise_std if noise_std > 0 else float("inf")
    if not snr >= min_snr:
        return PeakResult(False, None, s_peak, background, noise_std, float(snr))

    level = background + centroid_fraction * (s_peak - background)
    i0 = k
    while i0 > 0 and ss[i0 - 1] > level:
        i0 -= 1
    i1 = k
    while i1 < ss.size - 1 and ss[i1 + 1] > level:
        i1 += 1
    w = ss[i0:i1 + 1] - level
    x_peak = float(np.sum(w * xs[i0:i1 + 1]) / np.sum(w)) if np.sum(w) > 0 else float(xs[k])
    return PeakResult(True, x_peak, s_peak, background, noise_std, float(snr))
