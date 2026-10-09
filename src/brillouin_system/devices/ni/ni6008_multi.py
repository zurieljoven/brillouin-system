from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# USB-6008: one ADC multiplexed across channels -> the 10 kS/s limit is the
# aggregate over all channels in the task.
MAX_AGGREGATE_RATE_HZ = 10_000.0

# USB-6008 input ranges. RSE is fixed at +-10 V (DAQmx coerces any smaller
# request up to it); differential offers the smaller ranges.
RSE_RANGES_V = (10.0,)
DIFF_RANGES_V = (20.0, 10.0, 5.0, 4.0, 2.5, 2.0, 1.25, 1.0)


@dataclass(frozen=True, slots=True)
class NIMultiReadResult:
    """
    Multi-channel read with a perf_counter anchor.

    Timing model (sample-index based, shared hardware clock for all channels):
        time_of(i) = t0_perf + i / sample_rate_hz
    t0_perf is a best-effort software anchor (the 6008 has no hardware
    timestamps), computed like NI6008's background acquisition.
    """
    values: np.ndarray          # (n_ch, n) volts
    sample_rate_hz: float       # per channel
    t0_perf: float
    channels: tuple[str, ...]

    def time_of(self, i: float) -> float:
        return self.t0_perf + float(i) / float(self.sample_rate_hz)


def effective_range_v(terminal: str, v_range: float) -> float:
    """The range the 6008 actually uses (RSE is always +-10 V)."""
    return 10.0 if terminal.upper() == "RSE" else float(v_range)


class NI6008Multi:
    """
    Hardware-timed multi-channel analog input on an NI USB-6008.

    All channels live in one DAQmx task on one sample clock, so samples of
    different channels are simultaneous up to the multiplexer skew
    (1 / aggregate rate, i.e. 100 us at 10 kS/s), irrelevant for envelopes.

    A task is created per acquire() call, so channel/terminal/range
    settings can change between acquisitions. The legacy single-channel
    NI6008 helper must not be streaming at the same time (the 6008 runs one
    AI task at a time).

    Wiring (16-pin analog terminal block):
        RSE:  ai0 = pin 2, ai1 = pin 5, GND = pins 1, 4, 7, ...
        DIFF: ai0 = pin 2 (+) / pin 3 (-), ai1 = pin 5 (+) / pin 6 (-)
    """

    def __init__(
        self,
        device: str = "Dev1",
        channels: tuple[str, ...] = ("ai0", "ai1"),
        *,
        sample_rate_hz: float = 5000.0,
        terminal: str = "RSE",
        v_range: float = 10.0,
    ):
        self.device = device
        self.channels = tuple(channels)
        self.sample_rate_hz = float(sample_rate_hz)
        self.terminal = terminal.upper()
        self.v_range = float(v_range)

    def configure(self, *, channels=None, sample_rate_hz=None, terminal=None, v_range=None) -> None:
        if channels is not None:
            self.channels = tuple(channels)
        if sample_rate_hz is not None:
            self.sample_rate_hz = float(sample_rate_hz)
        if terminal is not None:
            self.terminal = terminal.upper()
        if v_range is not None:
            self.v_range = float(v_range)

    def validate(self) -> None:
        if not self.channels:
            raise ValueError("At least one channel is required")
        if self.terminal not in ("RSE", "DIFF"):
            raise ValueError(f"terminal must be 'RSE' or 'DIFF', got {self.terminal!r}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        agg = self.sample_rate_hz * len(self.channels)
        if agg > MAX_AGGREGATE_RATE_HZ:
            raise ValueError(
                f"{len(self.channels)} channels x {self.sample_rate_hz:.0f} S/s = {agg:.0f} S/s "
                f"exceeds the USB-6008 aggregate limit of {MAX_AGGREGATE_RATE_HZ:.0f} S/s")

    @property
    def effective_range_v(self) -> float:
        return effective_range_v(self.terminal, self.v_range)

    def acquire(
        self,
        duration_s: float,
        *,
        on_chunk: Optional[Callable[[int], None]] = None,
        stop_evt: Optional[threading.Event] = None,
        chunk_size: int = 500,
        read_timeout_s: float = 1.0,
    ) -> NIMultiReadResult:
        """
        Acquire duration_s seconds from all channels.

        on_chunk(n_acquired) is called (in this thread) after every block, so
        the caller can start motion once the pre-roll is recorded. It must
        not block for long; the DAQ buffer holds several seconds at most.
        stop_evt ends the acquisition early; the samples so far are returned.
        """
        self.validate()

        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        if self.terminal == "RSE":
            term = TerminalConfiguration.RSE
        else:  # renamed DIFFERENTIAL -> DIFF in newer nidaqmx releases
            term = getattr(TerminalConfiguration, "DIFF", None) or TerminalConfiguration.DIFFERENTIAL

        fs = self.sample_rate_hz
        n_ch = len(self.channels)
        n_target = max(1, int(round(float(duration_s) * fs)))
        buf = np.empty((n_ch, n_target), dtype=np.float64)
        rng = self.effective_range_v

        n = 0
        t0_perf: Optional[float] = None
        with nidaqmx.Task() as task:
            for ch in self.channels:
                task.ai_channels.add_ai_voltage_chan(
                    f"{self.device}/{ch}", min_val=-rng, max_val=rng, terminal_config=term)
            task.timing.cfg_samp_clk_timing(
                rate=fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(int(fs * 5), 10 * chunk_size),  # host buffer: ~5 s
            )
            reader = AnalogMultiChannelReader(task.in_stream)
            task.start()
            try:
                while n < n_target:
                    if stop_evt is not None and stop_evt.is_set():
                        break
                    t_before = time.perf_counter()
                    avail = int(task.in_stream.avail_samp_per_chan)
                    if avail <= 0:
                        time.sleep(0.002)
                        continue
                    want = min(avail, int(chunk_size), n_target - n)
                    data = np.empty((n_ch, want), dtype=np.float64)
                    reader.read_many_sample(data, number_of_samples_per_channel=want,
                                            timeout=read_timeout_s)
                    if t0_perf is None:
                        # same anchor as NI6008: the newest available sample
                        # was acquired just before t_before
                        t0_perf = t_before - (avail - 1) / fs
                    buf[:, n:n + want] = data
                    n += want
                    if on_chunk is not None:
                        on_chunk(n)
            finally:
                task.stop()

        if t0_perf is None:
            t0_perf = time.perf_counter()
        return NIMultiReadResult(values=buf[:, :n].copy(), sample_rate_hz=fs,
                                 t0_perf=t0_perf, channels=self.channels)
