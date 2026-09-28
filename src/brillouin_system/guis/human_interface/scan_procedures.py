"""The acquisition procedures of the human interface.

Each procedure is a FUNCTION over the backend: it drives the backend's
devices (camera, lens, microwave, DAQ) and display callbacks, builds the
resulting AxialScan / CalibrationData, and registers it on the backend.
The backend itself stays device state + primitives (snap a frame, move
the lens, find the reflection plane) — the same split as the analysis
side, where fit_axial_scan drives the fitter instead of living in it.
"""
import itertools
import random
import time

from brillouin_system.calibration.calibration import (
    CalibrationData,
    CalibrationMeasurementPoint,
    MeasurementsPerFreq,
)
from brillouin_system.calibration.config.calibration_config import (
    CalibrationConfig,
    calibration_config,
)
from brillouin_system.guis.human_interface.predefined_plan import (
    cycle_motion_deltas,
    passes_motion_limit,
)
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.my_dataclasses.measurement_point import MeasurementPoint
from brillouin_system.my_dataclasses.request_axial_step_scan import RequestAxialStepScan
from brillouin_system.my_dataclasses.request_sweep_scan import RequestSweepScan
from brillouin_system.my_dataclasses.sweep_cycle import SweepCycle
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult

log = get_logger(__name__)

# Random-order scans reverse direction between frames; every target is
# approached from this far below so the final move is always forward (+z),
# as in scan_managers.reflection_error_characterization.
RANDOM_SCAN_BACKLASH_PRELOAD_UM = 100.0


def random_scan_targets(start_um: float, step_um: float, n: int,
                        seed: int) -> list[float]:
    """The ordinary step scan's targets (start + k*step, k = 1..n), shuffled."""
    targets = [start_um + step_um * (k + 1) for k in range(n)]
    random.Random(seed).shuffle(targets)
    return targets


def take_axial_step_scan(backend, request_axial_scan: RequestAxialStepScan) -> bool:
    lens_x0 = backend.zaber_eye_lens.get_position()
    all_results = []
    reflection_result_forwards: ReflectionResult | None = None
    reflection_result_backwards: ReflectionResult | None = None

    if backend.is_reference_mode:
        log.info(f"[Axial Scan] Measuring N Times the Reference Signal "
                 f"{request_axial_scan.n_measurements}.")

        for i in range(request_axial_scan.n_measurements):
            log.info(f"[Axial Scan] Frame {i + 1}/{request_axial_scan.n_measurements}")
            if backend.f2b_cancel_callback():
                log.info(f"[Axial Scan] Cancelled during step {i + 1}.")
                return False

            frame = backend.get_andor_frame()

            backend.display_spectrum(frame=frame)

            all_results.append(
                MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=lens_x0,
                    time_stamp=time.perf_counter())
            )

    else:

        dx = request_axial_scan.step_size_um

        log.info(f"[Axial Scan] Starting: {request_axial_scan.n_measurements} steps, "
                 f"step size: {request_axial_scan.step_size_um} µm, "
                 f"ID: {request_axial_scan.id}")

        if request_axial_scan.find_reflection_plane:
            reflection_result_forwards = backend.find_reflection_plane(is_go_forwards=True)
            if reflection_result_forwards.found:
                z_pos = (reflection_result_forwards.event_z_um
                         + reflection_result_forwards.z_offset_um)
                backend.zaber_eye_lens.move_abs(z_pos)
            else:
                backend.zaber_eye_lens.move_abs(lens_x0)
                return False

        targets = None
        if request_axial_scan.randomize_order:
            seed = request_axial_scan.random_seed
            if seed is None:
                seed = random.SystemRandom().randrange(2 ** 32)
            targets = random_scan_targets(backend.zaber_eye_lens.get_position(), dx,
                                          request_axial_scan.n_measurements, seed)
            log.info(f"[Axial Scan] Random order, seed {seed}, backlash preload "
                     f"{RANDOM_SCAN_BACKLASH_PRELOAD_UM:.0f} µm")

        for i in range(request_axial_scan.n_measurements):
            if backend.f2b_cancel_callback():
                log.info(f"[Axial Scan] Cancelled during step {i + 1}. "
                         f"Returning lens to starting position.")
                backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
                return False

            log.info(f"[Axial Scan] Frame {i + 1}/{request_axial_scan.n_measurements}")
            if targets is None:
                backend.zaber_eye_lens.move_rel(delta_um=dx)
            else:
                backend.zaber_eye_lens.move_abs(targets[i] - RANDOM_SCAN_BACKLASH_PRELOAD_UM)
                backend.zaber_eye_lens.move_abs(targets[i])
            zaber_pos = backend.zaber_eye_lens.get_position()
            backend.b2f_emit_update_zaber_lens_position(zaber_pos)

            frame = backend.get_andor_frame()

            backend.display_spectrum(frame=frame)

            all_results.append(
                MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=zaber_pos,
                    time_stamp=time.perf_counter())
            )

    if request_axial_scan.find_reflection_plane:
        reflection_result_backwards = backend.find_reflection_plane(is_go_forwards=False)

    # Move lens back to original position
    backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)

    axial_scan = AxialScan(
        i=backend.next_axial_scan_index(),
        id=request_axial_scan.id,
        measurements=all_results,
        system_state=backend.get_current_system_state(),
        calibration_params=backend.calibration_poly_fit_params,
        calibration_data=backend.calibration_data_to_store(),
        eye_tracker_results=request_axial_scan.eye_tracker_results,
        reflection_result_forwards=reflection_result_forwards,
        reflection_result_backwards=reflection_result_backwards,
    )
    backend.register_axial_scan(axial_scan)

    return True


def _accept_crossing(
    result: ReflectionResult | None,
    *,
    reference_z_um: float,
    gate_um: float,
    reference_peak: float | None,
    min_peak_fraction: float,
    reference_name: str,
) -> tuple[bool, str | None]:
    """
    Decide whether a sweep-scan crossing is the real surface.

    Two independent gates, both needed:
      - DISTANCE from a recent reference. Catches a crossing that is
        plausible in shape but in the wrong place.
      - PEAK AMPLITUDE relative to a reference peak. Every false crossing
        observed on 2026-07-30 was a WEAK peak while genuine ones stayed
        above 0.8x their reference: the plastic cuvette's back wall came in
        at 0.12x, and a finder outlier at 0.006x. Amplitude separates these
        by a wide margin and, unlike distance, needs no trade-off against
        how far the eye may really have moved.

    Returns (accepted, reason_if_rejected).
    """
    if result is None or not result.found:
        return False, "no crossing found"

    delta = result.event_z_um - reference_z_um
    if abs(delta) > gate_um:
        return False, (f"{delta:+.1f} µm from {reference_name} "
                       f"(gate {gate_um:.0f} µm)")

    if reference_peak and result.peak_value is not None:
        frac = result.peak_value / reference_peak
        if frac < min_peak_fraction:
            return False, (f"peak {result.peak_value:.3f} V is {frac:.2f}× the "
                           f"{reference_name} peak {reference_peak:.3f} V "
                           f"(min {min_peak_fraction:.2f}×)")

    return True, None


def take_sweep_scan(backend, request: RequestSweepScan) -> bool:
    """
    In-out sweep scan: repeated find-measure-find cycles.

    One cycle: search inward through the corneal reflection, park at the
    in-crossing + target_depth_um, snap a frame, continue inward
    approach_um past the plane, search outward (recording the
    out-crossing), park approach_um outside the freshest plane estimate,
    turn around. The two crossings of a cycle bracket the frame in time;
    depth labels ((in+out)/2 vs single-crossing) are computed in analysis,
    NOT here — both crossings are stored raw per cycle in sweep_cycles.

    Search speed/detection parameters come from the shared axial
    ScanningConfig; cycle geometry from SweepScanConfig. z_offset_um of
    the plain finder is intentionally NOT applied — the sweep scan's
    target_depth_um replaces it.
    """
    if backend.is_reference_mode:
        log.info("[Sweep Scan] System is in Reference Mode - Change to Sample Mode.")
        return False

    sw = backend.sweep_scan_config
    lens_x0 = backend.zaber_eye_lens.get_position()

    # --- Max-time budget (predictive) ---------------------------------------
    # We never interrupt a find mid-motion, so instead of stopping when the
    # limit is already blown we predict, at each checkpoint, whether the next
    # segment can finish in time and stop first if it can't. The estimate is
    # deliberately worst-case:
    #   * a single find can travel the full search distance -> a "find" costs
    #     up to max_distance_um / speed_um_s;
    #   * one measured cycle is a round trip (in-find + out-find) plus one
    #     frame acquisition (settle + camera exposure).
    # 0 (or negative) max_time_s disables the budget.
    #
    # The budget clock (t_start) starts here, at the start of the sweep scan.
    # For a prescribed measurement the sequence is Move XY -> Move Z -> (this)
    # timed sweep scan; those moves position the eye outside the cornea and run
    # as separate requests before this, so they do not count against the sweep
    # time - the clock begins only once the sweep itself starts.
    max_time_s = getattr(sw, "max_time_s", 0.0) or 0.0
    t_start = time.monotonic()
    _speed = abs(getattr(backend.axial_scan_config, "speed_um_s", 0.0)) or 1.0
    t_find_worst = backend.axial_scan_config.max_distance_um / _speed
    try:
        exposure_s = float(backend.sample_state_mode.andor_camera_info.exposure)
    except Exception:
        exposure_s = 0.0
    t_acquire = sw.settle_s + exposure_s
    t_cycle_worst = 2.0 * t_find_worst + t_acquire  # round trip + acquisition

    def _time_left() -> float:
        return max_time_s - (time.monotonic() - t_start)

    timed = bool(getattr(request, "timed", False))
    if timed and max_time_s <= 0:
        log.warning("[Sweep Scan] Timed sweep requires max_time_s > 0 - "
                    "aborting.")
        backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
        return False

    if timed:
        log.info(f"[Sweep Scan] Starting TIMED: budget {max_time_s:.1f} s, "
                 f"target depth {sw.target_depth_um} µm, "
                 f"approach {sw.approach_um} µm, ID: {request.id}")
    else:
        log.info(f"[Sweep Scan] Starting: {sw.n_repeats} cycles, "
                 f"target depth {sw.target_depth_um} µm, "
                 f"approach {sw.approach_um} µm, ID: {request.id}")
    if max_time_s > 0:
        log.info(f"[Sweep Scan] Max time {max_time_s:.1f} s; worst-case per "
                 f"cycle ~{t_cycle_worst:.1f} s "
                 f"(2 x find {t_find_worst:.1f} s + acquire {t_acquire:.2f} s).")

    # Initial full-distance find (normal finder settings) to bootstrap.
    r0: ReflectionResult = backend.find_reflection_plane(is_go_forwards=True)
    if not r0.found:
        log.info("[Sweep Scan] Initial reflection find failed - aborting.")
        backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
        return False
    plane_est = r0.event_z_um
    # Amplitude reference for the in-crossings. The out-crossing of each
    # cycle is instead judged against that cycle's own in-crossing, so a
    # slow legitimate change in signal does not accumulate into a rejection.
    ref_peak = r0.peak_value
    log.info(f"[Sweep Scan] Plane at {plane_est:.1f} µm, reference peak "
             f"{ref_peak:.3f} V (crossings must exceed "
             f"{sw.min_peak_fraction * ref_peak:.3f} V).")

    measurements: list[MeasurementPoint] = []
    cycles: list[SweepCycle] = []
    ended_early = False

    # Timed mode runs unbounded cycles; the max-time budget checks below stop
    # it. A fixed-count sweep runs exactly n_repeats cycles.
    cycle_indices = itertools.count() if timed else range(sw.n_repeats)

    for k in cycle_indices:
        # "End scan early": stop the remaining cycles but keep (save) the
        # cycles already recorded, and count the scan as successful.
        if backend.f2b_end_scan_early_callback():
            log.info(f"[Sweep Scan] End-scan-early requested before cycle "
                     f"{k + 1}. Stopping and saving {len(measurements)} "
                     f"frame(s) collected so far.")
            ended_early = True
            break

        # Predictive budget: don't start a cycle we can't finish in time.
        if max_time_s > 0 and _time_left() < t_cycle_worst:
            log.info(f"[Sweep Scan] Not enough time for another round trip + "
                     f"frame before cycle {k + 1} (need ~{t_cycle_worst:.1f} s, "
                     f"{_time_left():.1f} s left of {max_time_s:.1f} s). Ending "
                     f"early with {len(measurements)} frame(s).")
            ended_early = True
            break

        if backend.f2b_cancel_callback():
            log.info(f"[Sweep Scan] Cancelled during cycle {k + 1}. "
                     f"Returning lens to starting position.")
            backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)
            return False

        log.info(f"[Sweep Scan] Cycle {k + 1}"
                 f"{'' if timed else f'/{sw.n_repeats}'}")

        # Park outside the current plane estimate and search inward.
        backend.zaber_eye_lens.move_abs(plane_est - sw.approach_um)
        r_in: ReflectionResult = backend.find_reflection_plane(is_go_forwards=True)
        in_ok, in_why = _accept_crossing(
            r_in,
            reference_z_um=plane_est,
            gate_um=sw.plausibility_gate_um,
            reference_peak=ref_peak,
            min_peak_fraction=sw.min_peak_fraction,
            reference_name="the last plane estimate",
        )
        if r_in.found and not in_ok:
            log.warning(f"[Sweep Scan] Cycle {k + 1}: in-crossing at "
                        f"{r_in.event_z_um:.1f} µm rejected - {in_why}.")

        measurement_index = None
        r_out: ReflectionResult | None = None

        if in_ok:
            plane_est = r_in.event_z_um

            # Park at the target depth and take the frame (lens stopped).
            backend.zaber_eye_lens.move_abs(plane_est + sw.target_depth_um)
            time.sleep(sw.settle_s)
            zaber_pos = backend.zaber_eye_lens.get_position()
            backend.b2f_emit_update_zaber_lens_position(zaber_pos)

            frame = backend.get_andor_frame()
            backend.display_spectrum(frame=frame)
            measurements.append(
                MeasurementPoint(
                    frame_andor=frame,
                    lens_zaber_position=zaber_pos,
                    time_stamp=time.perf_counter())
            )
            measurement_index = len(measurements) - 1

            # Second checkpoint: the frame for this cycle is already saved, so
            # bail here rather than pay for the long out-search if we're asked
            # to end early, or if the budget can't cover one more find. The
            # finder can't be interrupted mid-motion, so a single find is the
            # smallest granularity here.
            if backend.f2b_end_scan_early_callback():
                log.info(f"[Sweep Scan] End-scan-early requested mid cycle "
                         f"{k + 1} (frame taken). Skipping the out-search and "
                         f"stopping.")
                ended_early = True
            elif max_time_s > 0 and _time_left() < t_find_worst:
                log.info(f"[Sweep Scan] Not enough time for the out-search of "
                         f"cycle {k + 1} (need ~{t_find_worst:.1f} s, "
                         f"{_time_left():.1f} s left). Keeping this frame and "
                         f"stopping.")
                ended_early = True
            else:
                # Continue inward past the plane, then search outward. The
                # out-crossing is judged against THIS cycle's in-crossing —
                # only ~1 s old, so both gates can be tight.
                backend.zaber_eye_lens.move_abs(plane_est + sw.approach_um)
                r_out = backend.find_reflection_plane(is_go_forwards=False)
                out_ok, out_why = _accept_crossing(
                    r_out,
                    reference_z_um=r_in.event_z_um,
                    gate_um=sw.out_gate_um,
                    reference_peak=r_in.peak_value,
                    min_peak_fraction=sw.min_peak_fraction,
                    reference_name="this cycle's in-crossing",
                )
                if r_out.found:
                    delta_um = r_out.event_z_um - r_in.event_z_um
                    log.info(f"[Sweep Scan] Cycle {k + 1}: front (in) "
                             f"{r_in.event_z_um:.1f} µm, back (out) "
                             f"{r_out.event_z_um:.1f} µm, delta "
                             f"{delta_um:+.1f} µm.")
                if r_out.found and not out_ok:
                    log.warning(f"[Sweep Scan] Cycle {k + 1}: out-crossing at "
                                f"{r_out.event_z_um:.1f} µm rejected - {out_why}.")
                if out_ok:
                    # Freshest estimate for aiming the next cycle. The
                    # bias-free (in+out)/2 label is computed in analysis.
                    plane_est = r_out.event_z_um
                else:
                    log.info(f"[Sweep Scan] Cycle {k + 1}: no valid out-crossing - "
                             f"frame keeps its single-crossing (in) reference.")
        else:
            log.info(f"[Sweep Scan] Cycle {k + 1}: no valid in-crossing - "
                     f"skipping the frame this cycle.")

        cycles.append(SweepCycle(
            cycle_index=k,
            reflection_in=r_in,
            reflection_out=r_out,
            measurement_index=measurement_index,
        ))

        # Mid-cycle end-scan-early: record this cycle (done above), then stop.
        if ended_early:
            break

    # Park outside the plane, then return the lens to its start position.
    backend.move_and_update_gui_zaber_eye_lens_abs(lens_x0)

    n_frames = len(measurements)
    n_pairs = sum(1 for c in cycles
                  if c.measurement_index is not None
                  and c.reflection_out is not None and c.reflection_out.found)
    frames_taken = f"{n_frames}" if timed else f"{n_frames}/{sw.n_repeats}"
    log.info(f"[Sweep Scan] Done: {frames_taken} frames taken, "
             f"{n_pairs} with a full in/out pair.")

    # No frames (e.g. every cycle skipped for want of an in-crossing): there is
    # nothing to save. Registering an empty scan produces a 0-measurement file
    # that crashes on open (index out of bounds), so treat this like any other
    # failed/cancelled scan - don't save, report failure.
    if n_frames == 0:
        log.info("[Sweep Scan] No frames acquired - not saving; treating as a "
                 "failed scan.")
        return False

    # Motion quality gate (predefined measurements). The eye must have held
    # still for at least one measured cycle: |out-crossing z - in-crossing z|
    # below the limit. If no cycle qualifies, the scan is dropped exactly like
    # a failed scan - not saved, failure reported, predefined list stays put.
    motion_limit_um = getattr(request, "motion_limit_um", None)
    if motion_limit_um and motion_limit_um > 0:
        deltas = cycle_motion_deltas(cycles)
        if not passes_motion_limit(deltas, motion_limit_um):
            best = min(deltas) if deltas else None
            best_txt = f"{best:.1f} µm" if best is not None else "no in/out pair"
            log.info(f"[Sweep Scan] Motion gate FAILED: no cycle under "
                     f"{motion_limit_um:.1f} µm (best {best_txt} over "
                     f"{len(deltas)} in/out pair(s)). Not saving; treating as a "
                     f"failed scan.")
            return False
        log.info(f"[Sweep Scan] Motion gate passed: best "
                 f"{min(deltas):.1f} µm < {motion_limit_um:.1f} µm.")

    axial_scan = AxialScan(
        i=backend.next_axial_scan_index(),
        id=request.id,
        measurements=measurements,
        system_state=backend.get_current_system_state(),
        calibration_params=backend.calibration_poly_fit_params,
        eye_tracker_results=request.eye_tracker_results,
        reflection_result_forwards=r0,
        reflection_result_backwards=None,
        calibration_data=backend.calibration_data_to_store(),
        sweep_cycles=cycles,
        sweep_config=sw,
        scanning_config=backend.axial_scan_config,
    )
    backend.register_axial_scan(axial_scan)

    if ended_early:
        # Stopped before running every planned cycle - either a timed sweep
        # reaching its budget (its normal end), or an "End Scan Early" / max
        # time stop on a fixed-count sweep. Either way the data is saved above
        # and the scan counts as successful (elapsed time recorded, predefined
        # list advances).
        if timed:
            log.info(f"[Sweep Scan] Timed sweep complete; {n_frames} frame(s) "
                     f"within the {max_time_s:.1f} s budget.")
        else:
            log.info(f"[Sweep Scan] Ended early; saved scan with {n_frames} "
                     f"frame(s).")
        return True

    return n_frames > 0


def perform_calibration(backend) -> bool:
    config: CalibrationConfig = calibration_config.get()

    log.info("[Calibration] Starting calibration.")

    try:
        with backend.force_reference_mode():
            measured_freqs = []

            i = 0
            freqs = config.calibration_freqs
            n = len(freqs)
            for freq in freqs:
                if backend.f2b_cancel_callback():
                    log.info("[Calibration] Cancelled by user.")
                    return False

                backend.microwave.set_frequency(freq)
                # time.sleep(0.2) # testing for settling effects -Zuriel
                i += 1
                log.info(f"Freq {i}/{n}")
                freq_points = []

                for _ in range(config.n_per_freq):
                    if backend.f2b_cancel_callback():
                        log.info("[Calibration] Cancelled by user.")
                        return False

                    frame = backend.get_andor_frame()

                    # The live fit is for the display only; the stored
                    # calibration is raw frames + frequencies — the one
                    # fitting pass happens in calibrate().
                    cali_point = CalibrationMeasurementPoint(
                        frame=frame,
                        microwave_freq=backend.microwave.get_frequency(),
                    )
                    freq_points.append(cali_point)
                    backend.display_spectrum(frame=frame)

                measured_freqs.append(MeasurementsPerFreq(
                    set_freq_ghz=freq,
                    cali_meas_points=freq_points
                ))

            backend.calibration_data = CalibrationData(measured_freqs=measured_freqs)
            backend.update_calibration_calculator()
            log.info(backend.calibration_calculator.get_str_all_models())
            return True

    except Exception as e:
        log.info(f"[Calibration] Exception: {e}")
        return False
