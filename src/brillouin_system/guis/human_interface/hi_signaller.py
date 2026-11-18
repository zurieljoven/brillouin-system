from enum import Enum
import threading
import time

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication
from PyQt5 import QtCore

from brillouin_system.calibration.config.calibration_config import CalibrationConfig
from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import LaserOffset
from brillouin_system.guis.human_interface.hi_backend import HiBackend
from brillouin_system.guis.human_interface import scan_procedures
from brillouin_system.guis.human_interface.predefined_plan import build_progress
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.my_dataclasses.request_axial_step_scan import RequestAxialStepScan
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FittingConfigs

log = get_logger(__name__)


class SystemState(Enum):
    IDLE = 0
    FREERUNNING = 1
    BUSY = 2



class HiSignaller(QObject):


    # Start / Stop Signals
    start_live_signal = pyqtSignal()
    stop_live_signal = pyqtSignal()

    # Bool Signals outwards
    illumination_mode_state  = pyqtSignal(bool)
    reference_mode_state = pyqtSignal(bool)
    do_live_fitting_state = pyqtSignal(bool)

    # Signals outwards
    camera_settings_ready = pyqtSignal(dict)
    zaber_lens_position_updated = pyqtSignal(float)
    microwave_frequency_updated = pyqtSignal(float)
    # frame_and_fit_ready = pyqtSignal(object)
    measurement_result_ready = pyqtSignal(object)
    camera_shutter_state_changed = pyqtSignal(bool)
    calibration_finished = pyqtSignal()
    calibration_failed = pyqtSignal()       # perform_calibration returned False (cancelled or an error)
    calibration_result_ready = pyqtSignal(object)
    send_update_stored_axial_scans = pyqtSignal(list)
    # Emitted at the START of a sweep scan (after any Move XY / Move Z), so the
    # frontend can arm a hard max-time backstop from the right moment. Carries
    # the max_time_s budget the backend will use (0 = no limit).
    sweep_scan_started = pyqtSignal(float)
    # (success, elapsed_s). success is True only when a sweep scan actually
    # completed and produced data; False if it failed to start / was cancelled
    # (initial find failed, no calibration, cancelled mid-scan, no frames).
    # elapsed_s is wall-clock time measured from the START of the sweep scan
    # (excludes the Move XY / Move Z that ran as earlier requests).
    sweep_scan_finished = pyqtSignal(bool, float)
    axial_scan_data_ready = pyqtSignal(object)
    # Result of a Check Progress request: a predefined_plan.ProgressReport
    # joining the sent plan against the currently-saved scans.
    predefined_progress_ready = pyqtSignal(object)
    # Outcome of a Load/Take Ref. Bkg. request: a short status label, or
    # an "ERROR: ..." string the frontend shows in a message box.
    ref_bkg_state = pyqtSignal(str)
    send_axial_scans_to_save = pyqtSignal(list)
    send_message_to_frontend = pyqtSignal(str, str)
    new_andor_display_ready = pyqtSignal()
    close_event_finished = pyqtSignal()
    laser_coord_calibration_ready = pyqtSignal(object)

    zaber_stage_positions_updated = pyqtSignal(float, float, float)
    # (x, y, z) positions in µm





    # Signals to Frontend
    update_system_state_in_frontend = pyqtSignal(SystemState)

    # Eye Tracking
    eye_tracking_result_ready = pyqtSignal(object)


    def __init__(self, backend: HiBackend):
        super().__init__()
        self.backend = backend
        self._running = False
        self._thread_active = False
        self._camera_shutter_open = True
        self._is_cancel_operations = False
        self._is_end_scan_early = False

        self._mb_lock = threading.Lock()
        self._mb_latest_display = None

        # in HiSignaller.__init__ (after moveToThread happens in the host)
        self._acq_timer = None
        self._acq_interval_ms = 20


        self.system_state = SystemState.IDLE

        self.backend.init_f2b_signals(
            cancel_callback=self.is_cancel_requested,
            end_scan_early_callback=self.is_end_scan_early_requested,
        )
        self.backend.init_b2f_emit_display_result(emit_display_result=self.emit_display_result)


        # Init Zaber position updates
        self.backend.init_b2f_zaber_position_updates_human_interface(
            emit_update_zaber_lens_position=self.update_zaber_lens_position
        )


    def update_system_state(self, new_state: SystemState):
        self.system_state = new_state
        self.update_system_state_in_frontend.emit(new_state)
        if new_state in (SystemState.BUSY, SystemState.FREERUNNING):
            self.reset_cancel()

    def update_gui(self):
        self.emit_is_illumination_continuous()
        self.emit_camera_settings()
        self.emit_do_live_fitting_state()
        self.update_stage_positions() #xyz stage
        self.update_zaber_lens_position(self.backend.zaber_eye_lens.get_position()) # lens


    def _mailbox_push_andor_display(self, display):
        need_notify = False
        with self._mb_lock:
            # if GUI hasn't consumed yet, mailbox is not None: just overwrite, don't notify
            if self._mb_latest_display is None:
                need_notify = True
            self._mb_latest_display = display
        if need_notify:
            self.new_andor_display_ready.emit()  # queued across threads

    def fetch_latest_andor_display(self):
        # consumer (GUI thread): swap out and return
        with self._mb_lock:
            item = self._mb_latest_display
            self._mb_latest_display = None
        return item

    @pyqtSlot()
    def emit_is_illumination_continuous(self):
        self.illumination_mode_state.emit(self.backend.is_shutter_open)

    @pyqtSlot()
    def toggle_do_live_fitting(self):
        self.backend.do_live_fitting = not self.backend.do_live_fitting
        log.info(f"Do Live Fitting toggled to: {self.backend.do_live_fitting}")
        self.do_live_fitting_state.emit(self.backend.do_live_fitting)

    @pyqtSlot()
    def emit_do_live_fitting_state(self):
        self.do_live_fitting_state.emit(self.backend.do_live_fitting)

    @pyqtSlot()
    def toggle_illumination_mode(self):
        if self.backend.is_shutter_open:
            self.backend.close_sample_shutter()
            self.stop_live_view()
            log.info("Switched to pulsed illumination")
        else:
            self.backend.open_sample_shutter()
            self.start_live_view()
            log.info("Switched to continuous illumination")

        self.illumination_mode_state.emit(self.backend.is_shutter_open)

    @pyqtSlot()
    def toggle_reference_mode(self):
        if self.backend.is_reference_mode:
            self.backend.change_to_sample_mode()
        else:
            self.backend.change_to_reference_mode()
        self.emit_camera_settings()
        self.reference_mode_state.emit(self.backend.is_reference_mode)

    @pyqtSlot()
    def emit_camera_settings(self):
        cam = self.backend.andor_camera
        settings = {
            "exposure": round(cam.get_exposure_time(),ndigits=4),
            "gain": cam.get_emccd_gain(),
        }
        self.camera_settings_ready.emit(settings)

    @pyqtSlot(dict)
    def apply_camera_settings(self, settings: dict):
        try:
            self.backend.set_andor_exposure(
                exposure_time=settings["exposure"],
                emccd_gain=settings["gain"],
            )
            log.info(f"Camera settings applied: {settings}")
        except Exception as e:
            log.info(f"Failed to apply camera settings: {e}")

    @pyqtSlot(object)
    def update_andor_config_settings(self, andor_config: AndorConfig):
        self.backend.update_andor_config_settings(andor_config)

    @pyqtSlot(object)
    def update_scanning_config(self, scanning_config: ScanningConfig):
        self.backend.update_scanning_config_file(scanning_config)

    @pyqtSlot(object)
    def update_sweep_scan_config(self, sweep_config):
        self.backend.update_sweep_scan_config(sweep_config)

    @pyqtSlot(FittingConfigs)
    def update_fitting_configs(self, fitting_configs: FittingConfigs):
        self.backend.spectrum_fitter.update_configs(fitting_configs)

    @pyqtSlot()
    def toggle_camera_shutter(self):
        try:
            cam = self.backend.andor_camera

            if self._camera_shutter_open:
                cam.close_shutter()
                self._camera_shutter_open = False
                log.info("Camera shutter closed.")
            else:
                cam.open_shutter()
                self._camera_shutter_open = True
                log.info("Camera shutter opened.")

            self.camera_shutter_state_changed.emit(self._camera_shutter_open)

        except Exception as e:
            log.warning(f"Failed to toggle camera shutter: {e}")

    def update_zaber_lens_position(self, pos: float):
        self.zaber_lens_position_updated.emit(pos)

    @pyqtSlot(float)
    def move_zaber_stage_x_relative(self, step: float):
        def do():
            self.backend.zaber_hi.move_rel(dx=step)
            self.update_stage_positions()

        self._pause_acq_for(do)

    @pyqtSlot(float)
    def move_zaber_stage_y_relative(self, step: float):
        self._pause_acq_for(lambda: (self.backend.zaber_hi.move_rel(dy=step),
                                     self.update_stage_positions()))

    @pyqtSlot(float)
    def move_zaber_stage_z_relative(self, step: float):
        self._pause_acq_for(lambda: (self.backend.zaber_hi.move_rel(dz=step),
                                     self.update_stage_positions()))

    @pyqtSlot(float)
    def move_zaber_eye_lens_relative(self, step: float):
        def do():
            self.backend.zaber_eye_lens.move_rel(step)
            pos = self.backend.zaber_eye_lens.get_position()
            self.zaber_lens_position_updated.emit(pos)

        self._pause_acq_for(do)

    def update_stage_positions(self):
        try:
            x, y, z = self.backend.zaber_hi.get_position()
            self.zaber_stage_positions_updated.emit(x, y, z)
        except Exception as e:
            log.warning(f"Failed to read stage positions: {e}")

    @pyqtSlot(float)
    def set_microwave_frequency(self, freq: float):
        try:
            self.backend.microwave.set_frequency(freq)
            freq_real = self.backend.microwave.get_frequency()
            log.info(f"Microwave frequency set to {freq_real:.3f} GHz")
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            log.warning(f"Failed to set microwave frequency: {e}")


    @pyqtSlot()
    def emit_microwave_frequency(self):
        try:
            freq = self.backend.microwave.get_frequency()
            self.microwave_frequency_updated.emit(freq)
        except Exception as e:
            log.warning(f"Failed to read microwave frequency: {e}")

    @pyqtSlot()
    def snap_and_fit(self):
        try:
            frame = self.backend.get_andor_frame()
            self.backend.display_spectrum(frame=frame)

        except Exception as e:
            log.warning(f"Error snapping frame: {e}")

    @pyqtSlot()
    def start_live_view(self):
        if self._running:
            return
        self.backend.init_shutters()
        self._running = True
        self.update_system_state(new_state=SystemState.FREERUNNING)

        if self._acq_timer is None:
            self._acq_timer = QtCore.QTimer(self)  # parent = self (lives in worker)
            self._acq_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            self._acq_timer.timeout.connect(self._acq_tick)  # local, same-thread

        self._acq_timer.start(self._acq_interval_ms)

    @pyqtSlot()
    def stop_live_view(self):
        self._running = False
        if self._acq_timer:
            self._acq_timer.stop()
        self.update_system_state(new_state=SystemState.IDLE)

    def restart_live_view_when_ready(self):
        QTimer.singleShot(0, self.start_live_view)


    def _pause_acq_for(self, fn):
        was_running = self._running
        if self._acq_timer:
            self._acq_timer.stop()
        self._running = False
        try:
            return fn()
        finally:
            if was_running:
                self._running = True
                if self._acq_timer:
                    self._acq_timer.start(self._acq_interval_ms)

    def _acq_tick(self):
        # run at most one acq at a time; coalesce timeouts
        if not self._running or self._acq_timer is None:
            return

        self._acq_timer.stop()

        try:
            with self._mb_lock:
                if self._mb_latest_display is not None:
                    return  # GUI hasn't consumed yet; skip doing work
            self.snap_and_fit()
        finally:
            # schedule the next tick after we’re done (coalesced)
            if self._running and self._acq_timer:
                self._acq_timer.start(self._acq_interval_ms)

    def cancel_operations(self):
        self._is_cancel_operations = True

    def end_scan_early(self):
        """Request that the running scan stop early but SAVE what it has
        collected and count as successful (see take_sweep_scan)."""
        self._is_end_scan_early = True

    def update_stored_axial_scans(self):
        lines = self.backend.get_list_of_axial_scans()
        self.send_update_stored_axial_scans.emit(lines)

    def compute_predefined_progress(self, planned):
        """Join the sent plan (list[PlanStep]) against the currently-saved
        scans and emit a ProgressReport. Fast: only reads each scan's stored
        reflection crossings and lens positions — no fitting, no VIPA images.
        A scan the operator removed from the list is gone from
        axial_scan_dict, so it correctly stops counting as taken."""
        report = build_progress(
            planned, list(self.backend.axial_scan_dict.values()))
        self.predefined_progress_ready.emit(report)

    # ------- State control -------
    # SLOTS
    # in BrillouinSignaller
    #  cancel callback to pass down to the backend
    def is_cancel_requested(self) -> bool:
        # Either stop request halts polling loops; the sweep checks
        # end-scan-early first so it saves before the cancel path runs.
        return self._is_cancel_operations or self._is_end_scan_early

    def is_end_scan_early_requested(self) -> bool:
        return self._is_end_scan_early


    @pyqtSlot()
    def reset_cancel(self):
        self._is_cancel_operations = False
        self._is_end_scan_early = False


    @pyqtSlot()
    def get_calibration_results(self):
        self.backend.update_calibration_calculator() # this recalculates the calibration
        self.calibration_result_ready.emit(
            (self.backend.calibration_data, self.backend.calibration_calculator, self.backend.calibration_config))


    @pyqtSlot(object)
    def update_calibration_config_backend(self, config: CalibrationConfig):
        self.backend.update_calibration_config(config)

    def emit_display_result(self, display: DisplayResults):
        self._mailbox_push_andor_display(display)

    @pyqtSlot()
    def run_calibration(self):
        old_state = self.system_state
        self.update_system_state(new_state=SystemState.BUSY)
        try:
            is_sucess = scan_procedures.perform_calibration(self.backend)
            if is_sucess:
                self.calibration_finished.emit()
            else:
                self.calibration_failed.emit()
        finally:
            self.update_system_state(new_state=old_state)

    @pyqtSlot(object)
    def take_axial_step_scan(self, request_axial_scan: RequestAxialStepScan):
        """
        Generates a series of ZaberPositions using the given axis, n, and step,
        then takes measurements and updates the GUI accordingly.
        """
        if self.backend.calibration_poly_fit_params is None:
            self.send_message_to_user('Warning', "No Calibration available. Run Calibration first.")
            return

        old_state = self.system_state
        self.stop_live_view()
        self.update_system_state(new_state=SystemState.BUSY)
        QCoreApplication.processEvents()

        try:
            scan_procedures.take_axial_step_scan(self.backend, request_axial_scan)
            self.update_stored_axial_scans()
        finally:
            self.update_system_state(new_state=old_state)
            self.restart_live_view_when_ready()

    @pyqtSlot(object)
    def take_sweep_scan(self, request):
        """
        Run the in-out sweep scan (repeated find-measure-find cycles) and
        register the result in the stored axial scans.
        """
        if self.backend.calibration_poly_fit_params is None:
            self.send_message_to_user('Warning', "No Calibration available. Run Calibration first.")
            self.sweep_scan_finished.emit(False, 0.0)
            return

        old_state = self.system_state
        self.stop_live_view()
        self.update_system_state(new_state=SystemState.BUSY)
        QCoreApplication.processEvents()

        success = False
        # Time the sweep from here - the Move XY / Move Z requests already ran
        # in earlier slots, so this measures the sweep scan itself.
        t_start = time.monotonic()
        elapsed_s = 0.0
        # Tell the frontend the sweep is starting now (after the moves) so it
        # can arm a hard max-time backstop; pass the budget the backend uses.
        budget_s = float(getattr(self.backend.sweep_scan_config, "max_time_s", 0.0) or 0.0)
        self.sweep_scan_started.emit(budget_s)
        try:
            success = bool(scan_procedures.take_sweep_scan(self.backend, request))
            elapsed_s = time.monotonic() - t_start
            self.update_stored_axial_scans()
        finally:
            self.update_system_state(new_state=old_state)
            self.restart_live_view_when_ready()
            self.sweep_scan_finished.emit(success, elapsed_s)

    @pyqtSlot()
    def calibrate_laser_camera_position_delegate(self):
        # log.info(f"I disabled this function in normal use (for safety).\n"
        #          f"Go to hi_signaller.py -> def calibrate_laser_camera_position_delegate() and activate it "
        #          f"when required")
        laser_offset: LaserOffset = self.backend.run_laser_xy_calibration()
        self.laser_coord_calibration_ready.emit(laser_offset)


    @pyqtSlot(int)
    def handle_request_axial_scan_data(self, index: int):
        scan_data = self.backend.get_axial_scan_data(index)  # pure Python call
        if scan_data is not None:
            self.axial_scan_data_ready.emit(scan_data)
        else:
            log.warning(f"Requested scan index {index} not found.")

    @pyqtSlot(str)
    def handle_load_ref_bkg_from_file(self, path: str):
        """Load/build a reflection background from a file — runs in the
        backend thread so the calibration re-fit never blocks the GUI."""
        self.ref_bkg_state.emit(
            self.backend.load_reflection_background_from_file(path))

    @pyqtSlot(int)
    def handle_load_ref_bkg_from_scan(self, index: int):
        """Adopt a stored scan (a just-taken 'Take Ref. Bkg.' capture) as
        the live reflection background."""
        self.ref_bkg_state.emit(
            self.backend.load_reflection_background_from_scan(index))

    @pyqtSlot()
    def delegate_find_reflection_plane(self):
        self._do_find_reflection_plane(is_go_forwards=True)

    @pyqtSlot()
    def delegate_find_reflection_plane_backwards(self):
        self._do_find_reflection_plane(is_go_forwards=False)

    def _do_find_reflection_plane(self, is_go_forwards: bool):
        old_state = self.system_state
        was_running = self._running  # remember if live view was on

        self.stop_live_view()
        self.update_system_state(new_state=SystemState.BUSY)
        QCoreApplication.processEvents()

        try:
            z0 = self.backend.zaber_eye_lens.get_position()
            result = self.backend.find_reflection_plane(is_go_forwards=is_go_forwards)
            if result.found:
                z = result.event_z_um + result.z_offset_um
            else:
                z = z0
            self.backend.zaber_eye_lens.move_abs(z)
            self.update_zaber_lens_position(z)

        finally:
            self.update_system_state(new_state=old_state)
            if was_running:
                self.restart_live_view_when_ready()


    @pyqtSlot()
    def close_all_shutters(self):
        self.backend.shutter_manager.close_all()



    # ----------------- Saving -----------------------
    @pyqtSlot()
    def save_all_axial_scans(self):
        scans = list(self.backend.axial_scan_dict.values())
        self.send_axial_scans_to_save.emit(scans)

    @pyqtSlot(list)
    def save_multiple_axial_scans(self, indices: list[int]):
        scans = [
            scan for i, scan in self.backend.axial_scan_dict.items()
            if i in indices
        ]
        self.send_axial_scans_to_save.emit(scans)


    @pyqtSlot(list)
    def remove_selected_axial_scans(self, indices: list[int]):
        removed = 0
        for index in indices:
            if index in self.backend.axial_scan_dict:
                del self.backend.axial_scan_dict[index]
                removed += 1
                log.info(f"Removed axial scan {index}")
            else:
                log.warning(f"Scan {index} not found for removal")

        if removed > 0:
            self.update_stored_axial_scans()

    @pyqtSlot()
    def send_message_to_user(self, title: str, message: str):
        self.send_message_to_frontend.emit(title, message)



    def close(self):
        print("Stopping BrillouinWorker and closing hardware...")
        self._running = False
        try:
            self.backend.close()
        except Exception as e:
            print(f"Error during backend shutdown: {e}")
        finally:
            self.close_event_finished.emit()

