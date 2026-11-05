
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np


from brillouin_system.devices.cameras.andor.andor_frame.andor_config import andor_frame_config, AndorConfig
from brillouin_system.calibration.config.calibration_config import CalibrationConfig, calibration_config
from brillouin_system.devices.cameras.andor.baseCamera import BaseCamera
from brillouin_system.devices.cameras.andor.dummyCamera import DummyCamera
from brillouin_system.devices.cameras.andor.ixonUltra import IxonUltra
from brillouin_system.devices.microwave_device import Microwave, MicrowaveDummy

from brillouin_system.devices.shutter_device import ShutterManager, ShutterManagerDummy
from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_human_interface import ZaberHumanInterface, \
    ZaberHumanInterfaceDummy
from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import LaserOffset, \
    CalibRigLaserPosition

from brillouin_system.scan_managers.homodyne_plane_finder import HomodyneFinderParams, find_reflection_homodyne
from brillouin_system.scan_managers.ni_reflection_finder4 import ReflectionResult
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig, \
    axial_scanning_config
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import SweepScanConfig, \
    sweep_scan_config
from brillouin_system.logging_utils.logging_setup import get_logger

from brillouin_system.my_dataclasses.display_results import DisplayResults
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.my_dataclasses.system_state import SystemState
from brillouin_system.calibration.calibration import CalibrationData, \
    CalibrationCalculator, CalibrationPolyfitParameters, calibrate
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum

from brillouin_system.devices.zaber_engines.zaber_human_interface.zaber_eye_lens import ZaberEyeLens
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter, \
    config_requires_reflection_background, config_requires_dho_axes
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackgroundMapper,
    get_current_background,
)


log = get_logger(__name__)

# Dummy mode only: how far below the simulated cornea the eye-lens focus should
# sit when the eye-tracker Δc reads zero. The reflection find searches FORWARD
# (increasing z), so it needs the lens parked below the plane with room to
# approach; a Δc=0 that left the lens exactly on the plane makes the forward
# find unreliable. 300 µm is comfortably inside the search range and gave 6/6
# robust dummy sweeps in testing. Physically it stands in for the offset
# between the eye tracker's corneal-corner reference and the detected plane.
SIM_DELTAC_MARGIN_UM = 300.0


class HiBackend:


    def __init__(self,
                 use_dummy: bool = True
                 ):


        # init Spectrum Fitter:
        self.spectrum_fitter = SpectrumFitter()

        # Reflection background ("ReflectionBG") for the reflection
        # background / prmr preset: the current template comes from the
        # runtime registry (no default fallback — with none loaded, fits
        # warn and drop the reflection term); the mapper is rebuilt whenever
        # the template, calibration or row band changes.
        self._reflection_mapper: ReflectionBackgroundMapper | None = None
        self._reflection_mapper_key = None
        # Strong ref to the calculator the cached mapper was built from: the
        # mapper itself only copies the polynomials, and the cache key uses
        # id(calculator) — without this ref a replaced calculator could be
        # garbage collected and its id reused by the next one, letting a
        # stale key match and the old registration survive a recalibration.
        self._reflection_mapper_calc = None

        # Devices
        if use_dummy:
            shutter_manager=ShutterManagerDummy('human_interface')
            microwave=MicrowaveDummy()
            # Synthetic two-peak frames; in reference mode the EOM peaks
            # follow the dummy microwave, so calibration sweeps work.
            camera=DummyCamera(microwave=microwave, shutter_manager=shutter_manager)
            # Simulated NI + eye lens with a moving simulated cornea: the DAQ
            # signal is coupled to the lens-cornea distance, so reflection
            # finding actually works in dummy mode.
            from brillouin_system.patient_movement_analysis.simulated_devices import (
                SimNI, SimZaberLens, SimulatedCornea)
            sim_cornea = SimulatedCornea()
            zaber_eye_lens = SimZaberLens(start_um=9000.0)
            ni = SimNI(zaber_eye_lens, sim_cornea)
            from brillouin_system.guis.homodyne_test.sim_devices import SimHomodyneDAQ
            homodyne_daq = SimHomodyneDAQ(zaber_eye_lens, cornea=sim_cornea,
                                          peak_amplitude_v=0.2, noise_std_v=0.006)
            zaber_hi=ZaberHumanInterfaceDummy()
            # Ground-truth cornea shared with the reflection finder, so the
            # eye-tracker Δc can be made consistent with it in dummy mode
            # (see simulated_delta_laser_corner_mm). None on real hardware.
            self._sim_cornea = sim_cornea

        else:
            self._sim_cornea = None
            try:
                camera=IxonUltra(
                    index = 0,
                    temperature = "off",
                    fan_mode = "full",
                    x_start = 40, x_end  = 120,
                    y_start= 300, y_end  = 315,
                    vbin= 1, hbin  = 1,
                    verbose = True,
                    advanced_gain_option=False
                )
                # camera = DummyCamera()
            except Exception as e:
                raise RuntimeError("Camera not connected") from e

            shutter_manager=ShutterManager('human_interface')
            try:
                microwave=Microwave()
                # microwave = MicrowaveDummy()
            except Exception as e:
                raise RuntimeError("Microwave not connected") from e
            zaber_eye_lens=ZaberEyeLens()
            zaber_hi=ZaberHumanInterface()
            from brillouin_system.devices.ni.ni6008 import NI6008
            ni = NI6008()
            from brillouin_system.devices.ni.ni6008_multi import NI6008Multi
            homodyne_daq = NI6008Multi()    # ai0, ai1, RSE

        self.andor_camera: BaseCamera | DummyCamera | IxonUltra = camera
        self._andor_config: AndorConfig = andor_frame_config.get()
        self.update_andor_config_settings(andor_config=self._andor_config)

        self.axial_scan_config: ScanningConfig = axial_scanning_config.get()
        self.sweep_scan_config: SweepScanConfig = sweep_scan_config.get()

        self.shutter_manager: ShutterManager | ShutterManagerDummy = shutter_manager

        self.microwave: Microwave | MicrowaveDummy = microwave

        self.microwave.set_power(power_dbm=-20)

        self.zaber_eye_lens = zaber_eye_lens
        self.zaber_hi = zaber_hi

        # DAQ
        self.ni = ni
        self.homodyne_daq = homodyne_daq

        # State
        self.is_shutter_open: bool = True
        self.is_reference_mode: bool = False
        self.do_live_fitting = False

        # Calibration
        self.calibration_data: CalibrationData | None = None
        self.calibration_poly_fit_params: CalibrationPolyfitParameters | None = None
        self.calibration_calculator: CalibrationCalculator | None = None
        self.calibration_config: CalibrationConfig = calibration_config.get()

        self.init_shutters()
        self.init_camera_settings()

        # Init state modes
        self.reference_state_mode: SystemState = self.init_state_mode(is_reference_mode=True)
        self.sample_state_mode: SystemState = self.init_state_mode(is_reference_mode=False)


        # Init Signals (they are sent down from the signaller)
        # b2f = backend to frontend
        # f2b = frontend to backend
        self.b2f_send_system_state_signal = None
        self.b2f_emit_display_result = None
        self.f2b_cancel_callback: Callable[[], bool] = lambda: False
        # "End scan early": stop remaining work but save what's collected and
        # treat the scan as successful. Polled by take_sweep_scan.
        self.f2b_end_scan_early_callback: Callable[[], bool] = lambda: False

        # Init Zaber position Signals
        # Human Interface
        self.b2f_emit_update_zaber_lens_position = None



        # Store measurements for Human Interface
        self._i_axial_scans: int = 0
        self.axial_scan_dict: dict[int, AxialScan] = {}

    def get_list_of_axial_scans(self) -> list[str]:
        if not self.axial_scan_dict:
            return [""]
        lines = [f"{scan.i} - ID: {scan.id}" for scan in sorted(self.axial_scan_dict.values(), key=lambda s: s.i)]
        return lines

    def init_shutters(self):
        if self.is_reference_mode:
            self.shutter_manager.change_to_reference()
        else:
            self.shutter_manager.change_to_objective()
        if self.is_shutter_open:
            self.shutter_manager.sample.open()
        else:
            self.shutter_manager.sample.close()

    def init_camera_settings(self):
        andor_config = self._andor_config

        self.andor_camera.set_fixed_pre_amp_mode(index=andor_config.pre_amp_mode)
        self.andor_camera.set_vss_index(index=andor_config.vss_index)

        self.andor_camera.set_roi(x_start=andor_config.x_start,
                                  x_end=andor_config.x_end,
                                  y_start=andor_config.y_start,
                                  y_end=andor_config.y_end, )
        self.andor_camera.set_binning(hbin=andor_config.hbin,
                                      vbin=andor_config.vbin)

        self.andor_camera.set_flip_image_horizontally(flip=andor_config.flip_image_horizontally)


    def init_state_mode(self, is_reference_mode: bool) -> SystemState:
        return SystemState(
            is_reference_mode=is_reference_mode,
            andor_camera_info=self.andor_camera.get_camera_info_dataclass()
        )

    def update_calibration_config(self, config: CalibrationConfig):
        self.calibration_config = config

    def init_f2b_signals(self, cancel_callback: Callable[[], bool],
                         end_scan_early_callback: Callable[[], bool] | None = None):
        self.f2b_cancel_callback = cancel_callback
        if end_scan_early_callback is not None:
            self.f2b_end_scan_early_callback = end_scan_early_callback

    def init_b2f_emit_display_result(self, emit_display_result: Callable[[DisplayResults], None]):
        self.b2f_emit_display_result = emit_display_result

    def init_b2f_zaber_position_updates_human_interface(self,
                                                        emit_update_zaber_lens_position:
                                                        Callable[[float], None]):
        self.b2f_emit_update_zaber_lens_position = emit_update_zaber_lens_position

    def move_and_update_gui_zaber_eye_lens_rel(self, dz_um: float) -> float:
        """
        Move Zaber eye lens by a relative distance (µm).

        Returns:
            New absolute lens position (µm).
        """
        self.zaber_eye_lens.move_rel(dz_um)
        z = self.zaber_eye_lens.get_position()
        if self.b2f_emit_update_zaber_lens_position:
            self.b2f_emit_update_zaber_lens_position(z)
        return z

    def move_and_update_gui_zaber_eye_lens_abs(self, z_um: float) -> float:
        """
        Move Zaber eye lens to an absolute position (µm).

        Returns:
            New absolute lens position (µm).
        """
        self.zaber_eye_lens.move_abs(z_um)
        z = self.zaber_eye_lens.get_position()
        if self.b2f_emit_update_zaber_lens_position:
            self.b2f_emit_update_zaber_lens_position(z)
        return z

    # ---------------- Change Modes ----------------



    def open_sample_shutter(self):
        self.is_shutter_open = True
        self.shutter_manager.objective.open()
        log.info("[BrillouinBackend] Switched to continuous illumination mode.")

    def close_sample_shutter(self):
        self.is_shutter_open = False
        self.shutter_manager.objective.close()
        log.info("[BrillouinBackend] Switched to pulsed illumination mode.")

    def change_system_state(self, state_mode: SystemState):
        self.is_reference_mode = state_mode.is_reference_mode
        self.set_andor_exposure(
            exposure_time=state_mode.andor_camera_info.exposure,
            emccd_gain=state_mode.andor_camera_info.gain,
        )

    def get_current_system_state(self) -> SystemState:
        return SystemState(
            is_reference_mode=self.is_reference_mode,
            andor_camera_info=self.andor_camera.get_camera_info_dataclass()
        )

    def change_to_reference_mode(self):
        # Store current state mode of the sample for future:
        self.sample_state_mode = self.get_current_system_state()

        self.shutter_manager.change_to_reference()
        self.change_system_state(state_mode=self.reference_state_mode)
        log.info("[BrillouinBackend] Switched to reference mode.")


    def change_to_sample_mode(self):
        # Store current state mode of the sample for future:
        self.reference_state_mode = self.get_current_system_state()
        self.shutter_manager.reference.close()
        if self.is_shutter_open:
            self.shutter_manager.objective.open()
        self.change_system_state(state_mode=self.sample_state_mode)
        log.info("[BrillouinBackend] Switched to sample mode.")



    # ---------------- Get Frames  ----------------
    def _get_andor_camera_snap(self) -> np.ndarray:
        """Pull a raw frame from the camera.
        Returns: frame, time.time()
        """
        frame = self.andor_camera.snap()
        return frame.astype(np.float64)


    def get_andor_frame(self) -> np.ndarray:
        return self._get_andor_camera_snap()

    def get_fitted_spectrum(self, frame) -> FittedSpectrum:
        """
        Fits a Brillouin spectrum depending on reference mode and background subtraction.
        If live fitting is disabled, returns an unsuccessful fit but includes a raw spectrum line.

        Args:
            frame (np.ndarray): The input camera frame.

        Returns:
            FittedSpectrum: Dataclass containing fit results and metadata.
        """


        px, sline = self.spectrum_fitter.get_px_sline_from_image(frame)

        if not self.do_live_fitting and not self.is_reference_mode:
            return self.spectrum_fitter.get_empty_fitting(px, sline)

        try:
            reflection_bg = self._reflection_background_if_required(px)
            dho_axes = self._dho_axes_if_required()
            return self.spectrum_fitter.fit(px, sline, is_reference_mode=self.is_reference_mode,
                                            reflection_background=reflection_bg,
                                            dho_axes=dho_axes)
        except Exception as e:
            log.info(f"Fitting error: {e}")
            return self.spectrum_fitter.get_empty_fitting(px, sline)

    def calibration_data_to_store(self) -> CalibrationData | None:
        """The raw calibration frames to travel with a scan (None only when
        no calibration has been taken yet).

        ALWAYS stored — the off-toggle was removed 2026-08-24 (user
        decision): re-fitting a scan against its OWN calibration and
        anchoring a reflection-background template both need the frames,
        and the fitted polynomials alone cannot be re-derived with a
        different lineshape model.
        """
        return self.calibration_data

    def _reflection_background_if_required(self, px) -> np.ndarray | None:
        """The mapped reflection background for sample fits, or None.

        Only built when the sample config uses the reflection background
        (the prmr preset). The CURRENT template (runtime registry, no
        default fallback — None makes fit() warn once and drop the
        reflection term) is registered onto the CURRENT calibration in
        frequency space, so it survives VIPA realignment; raises if no
        calibration is loaded (no fallback)."""
        if self.is_reference_mode:
            return None
        if not config_requires_reflection_background(self.spectrum_fitter.sample_config):
            return None
        background = get_current_background()
        if background is None:
            return None
        if self.calibration_calculator is None:
            raise ValueError(
                "Background 'reflection' requires a calibration to "
                "register the reflection background, but none is loaded."
            )
        try:
            # The SAME rows the sample sline sums (manual: the config band;
            # auto: the band frozen by auto_select_rows). Before an auto band
            # exists there is nothing to sum the template over — skip the
            # term for this frame, the next fitted frame freezes the band.
            rows = self.spectrum_fitter.get_selected_rows()
        except ValueError:
            return None
        margin = getattr(self.spectrum_fitter.sample_config,
                         "reflection_margin_ghz", None)
        key = (id(background), id(self.calibration_calculator), tuple(rows),
               margin)
        if self._reflection_mapper is None or self._reflection_mapper_key != key:
            self._reflection_mapper = ReflectionBackgroundMapper(
                background, self.calibration_calculator,
                rows=rows,
                g_margin_ghz=margin)
            self._reflection_mapper_key = key
            self._reflection_mapper_calc = self.calibration_calculator
        return self._reflection_mapper.render(px)

    def _dho_axes_if_required(self):
        """The per-peak calibration axes for 'dho_x_psf' sample fits, or None.

        No degraded fallback (unlike the reflection template): without a
        calibration the DHO cannot fit at all, so this raises — the live
        loop catches it per frame and shows the unfitted sline until a
        calibration is taken."""
        if self.is_reference_mode:
            return None
        if not config_requires_dho_axes(self.spectrum_fitter.sample_config):
            return None
        if self.calibration_calculator is None:
            raise ValueError(
                "Model 'dho_x_psf' needs the current calibration's frequency "
                "tracks and instrument widths, but no calibration is loaded."
            )
        return self.calibration_calculator.dho_axes()

    def _na_shift_ratio(self) -> float | None:
        """The post-hoc NA cone factor for the CURRENT sample config,
        cached on the NA parameters (the integral is not free per frame).
        None when unconfigured/uncomputable — display stays raw then."""
        cfg = self.spectrum_fitter.sample_config
        key = (getattr(cfg, "na_weighting", None),
               getattr(cfg, "na_collection", None),
               getattr(cfg, "na_beam_diameter_mm", None),
               getattr(cfg, "na_focal_length_mm", None),
               getattr(cfg, "na_n_sample", None))
        cached = getattr(self, "_na_ratio_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from brillouin_system.spectrum_fitting.na_lineshape import (
                na_mean_shift_ratio)
            ratio = float(na_mean_shift_ratio(cfg))
        except Exception:
            ratio = None
        self._na_ratio_cache = (key, ratio)
        return ratio

    # -------- Reflection background loading (live sample fits) --------

    def load_reflection_background_from_file(self, path: str) -> str:
        """Make a saved .npz template or an .h5 scan the live reflection
        background. Returns a short status label, or 'ERROR: ...'."""
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        try:
            from brillouin_system.spectrum_fitting.reflection_background import (
                ReflectionBackground, set_current_background)
            if str(path).lower().endswith(".npz"):
                set_current_background(ReflectionBackground.load(path))
                log.info(f"[Backend] Live reflection background loaded "
                         f"from template {name}.")
                return name
            from brillouin_system.saving_and_loading.known_dataclasses_lookup import (
                known_classes)
            from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
                dict_to_dataclass_tree, load_dict_from_hdf5)
            loaded = dict_to_dataclass_tree(load_dict_from_hdf5(path),
                                            known_classes)
            scans = loaded if isinstance(loaded, list) else [loaded]
            # Prefer a background capture; fall back to the last scan.
            bg_scans = [s for s in scans
                        if "reflection" in str(getattr(s, "id", ""))]
            scan = (bg_scans or scans)[-1]
            return self._adopt_scan_as_reflection_background(
                scan, f"{name} ({scan.id})")
        except Exception as e:
            log.exception("[Backend] Failed to load the reflection "
                          "background")
            return f"ERROR: could not load {name}: {type(e).__name__}: {e}"

    def load_reflection_background_from_scan(self, index: int) -> str:
        """Adopt stored scan #index (a 'Take Ref. Bkg.' capture) as the
        live reflection background. Returns a status label or 'ERROR:'."""
        scan = self.get_axial_scan_data(index)
        if scan is None:
            return f"ERROR: scan index {index} not found."
        return self._adopt_scan_as_reflection_background(
            scan, f"scan {index} - {scan.id}")

    def _adopt_scan_as_reflection_background(self, scan, label: str) -> str:
        try:
            from brillouin_system.spectrum_fitting.build_reflection_background import (
                background_from_scan)
            from brillouin_system.spectrum_fitting.reflection_background import (
                set_current_background)
            log.info(f"[Backend] Building the reflection background from "
                     f"scan '{scan.id}' ({len(scan.measurements)} "
                     f"frames)...")
            set_current_background(background_from_scan(scan, source=label))
            log.info(f"[Backend] Live reflection background set ({label}) "
                     f"— sample fits with a 'reflection' background now "
                     f"use it.")
            return label
        except Exception as e:
            log.exception("[Backend] Failed to build the reflection "
                          "background")
            return (f"ERROR: could not build a background from "
                    f"'{getattr(scan, 'id', '?')}': "
                    f"{type(e).__name__}: {e}")

    def update_calibration_calculator(self):
        if self.calibration_data is None:
            self.calibration_poly_fit_params = None
            self.calibration_calculator = None
        else:
            # Same fitter as the live frames: it holds the frozen row band, and
            # the band must not move between a calibration and its samples.
            self.calibration_poly_fit_params = calibrate(data=self.calibration_data,
                                                         polyfit_degree=self.calibration_config.degree,
                                                         fitter=self.spectrum_fitter)
            self.calibration_calculator: CalibrationCalculator = CalibrationCalculator(
                parameters=self.calibration_poly_fit_params)




    # The acquisition procedures (axial step scan, sweep scan, calibration)
    # live in scan_procedures.py — functions over this backend. The backend
    # keeps device state and the primitives they compose.

    def next_axial_scan_index(self) -> int:
        self._i_axial_scans += 1
        return self._i_axial_scans

    def register_axial_scan(self, axial_scan: AxialScan) -> None:
        self.axial_scan_dict[axial_scan.i] = axial_scan
    def update_sweep_scan_config(self, sweep_config: SweepScanConfig):
        self.sweep_scan_config = sweep_config
        log.info(f"[Sweep Scan] Config updated: {sweep_config}")

    def display_spectrum(self, frame):
        fs = self.get_fitted_spectrum(frame)
        self.b2f_emit_display_result(self.get_display_results(frame=frame, fitting=fs))


    def get_axial_scan_data(self, index: int):
        try:
            return self.axial_scan_dict[index]
        except (IndexError, KeyError):
            return None

    def get_freq_shift(self, fitting: FittedSpectrum) -> float | None:
        if self.calibration_calculator is None or not fitting.is_success:
            return None
        calc = self.calibration_calculator
        reference = self.calibration_config.reference
        if reference == "left":
            return float(calc.freq_left_peak(fitting.left_peak_center_px))
        if reference == "right":
            return float(calc.freq_right_peak(fitting.right_peak_center_px))
        if reference == "combined":
            combined = calc.combined_shift(fitting)
            return combined.combined_ghz if combined is not None else None
        return float(calc.freq_peak_distance(fitting.inter_peak_distance))

    def get_hwhm_shift(self, fitting: FittedSpectrum) -> tuple:
        """

        Args:
            fitting:

        Returns: hwhm_left_peak_ghz, hwhm_right_peak_ghz

        The raw fitted width, as the peak lands on the detector — what you want
        while watching the live spectrum. The instrument-subtracted sample
        linewidth is an analysis output (see AnalyzedFreqShifts).
        """
        if self.calibration_calculator is None:
            return None, None
        else:
            return self.calibration_calculator.hwhm_ghz(fitting)


    def get_display_results(self, frame: np.ndarray, fitting: FittedSpectrum) -> DisplayResults:
        # Sample linewidth vs the LAST calibration — sample mode only.
        # sample_linewidth_ghz itself returns (None, None) for non-PSF
        # (plain lorentzian) fits and width-less calibrations, so the
        # frontend can blank on None without knowing why.
        linewidth_lp_ghz, linewidth_rp_ghz = None, None
        shift_lp_ghz, shift_rp_ghz = None, None
        if self.is_reference_mode:
            freq_shift_ghz = self.microwave.get_frequency()
            hwhm_lp_ghz, hwhm_rp_ghz = self.get_hwhm_shift(fitting)

        elif fitting.is_success:
            freq_shift_ghz = self.get_freq_shift(fitting)
            hwhm_lp_ghz, hwhm_rp_ghz = self.get_hwhm_shift(fitting)
            if self.calibration_calculator is not None:
                linewidth_lp_ghz, linewidth_rp_ghz = (
                    self.calibration_calculator.sample_linewidth_ghz(fitting))
        else:
            freq_shift_ghz = None
            hwhm_lp_ghz, hwhm_rp_ghz = None, None

        # Per-peak frequencies from the calibration tracks (both modes) —
        # the frontend shows their difference as the live lean meter.
        if fitting.is_success and self.calibration_calculator is not None:
            try:
                shift_lp_ghz = float(self.calibration_calculator
                                     .freq_left_peak(
                                         fitting.left_peak_center_px))
                shift_rp_ghz = float(self.calibration_calculator
                                     .freq_right_peak(
                                         fitting.right_peak_center_px))
            except Exception:
                shift_lp_ghz, shift_rp_ghz = None, None

        # Post-hoc NA cone correction on the displayed SHIFTS (sample mode
        # only — never inside the fit, and never on the reference/EOM
        # frequencies). Widths stay uncorrected (the NA width term is a
        # flat sub-MHz analysis-side correction).
        na_corrected = False
        if not self.is_reference_mode:
            ratio = self._na_shift_ratio()
            if ratio is not None and ratio != 1.0:
                na_corrected = True
                if freq_shift_ghz is not None:
                    freq_shift_ghz = freq_shift_ghz / ratio
                if shift_lp_ghz is not None:
                    shift_lp_ghz = shift_lp_ghz / ratio
                if shift_rp_ghz is not None:
                    shift_rp_ghz = shift_rp_ghz / ratio

        if fitting.is_success:
            return DisplayResults(
                is_fitting_available=True,
                frame=frame,
                x_pixels=fitting.x_pixels,
                sline=fitting.sline,
                mask_for_fitting=fitting.mask_for_fitting,
                x_fit_refined=fitting.x_fit_refined,
                y_fit_refined=fitting.y_fit_refined,
                inter_peak_distance=fitting.inter_peak_distance,
                freq_shift_ghz=freq_shift_ghz,
                hwhm_left_peak=hwhm_lp_ghz,
                hwhm_right_peak=hwhm_rp_ghz,
                linewidth_left_peak=linewidth_lp_ghz,
                linewidth_right_peak=linewidth_rp_ghz,
                shift_left_peak=shift_lp_ghz,
                shift_right_peak=shift_rp_ghz,
                na_corrected=na_corrected,
            )
        else:
            return DisplayResults(
                is_fitting_available=False,
                frame=frame,
                x_pixels=fitting.x_pixels,
                sline=fitting.sline,
            )

    def set_andor_exposure(self,
                            exposure_time: float,
                            emccd_gain: int,
                            ):

        self.andor_camera.set_exposure_time(seconds=exposure_time)
        self.andor_camera.set_emccd_gain(gain=emccd_gain)


        if self.is_reference_mode:
            self.reference_state_mode.andor_camera_info.exposure = exposure_time
            self.reference_state_mode.andor_camera_info.gain = emccd_gain
        else:
            self.sample_state_mode.andor_camera_info.exposure = exposure_time
            self.sample_state_mode.andor_camera_info.gain = emccd_gain


    def update_andor_config_settings(self, andor_config: AndorConfig):
        self.andor_camera.set_from_config_file(andor_config)
        self._andor_config = andor_config

    def update_scanning_config_file(self, axial_scan_config: ScanningConfig):
        self.axial_scan_config = axial_scan_config

    @contextmanager
    def force_reference_mode(self):
        was_sample_mode = not self.is_reference_mode
        if was_sample_mode:
            self.change_to_reference_mode()
        try:
            yield
        finally:
            if was_sample_mode:
                self.change_to_sample_mode()


    def simulated_delta_laser_corner_mm(self) -> float | None:
        """Dummy-only: the Δc [mm] the eye tracker WOULD report if it were
        consistent with the simulated cornea the reflection finder tracks.

        Returns None on real hardware (no simulated cornea), so callers can use
        it as a "am I in dummy mode with a coupled plane?" test. Sign follows
        EyeTrackerResults.delta_laser_corner: > 0 = laser in front of (below)
        the plane. The SIM_DELTAC_MARGIN_UM offset means a Move Z that nulls Δc
        parks the eye lens that many µm below the cornea, giving the forward
        reflection find clean room to approach — without it the plan's Move Z
        marches the lens onto/past the plane and the find fails."""
        cornea = getattr(self, "_sim_cornea", None)
        if cornea is None:
            return None
        t = time.perf_counter()
        plane_um = cornea.z_at(t) - SIM_DELTAC_MARGIN_UM
        lens_um = self.zaber_eye_lens.z_at(t)
        return (plane_um - lens_um) / 1000.0

    def find_reflection_plane(self, is_go_forwards: bool=True) -> ReflectionResult:
        """

        Args:

            is_go_forwards: True (forwards) False (backwards)

        Returns:

        """
        if self.is_reference_mode:
            log.info("System is in Reference (Calibration Mode) - Change to Sample Mode")
            return ReflectionResult(found=False)

        cfg = self.axial_scan_config
        self.homodyne_daq.configure(sample_rate_hz=cfg.homodyne_sample_rate_hz)
        params = HomodyneFinderParams(
            speed_um_s=cfg.homodyne_speed_um_s,
            max_distance_um=cfg.homodyne_max_distance_um,
            band_hz=(cfg.homodyne_band_lo_hz, cfg.homodyne_band_hi_hz),
            rms_window_s=cfg.homodyne_rms_window_s,
            n_sigma=cfg.homodyne_n_sigma,
            stop_after_um=cfg.homodyne_stop_after_um,
        )
        try:
            result = find_reflection_homodyne(
                self.homodyne_daq,
                self.zaber_eye_lens,
                params,
                direction=1.0 if is_go_forwards else -1.0,
                z_offset_um=cfg.z_offset_um,
                z_poll_s=cfg.z_poll_s,
                chunk_size=cfg.homodyne_chunk_size,
            )
        except Exception:
            log.exception("[Reflection finder] Search failed.")
            return ReflectionResult(found=False)
        if result.found:
            log.info(f"[Reflection finder] Plane at z = {result.event_z_um:.1f} um, S peak {result.peak_value:.3f}")
        else:
            log.warning("[Reflection finder] Plane NOT found.")
        return result


    def run_laser_xy_calibration(self) -> LaserOffset:
        """
        Run the full laser XY calibration from the backend and save offset.toml.

        Returns:
            LaserCoordSystem
        """
        if self.is_reference_mode:
            raise RuntimeError("Laser XY calibration must be run in sample mode, not reference mode.")

        log.info("[Laser XY Calibration] Starting.")

        calib = CalibRigLaserPosition(
            ni=self.ni,
            zaber_eye_lens=self.zaber_eye_lens,
            zaber_hi=self.zaber_hi,
            cancel_callback=self.f2b_cancel_callback,
            axial_scan_config=self.axial_scan_config,
            find_plane=self.find_reflection_plane,
        )

        try:
            laser_coord_system = calib.run_calibration()

            log.info(
                f"[Laser XY Calibration] Done. "
                f"dx={laser_coord_system.dx:.3f}, "
                f"dy={laser_coord_system.dy:.3f}, "
                f"dz={laser_coord_system.dz:.3f}"
            )
            return laser_coord_system

        except Exception as e:
            log.exception(f"[Laser XY Calibration] Failed: {e}")
            raise


    def close(self):
        """Cleanly shut down all backend-controlled devices."""
        print("Shutting down BrillouinBackend devices...")

        try:
            self.shutter_manager.close_all()
            print("Shutters closed.")
        except Exception as e:
            print(f"Error closing shutter manager: {e}")

        try:
            self.andor_camera.close()
            print("Andor camera closed.")
        except Exception as e:
            print(f"Error closing Andor camera: {e}")

        try:
            self.microwave.shutdown()
            print("Microwave shut down.")
        except Exception as e:
            print(f"Error shutting down microwave: {e}")

        try:
            self.zaber_eye_lens.close()
            print("Zaber controller closed.")
        except Exception as e:
            print(f"Error closing Zaber controller: {e}")


        print("BrillouinBackend shutdown complete.")

