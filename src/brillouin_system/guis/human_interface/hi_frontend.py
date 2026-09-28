import logging
import threading
import time

from brillouin_system.eye_tracker.calibrate_camera_laser_position.calib_rig_laser_position import LaserOffset, \
    load_laser_coord_system_from_toml
from brillouin_system.eye_tracker.eye_position.coordinates import RigCoord
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config_gui import EyeTrackerConfigDialog
from brillouin_system.eye_tracker.eye_tracker_results import get_eye_tracker_results, EyeTrackerResults
from brillouin_system.guis.human_interface.eye_tracker_controller import EyeTrackerController
from brillouin_system.scan_managers.scanning_config.scanning_config import ScanningConfig
from brillouin_system.scan_managers.scanning_config.scanning_config_gui import \
    AxialScanningConfigDialog
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config import SweepScanConfig, \
    sweep_scan_config
from brillouin_system.scan_managers.sweep_scan_config.sweep_scan_config_gui import \
    SweepScanConfigDialog

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

# Must be set before QApplication is constructed:
QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
# QtCore.QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


import sys
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget, TextItem

from PyQt5.QtGui import QDoubleValidator, QFont, QIntValidator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGroupBox, QLabel, QLineEdit,
    QFileDialog, QPushButton, QHBoxLayout, QFormLayout, QVBoxLayout, QCheckBox, QComboBox, QListWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5 import QtCore

from brillouin_system.logging_utils.qt_log_handler import QtLogBridge, QtTextEditHandler
from brillouin_system.logging_utils.logging_setup import start_logging, install_crash_hooks, get_logger, \
    logging_fmt_gui

log = get_logger(__name__)

from brillouin_system.calibration.config.calibration_config import CalibrationConfig
from brillouin_system.calibration.config.calibration_config_gui import CalibrationConfigDialog
from brillouin_system.devices.cameras.allied.allied_config.allied_config_dialog import AlliedConfigDialog
from brillouin_system.devices.cameras.andor.andor_frame.andor_config import AndorConfig
from brillouin_system.devices.cameras.andor.andor_frame.andor_config_dialog import AndorConfigDialog
from brillouin_system.guis.human_interface.hi_backend import HiBackend
from brillouin_system.guis.human_interface.hi_signaller import HiSignaller
from brillouin_system.guis.human_interface.predefined_plan import (
    PlanStep,
    ProgressReport,
    build_progress,
    expand_plan,
    parse_plan_toml,
)
from brillouin_system.my_dataclasses.axial_scan import AxialScan
from brillouin_system.my_dataclasses.request_axial_step_scan import RequestAxialStepScan
from brillouin_system.my_dataclasses.request_sweep_scan import RequestSweepScan
from brillouin_system.calibration.calibration import CalibrationData, CalibrationCalculator
from brillouin_system.calibration.calibration_plotting import render_calibration_to_pixmap, CalibrationImageDialog
###
# Add other guis
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import dataclass_to_hdf5_native_dict, save_dict_to_hdf5
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import FittingConfigs
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config_gui import FindPeaksConfigDialog

#todo: fix sample mode closing shutter.

use_backend_dummy = True
# Eye Tracking
include_eye_tracking = True
use_eye_tracker_dummy = True

# put this near your imports (top of file)

class NotifyingViewBox(pg.ViewBox):
    userScaled = QtCore.pyqtSignal()
    def wheelEvent(self, ev, axis=None):
        super().wheelEvent(ev, axis)
        self.userScaled.emit()
    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis)
        if ev.isFinish():
            self.userScaled.emit()


class PredefinedProgressDialog(QtWidgets.QDialog):
    """Live QC view of a predefined plan against the currently-saved scans.

    Purely a display of the ProgressReport the backend computed (no fitting,
    no VIPA images): the taken/remaining split, the successful-scan count, and
    a coverage plot of the real averaged positions so unfilled depths are
    obvious. Non-modal, so the operator can keep it open and re-open to refresh.
    """

    _GREEN = (46, 160, 67)
    _RED = (200, 60, 60)
    _AMBER = (200, 150, 40)
    _GREY = (140, 140, 140)

    def __init__(self, report: ProgressReport, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Predefined Measurement Progress")
        self.resize(720, 640)

        layout = QVBoxLayout(self)
        layout.addWidget(self._make_summary_label(report))
        layout.addWidget(self._make_table(report), 1)
        layout.addWidget(self._make_plot(report), 2)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _make_summary_label(self, report: ProgressReport) -> QLabel:
        taken, total = report.taken_count, report.planned_total
        fp, ft = report.frames_pass, report.frames_total
        remaining = report.remaining_ids()
        parts = [
            f"<b>Taken:</b> {taken}/{total} planned steps",
            f"<b>Frames within motion limit:</b> {fp}/{ft}",
        ]
        if remaining:
            shown = ", ".join(remaining[:12])
            more = "" if len(remaining) <= 12 else f" (+{len(remaining) - 12} more)"
            parts.append(f"<b>Still to take:</b> {shown}{more}")
        else:
            parts.append("<b>Still to take:</b> none — every planned step is taken.")
        if report.unmatched_scan_ids:
            parts.append(
                f"<i>Saved scans not in this plan (ignored): "
                f"{len(report.unmatched_scan_ids)}</i>")
        label = QLabel("<br>".join(parts))
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        return label

    def _make_table(self, report: ProgressReport) -> QtWidgets.QTableWidget:
        headers = ["Step", "Depth µm", "R mm", "φ°", "Limit µm",
                   "Frames in limit", "Avg depth µm", "min|Δ| µm"]
        table = QtWidgets.QTableWidget(len(report.rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        table.verticalHeader().setVisible(False)

        for r, row in enumerate(report.rows):
            # Per-frame status: green if every frame is within the limit, red if
            # none are, amber if partial, grey if the step has no data yet.
            if not row.taken or row.n_frames == 0:
                status = "not taken" if not row.taken else "no in/out frames"
                color = self._GREY
            else:
                status = f"{row.n_pass_frames}/{row.n_frames} frames"
                if row.n_pass_frames == row.n_frames:
                    color = self._GREEN
                elif row.n_pass_frames == 0:
                    color = self._RED
                else:
                    color = self._AMBER
            avg = "—" if row.avg_actual_depth_um is None else f"{row.avg_actual_depth_um:.1f}"
            mind = "—" if row.min_delta_um is None else f"{row.min_delta_um:.1f}"
            limit = "off" if row.motion_limit_um <= 0 else f"{row.motion_limit_um:g}"
            values = [row.id, f"{row.depth_um:g}", f"{row.R_mm:g}",
                      f"{row.phi_deg:g}", limit, status, avg, mind]
            for c, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 5:  # status column
                    item.setForeground(pg.mkColor(color))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        return table

    def _make_plot(self, report: ProgressReport) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setLabel("bottom", "Prescribed depth", units="µm")
        plot.setLabel("left", "Measured depth (fwd/bwd averaged)", units="µm")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.addLegend(offset=(-10, 10))

        # y = x identity line: a well-behaved measurement lands here. Spans the
        # union of every prescribed depth and every plotted value.
        vals = [p.prescribed_depth_um for p in report.points]
        vals += [p.actual_depth_um for p in report.points]
        vals += [r.depth_um for r in report.rows]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.03 * (hi - lo or 1.0)
            plot.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                      pen=pg.mkPen((150, 150, 150), width=1,
                                   style=Qt.DashLine),
                      name="y = x (ideal)")

        # One point per measured cycle: green pass, faint red fail.
        passed = [p for p in report.points if p.passed]
        failed = [p for p in report.points if not p.passed]
        if passed:
            plot.plot([p.prescribed_depth_um for p in passed],
                      [p.actual_depth_um for p in passed],
                      pen=None, symbol="o", symbolSize=8,
                      symbolBrush=self._GREEN, symbolPen=None,
                      name=f"pass ({len(passed)})")
        if failed:
            plot.plot([p.prescribed_depth_um for p in failed],
                      [p.actual_depth_um for p in failed],
                      pen=None, symbol="o", symbolSize=8,
                      symbolBrush=(*self._RED, 90), symbolPen=None,
                      name=f"fail ({len(failed)})")

        # Prescribed depths with no data yet: a faint tick on the identity line
        # marks where points still need to land.
        untaken = sorted({r.depth_um for r in report.rows if not r.taken})
        if untaken:
            plot.plot(untaken, untaken, pen=None, symbol="x", symbolSize=8,
                      symbolBrush=(*self._GREY, 90), symbolPen=(*self._GREY, 90),
                      name="still to fill")
        return plot


class HiFrontend(QWidget):

    # Signals Outgoing
    apply_camera_settings_requested = pyqtSignal(dict)
    toggle_camera_shutter_requested = pyqtSignal()
    emit_camera_settings_requested = pyqtSignal()
    start_live_requested = pyqtSignal()
    stop_live_requested = pyqtSignal()
    update_microwave_freq_requested = pyqtSignal(float)
    toggle_illumination_requested = pyqtSignal()
    toggle_reference_mode_requested = pyqtSignal()

    move_zaber_eye_lens_requested = pyqtSignal(float)
    move_zaber_stage_x_requested = pyqtSignal(float)
    move_zaber_stage_y_requested = pyqtSignal(float)
    move_zaber_stage_z_requested = pyqtSignal(float)

    run_calibration_requested = pyqtSignal()
    update_calibration_config_requested = pyqtSignal(object)
    take_axial_step_scan_requested = pyqtSignal(object)
    take_sweep_scan_requested = pyqtSignal(object)
    # Predefined-measurement progress: send the current plan (list[PlanStep])
    # to the backend thread, which joins it against the saved scans and emits
    # predefined_progress_ready.
    check_predefined_progress_requested = pyqtSignal(object)
    update_sweep_scan_config_requested = pyqtSignal(object)
    shutdown_requested = pyqtSignal()
    get_calibration_results_requested = pyqtSignal()
    toggle_do_live_fitting_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    update_andor_config_requested = pyqtSignal(object)
    close_all_shutters_requested = pyqtSignal()
    update_fitting_configs_requested = pyqtSignal(FittingConfigs)
    request_axial_scan_data = pyqtSignal(int)
    update_scanning_config_requested = pyqtSignal(object)
    take_bg_value_reflection_plane_request = pyqtSignal()
    find_reflection_plane_request = pyqtSignal()
    find_reflection_plane_backwards_request = pyqtSignal()
    load_ref_bkg_from_file_requested = pyqtSignal(str)
    load_ref_bkg_from_scan_requested = pyqtSignal(int)
    calibrate_laser_camera_position_requested = pyqtSignal()

    # Saving Signals
    save_all_axial_scans_requested = pyqtSignal()
    save_selected_axial_scans_requested = pyqtSignal(list)
    remove_selected_axial_scans_requested = pyqtSignal(list)

    # Eye Tracking Signals
    set_et_allied_configs = pyqtSignal(object, object)  # left_cfg, right_cfg
    request_eye_shutdown = pyqtSignal()
    set_et_config = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        # Attribute
        self.history_data = deque(maxlen=100)


        self.laser_focus_position: RigCoord | None = None
        self._zaber_lens_um: float | None = None
        self.lastest_eye_tracker_results: EyeTrackerResults | None = None
        if include_eye_tracking:
            self._laser_offset = load_laser_coord_system_from_toml()
        else:
            self._laser_offset = LaserOffset(0,0,0)
        self._last_eye_update_monotonic = 0.0
        self._andor_exposure_time: float | None = None

        # --- Create log_view early so logging can safely use it ---
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120)

        # --- Logging to GUI (Qt handler) ---
        self.log_bridge = QtLogBridge()
        self.log_bridge.message.connect(self._append_log_line)

        self.qt_handler = QtTextEditHandler(self.log_bridge)
        # Use the same format as your file logs for consistency
        self.qt_handler.setFormatter(logging_fmt_gui)

        lg = get_logger()
        if self.qt_handler not in lg.handlers:
            lg.addHandler(self.qt_handler)
        lg.setLevel(logging.INFO)


        self.setWindowTitle("Brillouin Viewer (Live)")

        self.brillouin_signaller = HiSignaller(backend=HiBackend(use_backend_dummy))
        self.brillouin_signaller_thread = QThread()
        self.brillouin_signaller.moveToThread(self.brillouin_signaller_thread)

        # Sending signals
        self.apply_camera_settings_requested.connect(self.brillouin_signaller.apply_camera_settings)
        self.emit_camera_settings_requested.connect(self.brillouin_signaller.emit_camera_settings)
        self.toggle_camera_shutter_requested.connect(self.brillouin_signaller.toggle_camera_shutter)
        self.start_live_requested.connect(self.brillouin_signaller.start_live_view)
        self.stop_live_requested.connect(self.brillouin_signaller.stop_live_view)
        self.update_microwave_freq_requested.connect(self.brillouin_signaller.set_microwave_frequency)
        self.toggle_illumination_requested.connect(self.brillouin_signaller.toggle_illumination_mode)
        self.toggle_reference_mode_requested.connect(self.brillouin_signaller.toggle_reference_mode)

        self.move_zaber_eye_lens_requested.connect(self.brillouin_signaller.move_zaber_eye_lens_relative)
        self.move_zaber_stage_x_requested.connect(self.brillouin_signaller.move_zaber_stage_x_relative)
        self.move_zaber_stage_y_requested.connect(self.brillouin_signaller.move_zaber_stage_y_relative)
        self.move_zaber_stage_z_requested.connect(self.brillouin_signaller.move_zaber_stage_z_relative)

        self.run_calibration_requested.connect(self.brillouin_signaller.run_calibration)
        self.update_calibration_config_requested.connect(self.brillouin_signaller.update_calibration_config_backend)
        self.take_axial_step_scan_requested.connect(self.brillouin_signaller.take_axial_step_scan)
        self.take_sweep_scan_requested.connect(self.brillouin_signaller.take_sweep_scan)
        self.check_predefined_progress_requested.connect(
            self.brillouin_signaller.compute_predefined_progress)
        self.update_sweep_scan_config_requested.connect(self.brillouin_signaller.update_sweep_scan_config)
        self.shutdown_requested.connect(self.brillouin_signaller.close)
        self.get_calibration_results_requested.connect(self.brillouin_signaller.get_calibration_results)
        self.toggle_do_live_fitting_requested.connect(self.brillouin_signaller.toggle_do_live_fitting)
        self.cancel_requested.connect(self.brillouin_signaller.cancel_operations)
        self.calibrate_laser_camera_position_requested.connect(
            self.brillouin_signaller.calibrate_laser_camera_position_delegate)

        self.update_andor_config_requested.connect(self.brillouin_signaller.update_andor_config_settings)
        self.close_all_shutters_requested.connect(self.brillouin_signaller.close_all_shutters)
        self.update_fitting_configs_requested.connect(self.brillouin_signaller.update_fitting_configs)
        self.request_axial_scan_data.connect(self.brillouin_signaller.handle_request_axial_scan_data)
        self.update_scanning_config_requested.connect(self.brillouin_signaller.update_scanning_config)
        self.find_reflection_plane_request.connect(self.brillouin_signaller.delegate_find_reflection_plane)
        self.find_reflection_plane_backwards_request.connect(self.brillouin_signaller.delegate_find_reflection_plane_backwards)
        self.load_ref_bkg_from_file_requested.connect(
            self.brillouin_signaller.handle_load_ref_bkg_from_file)
        self.load_ref_bkg_from_scan_requested.connect(
            self.brillouin_signaller.handle_load_ref_bkg_from_scan)

        # Receiving signals
        self.brillouin_signaller.calibration_finished.connect(self.calibration_finished)
        self.brillouin_signaller.illumination_mode_state.connect(self.update_illumination_ui)
        self.brillouin_signaller.reference_mode_state.connect(self.update_reference_ui)
        self.brillouin_signaller.camera_settings_ready.connect(self.populate_camera_ui)
        self.brillouin_signaller.camera_shutter_state_changed.connect(self.update_camera_shutter_button)
        self.brillouin_signaller.new_andor_display_ready.connect(self._on_andor_mailbox, Qt.QueuedConnection)

        self.brillouin_signaller.zaber_lens_position_updated.connect(self.update_zaber_lens_position)
        self.brillouin_signaller.zaber_stage_positions_updated.connect(self.update_stage_positions)

        self.brillouin_signaller.microwave_frequency_updated.connect(self.update_ref_freq_input)

        self.brillouin_signaller.calibration_result_ready.connect(self.handle_requested_calibration)
        self.brillouin_signaller.do_live_fitting_state.connect(self.update_do_live_fitting_checkbox)
        self.brillouin_signaller.ref_bkg_state.connect(self.update_ref_bkg_label)

        self.brillouin_signaller.update_system_state_in_frontend.connect(self.update_system_state_label)
        self.brillouin_signaller.send_update_stored_axial_scans.connect(self.receive_axial_scan_list)
        self.brillouin_signaller.sweep_scan_started.connect(self.on_sweep_scan_started)
        self.brillouin_signaller.sweep_scan_finished.connect(self.on_sweep_scan_finished)
        self.brillouin_signaller.predefined_progress_ready.connect(
            self.show_predefined_progress)
        self.brillouin_signaller.axial_scan_data_ready.connect(self.handle_received_axial_scan_data)
        self.brillouin_signaller.send_axial_scans_to_save.connect(self.save_axial_scan_list_to_file)
        self.brillouin_signaller.send_message_to_frontend.connect(self.message_handler)
        self.brillouin_signaller.close_event_finished.connect(self._finalize_close)
        self.brillouin_signaller.laser_coord_calibration_ready.connect(self.update_laser_coord_calibration)

        # Saving Signals
        self.save_all_axial_scans_requested.connect(self.brillouin_signaller.save_all_axial_scans)
        self.save_selected_axial_scans_requested.connect(self.brillouin_signaller.save_multiple_axial_scans)
        self.remove_selected_axial_scans_requested.connect(self.brillouin_signaller.remove_selected_axial_scans)


        # Connect signals BEFORE starting the thread
        self.brillouin_signaller_thread.started.connect(self.run_gui)

        # Start the thread after all connections
        self.brillouin_signaller_thread.start()

        # Build UI (reuses self.log_view created above)
        self.init_ui()

        self.update_gui()

        # --- Eye tracker thread & controller ---
        if include_eye_tracking:
            self.start_eye_tracker()

    def start_eye_tracker(self):
        self.eye_thread = QThread(self)
        self.eye_ctrl = EyeTrackerController(use_dummy=use_eye_tracker_dummy)  # or False
        self.eye_ctrl.moveToThread(self.eye_thread)

        # start/stop/shutdown
        self.eye_thread.started.connect(self.eye_ctrl.start)

        # frames into GUI
        self.eye_ctrl.frames_ready.connect(self.on_eye_frames_ready)

        # Sending signals
        self.set_et_allied_configs.connect(self.eye_ctrl.proxy.set_allied_configs)
        self.set_et_config.connect(self.eye_ctrl.send_config)
        self.request_eye_shutdown.connect(self.eye_ctrl.shutdown)

        self.eye_thread.start()

    def _append_log_line(self, line: str):
        self.log_view.append(line)

    def init_ui(self):
        outer_layout = QHBoxLayout()
        self.setLayout(outer_layout)

        # LEFT COLUMN: Controls
        left_column_layout = QVBoxLayout()
        left_column_layout.addWidget(self.create_control_group())
        left_column_layout.addWidget(self.create_andor_camera_group())
        left_column_layout.addWidget(self.create_fitting_group())
        left_column_layout.addWidget(self.create_reference_group())
        left_column_layout.addWidget(self.create_illumination_group())
        left_column_layout.addWidget(self.create_allied_vision_group())
        left_column_layout.addWidget(self.create_axial_scans_group())
        left_column_layout.addWidget(self.create_show_scan_results())
        left_column_layout.addStretch()
        outer_layout.addLayout(left_column_layout, 0)

        # MID COLUMN: Plots + log view
        middle_column_layout = QVBoxLayout()
        middle_column_layout.addWidget(self.create_andor_display_group())

        zaber_movement_and_scan_layout = QHBoxLayout()
        first_column = QVBoxLayout()

        first_column.addWidget(self.create_zaber_manual_movement_group())
        first_column.addWidget(self.create_axial_scan_settings_group())
        zaber_movement_and_scan_layout.addLayout(first_column)

        axial_column = QVBoxLayout()
        axial_column.addWidget(self.create_take_axial_scan_group())
        axial_column.addWidget(self.create_predefined_measurement_group())

        zaber_movement_and_scan_layout.addLayout(axial_column)
        middle_column_layout.addLayout(zaber_movement_and_scan_layout)

        # log_view was created in __init__; just add it here
        # middle_column_layout.addWidget(self.log_view)

        outer_layout.addLayout(middle_column_layout, 1)

        # FAR RIGHT COLUMN: Eye Tracking
        right_column_layout = QVBoxLayout()
        if include_eye_tracking:
            right_column_layout.addWidget(self.create_eye_tracking_group())
            right_column_layout.addStretch()
            outer_layout.addLayout(right_column_layout, 0)

        right_column_layout.addWidget(self.create_log_group())
        right_column_layout.addStretch()
        outer_layout.addLayout(right_column_layout, 0)

    # ---------------- UI Sections ---------------- #

    def create_control_group(self):
        group = QGroupBox("Control")
        layout = QHBoxLayout()

        stop_btn = QPushButton("STOP")
        cancel_btn = QPushButton("Cancel")
        restart_btn = QPushButton("Restart")

        self.state_label = QLabel("● IDLE")
        self.state_label.setStyleSheet("color: gray; font-weight: bold")

        stop_btn.clicked.connect(self.on_stop_clicked)
        cancel_btn.clicked.connect(self.on_cancel_event_clicked)
        restart_btn.clicked.connect(self.on_restart_clicked)

        layout.addWidget(stop_btn)
        layout.addWidget(cancel_btn)
        layout.addWidget(restart_btn)
        layout.addWidget(self.state_label)
        group.setLayout(layout)
        return group

    def create_andor_camera_group(self):
        self.exposure_input = QLineEdit()
        self.exposure_input.setValidator(QDoubleValidator(0.001, 60.0, 3))

        self.gain_input = QLineEdit()
        self.gain_input.setValidator(QIntValidator(0, 1000))

        self.config_camera_btn = QPushButton("Settings")
        self.config_camera_btn.clicked.connect(self.on_andor_configs_clicked)

        self.toggle_camera_shutter_btn = QPushButton("Close")
        self.toggle_camera_shutter_btn.clicked.connect(self.toggle_camera_shutter_requested.emit)

        self.apply_camera_btn = QPushButton("Apply")
        self.apply_camera_btn.clicked.connect(self.apply_camera_settings)

        # Horizontal layout for the buttons
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.config_camera_btn)
        btn_row.addWidget(self.toggle_camera_shutter_btn)
        btn_row.addWidget(self.apply_camera_btn)

        # Main layout
        layout = QFormLayout()
        layout.addRow("Exp. Time (s):", self.exposure_input)
        layout.addRow("Gain:", self.gain_input)
        layout.addRow(btn_row)

        group = QGroupBox("Andor Camera")
        group.setLayout(layout)

        return group

    def create_fitting_group(self):
        self.fitting_config_btn = QPushButton("Config")
        self.fitting_config_btn.clicked.connect(self.on_fitting_configs_clicked)

        self.do_live_fitting_checkbox = QCheckBox("Do Live Sample Fitting")
        self.do_live_fitting_checkbox.stateChanged.connect(self.on_do_live_fitting_toggled)

        # Horizontal layout for button + checkbox
        row_layout = QHBoxLayout()
        row_layout.addWidget(self.fitting_config_btn)
        row_layout.addWidget(self.do_live_fitting_checkbox)

        # Reflection background for live sample fits: without one loaded,
        # a 'reflection' sample config warns once and fits per-peak flat
        # offsets only. 'Take Ref. Bkg.' auto-loads its scan when done.
        self.load_ref_bkg_btn = QPushButton("Load Ref. Bkg.")
        self.load_ref_bkg_btn.setToolTip(
            "Load a reflection background for live sample fitting — a "
            "saved .npz template, or an .h5 scan (built the same way the "
            "analyzer's Load Background does).")
        self.load_ref_bkg_btn.clicked.connect(self.on_load_ref_bkg_clicked)

        self.ref_bkg_label = QLabel("Ref. bkg: none")
        self.ref_bkg_label.setStyleSheet("color: gray")

        bkg_row = QHBoxLayout()
        bkg_row.addWidget(self.load_ref_bkg_btn)
        bkg_row.addWidget(self.ref_bkg_label)
        bkg_row.addStretch()

        # Vertical layout for the group box
        layout = QVBoxLayout()
        layout.addLayout(row_layout)
        layout.addLayout(bkg_row)

        group = QGroupBox("Fitting")
        group.setLayout(layout)

        return group

    def create_illumination_group(self):
        self.illum_label_cont = QLabel("● Open")
        self.illum_label_pulse = QLabel("○ Closed")

        self.illum_label_cont.setStyleSheet("color: gray")
        self.illum_label_pulse.setStyleSheet("color: gray")

        self.toggle_illum_btn = QPushButton("Switch")
        self.toggle_illum_btn.clicked.connect(self.toggle_illumination)


        # Row 1: Continuous label and toggle button
        row1 = QHBoxLayout()
        row1.addWidget(self.illum_label_cont)
        row1.addWidget(self.toggle_illum_btn)
        row1.addStretch()

        # Row 2: Pulsed label and snap button
        row2 = QHBoxLayout()
        row2.addWidget(self.illum_label_pulse)
        row2.addStretch()

        # Combine rows vertically
        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)

        group = QGroupBox("Illumination")
        group.setLayout(layout)
        return group

    def create_reference_group(self):
        # Mode labels
        self.calib_label_meas = QLabel("● Meas.")
        self.calib_label_calib = QLabel("○ Ref.")
        self.calib_label_meas.setStyleSheet("color: green; font-weight: bold")
        self.calib_label_calib.setStyleSheet("color: gray")

        self.toggle_calib_btn = QPushButton("Switch")
        self.toggle_calib_btn.setFixedWidth(60)
        self.toggle_calib_btn.clicked.connect(self.toggle_reference_mode)

        mode_column = QVBoxLayout()
        mode_column.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        mode_column.addWidget(self.calib_label_meas, alignment=Qt.AlignHCenter)
        mode_column.addWidget(self.calib_label_calib, alignment=Qt.AlignHCenter)
        mode_column.addSpacing(4)
        mode_column.addWidget(self.toggle_calib_btn, alignment=Qt.AlignHCenter)

        # Frequency row
        self.ref_freq_input = QLineEdit()
        self.ref_freq_input.setFixedWidth(50)
        self.ref_freq_input.setValidator(QDoubleValidator(0.0, 100.0, 4))
        self.ref_freq_input.setText("5.5")

        self.set_ref_btn = QPushButton("Set")
        self.set_ref_btn.setFixedWidth(40)
        self.set_ref_btn.clicked.connect(self.set_reference_freq)

        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Ref. Freq. (GHz):"))
        freq_row.addWidget(self.ref_freq_input)
        freq_row.addWidget(self.set_ref_btn)
        freq_row.addStretch()

        # Config + Cal. all
        self.config_ref_btn = QPushButton("Config")
        self.config_ref_btn.setFixedWidth(60)
        self.config_ref_btn.clicked.connect(self.on_reference_configs_clicked)

        self.calibrate_btn = QPushButton("Cal. all")
        self.calibrate_btn.setFixedWidth(70)
        self.calibrate_btn.clicked.connect(self.run_calibration)

        config_row = QHBoxLayout()
        config_row.addWidget(self.config_ref_btn)
        config_row.addWidget(self.calibrate_btn)
        config_row.addStretch()

        # Show + Save
        self.show_calib_btn = QPushButton("Show")
        self.show_calib_btn.setFixedWidth(60)
        self.show_calib_btn.clicked.connect(self.show_calibration_results)
        self.show_calib_btn.setEnabled(False)

        self.save_calib_btn = QPushButton("Save")
        self.save_calib_btn.setFixedWidth(60)
        self.save_calib_btn.clicked.connect(self.save_calibration_results)
        self.save_calib_btn.setEnabled(False)

        save_row = QHBoxLayout()
        save_row.addWidget(self.show_calib_btn)
        save_row.addWidget(self.save_calib_btn)
        save_row.addStretch()

        # Right side layout (all rows stacked)
        right_column = QVBoxLayout()
        right_column.setAlignment(Qt.AlignTop)
        right_column.addLayout(freq_row)
        right_column.addLayout(config_row)
        right_column.addLayout(save_row)

        # Combine left and right in fixed container layout
        content_layout = QHBoxLayout()
        content_layout.setAlignment(Qt.AlignLeft)
        content_layout.addLayout(mode_column)
        content_layout.addSpacing(15)
        content_layout.addLayout(right_column)

        # Wrap in main vertical layout to avoid stretching
        outer_layout = QVBoxLayout()
        outer_layout.setAlignment(Qt.AlignTop)
        outer_layout.addLayout(content_layout)

        group = QGroupBox("Reference")
        group.setLayout(outer_layout)
        return group

    def create_allied_vision_group(self):
        group = QGroupBox("Eye Tracker")

        # Buttons row for Allied Vision Cameras
        btn_row = QHBoxLayout()
        self.btn_allied_left = QPushButton("Left")
        self.btn_allied_right = QPushButton("Right")
        self.btn_eye_tracking = QPushButton("Config")

        btn_row.addWidget(self.btn_allied_left)
        btn_row.addWidget(self.btn_allied_right)
        btn_row.addWidget(self.btn_eye_tracking)
        btn_row.addStretch()

        btn_row2 = QHBoxLayout()
        self.btn_restart_eye = QPushButton("ReStart")
        self.btn_close_eye = QPushButton("Shutdown")
        self.btn_cal_laser_offset = QPushButton("Cal. Laser Offset")

        btn_row2.addWidget(self.btn_restart_eye)
        btn_row2.addWidget(self.btn_close_eye)
        btn_row2.addWidget(self.btn_cal_laser_offset)
        btn_row2.addStretch()

        # put the row into a real layout and set it on the group
        v = QVBoxLayout()
        v.addLayout(btn_row)
        v.addLayout(btn_row2)
        group.setLayout(v)

        # wire up
        self.btn_allied_left.clicked.connect(self.open_allied_left_dialog)
        self.btn_allied_right.clicked.connect(self.open_allied_right_dialog)
        self.btn_eye_tracking.clicked.connect(self.open_eye_tracker_config_dialog)
        self.btn_restart_eye.clicked.connect(self.on_restart_eye_clicked)
        self.btn_close_eye.clicked.connect(self.shutdown_eye_tracker)
        self.btn_cal_laser_offset.clicked.connect(self.calibrate_laser_position)

        return group

    def create_axial_scan_settings_group(self):
        group = QGroupBox("Axial Scan Settings and Reflection Plane Finding")
        layout = QFormLayout()

        self.axial_settings_btn = QPushButton("Settings")
        self.axial_settings_btn.clicked.connect(self.open_axial_scan_settings_dialog)

        self.find_reflection_plane_btn = QPushButton("Find Reflection Plane")
        self.find_reflection_plane_btn.clicked.connect(self.find_reflection_plane)

        self.find_reflection_plane_backwards_btn = QPushButton("Find Reflection Plane (Backwards)")
        self.find_reflection_plane_backwards_btn.clicked.connect(self.find_reflection_plane_backwards)


        btn_row = QHBoxLayout()
        btn_row.addWidget(self.axial_settings_btn)
        # btn_row.addWidget(self.take_bg_value_btn)
        btn_row.addWidget(self.find_reflection_plane_btn)
        btn_row.addWidget(self.find_reflection_plane_backwards_btn)

        btn_row.addStretch()

        layout.addRow(btn_row)

        group.setLayout(layout)
        return group

    def create_take_axial_scan_group(self):
        group = QGroupBox("Scan Axial Steps")
        layout = QFormLayout()

        # --- Axial scan (step mode) controls ---
        self.axial_id_input = QLineEdit()
        self.axial_id_input.setFixedWidth(80)
        self.axial_num_input = QLineEdit("10")
        self.axial_num_input.setFixedWidth(80)
        self.axial_num_input.textChanged.connect(self.update_axial_step_distance)
        self.axial_step_input = QLineEdit("10")
        self.axial_step_input.setFixedWidth(80)
        self.axial_step_input.textChanged.connect(self.update_axial_step_distance)

        self.axial_steps_scanned_dist_label = QLabel("100.00 µm")
        self.axial_steps_scanned_dist_label.setFixedWidth(80)


        layout.addRow("ID:", self.axial_id_input)
        layout.addRow("Num Meas:", self.axial_num_input)
        layout.addRow("Step Size (µm):", self.axial_step_input)
        layout.addRow("Scanned Dist (µm):", self.axial_steps_scanned_dist_label)

        self.axial_btn = QPushButton("Scan")
        self.axial_btn.clicked.connect(lambda: self.take_axial_step_scan(find_reflection_plane=False))

        self.axial_random_btn = QPushButton("Random Scan")
        self.axial_random_btn.setToolTip(
            "Same positions as Scan (current position + k x Step Size, "
            "k = 1..Num Meas), visited in shuffled order. Each position is "
            "approached from 100 µm below so backlash is always taken up the "
            "same way. Frames are stored in acquisition order with their lens "
            "positions; the seed is written to the log.")
        self.axial_random_btn.clicked.connect(
            lambda: self.take_axial_step_scan(find_reflection_plane=False, randomize_order=True))

        self.axial_btn2 = QPushButton("Find -> Scan")
        self.axial_btn2.clicked.connect(lambda: self.take_axial_step_scan(find_reflection_plane=True))

        self.take_background_btn = QPushButton("Take Ref. Bkg.")
        self.take_background_btn.setToolTip(
            "Records Num Meas frames at the CURRENT position through the "
            "normal scan pipeline (step 0), named 'reflection_background', "
            "stored and saved like any scan. Position at the reflection "
            "plane first. When the scan completes it is AUTOMATICALLY "
            "loaded as the live reflection background for sample fitting "
            "(see the Fitting group's status label).")
        self.take_background_btn.clicked.connect(self.take_background_scan)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.axial_btn)
        btn_row.addWidget(self.axial_random_btn)
        btn_row.addWidget(self.axial_btn2)
        btn_row.addWidget(self.take_background_btn)
        btn_row.addStretch()

        layout.addRow(btn_row)

        # --- In-out sweep scan (repeated find-measure-find cycles) ---
        self.sweep_scan_btn = QPushButton("Sweep Scan")
        # Wrap in a lambda: QPushButton.clicked passes a `checked` bool that
        # would otherwise land in take_sweep_scan's max_time_s argument.
        self.sweep_scan_btn.clicked.connect(lambda: self.take_sweep_scan())

        self.timed_sweep_scan_btn = QPushButton("Timed Sweep")
        self.timed_sweep_scan_btn.setToolTip(
            "Run as many sweep cycles as fit in the Sweep Settings 'Max time' "
            "budget, instead of a fixed number of cycles.")
        self.timed_sweep_scan_btn.clicked.connect(lambda: self.take_timed_sweep_scan())

        self.sweep_scan_settings_btn = QPushButton("Sweep Settings")
        self.sweep_scan_settings_btn.clicked.connect(self.open_sweep_scan_settings_dialog)

        self.end_scan_early_btn = QPushButton("End Scan Early")
        self.end_scan_early_btn.setToolTip(
            "Stop the running sweep scan after the current cycle, but keep and "
            "save the cycles already recorded (counts as a successful scan).")
        self.end_scan_early_btn.clicked.connect(self.on_end_scan_early_clicked)

        sweep_row = QHBoxLayout()
        sweep_row.addWidget(self.sweep_scan_btn)
        sweep_row.addWidget(self.timed_sweep_scan_btn)
        sweep_row.addWidget(self.sweep_scan_settings_btn)
        sweep_row.addWidget(self.end_scan_early_btn)
        sweep_row.addStretch()

        layout.addRow(sweep_row)

        # --- Sweep timing: last + longest + mean-on-record (resettable) ---
        self._sweep_longest_seconds = None
        self._sweep_total_seconds = 0.0
        self._sweep_count = 0

        self.sweep_last_label = QLabel("Last sweep: —")
        self.sweep_longest_label = QLabel("Longest sweep: —")
        self.sweep_mean_label = QLabel("Mean sweep: —")

        self.sweep_longest_reset_btn = QPushButton("Reset")
        self.sweep_longest_reset_btn.setToolTip(
            "Reset the longest and mean sweep scan duration records.")
        self.sweep_longest_reset_btn.clicked.connect(self.reset_sweep_stats)

        sweep_timing_row = QHBoxLayout()
        sweep_timing_row.addWidget(self.sweep_last_label)
        sweep_timing_row.addWidget(self.sweep_longest_label)
        sweep_timing_row.addWidget(self.sweep_mean_label)
        sweep_timing_row.addWidget(self.sweep_longest_reset_btn)
        sweep_timing_row.addStretch()

        layout.addRow(sweep_timing_row)

        group.setLayout(layout)
        return group

    # ================================================================
    # Predefined measurement (TOML plan -> sweep-scan sequence)
    # ----------------------------------------------------------------
    # A plan is a TOML file: one [defaults] table applied to every step, plus
    # an array of [[step]] tables that override only what differs (see
    # measurement_templates/example_measurement_plan.toml and predefined_plan.py
    # for the full schema). Each step carries:
    #
    #   depth_um         target depth past the plane [µm] -> SweepScanConfig.target_depth_um
    #   mode             "timed" (cycles bounded by max_time_s) or "fixed"
    #   cycles           number of in-out cycles          -> n_repeats (fixed only)
    #   max_time_s       max sweep time [s]               -> SweepScanConfig.max_time_s
    #   replicates       number of sweep scans at this depth
    #   R_mm             target laser radius [mm]         -> Move XY
    #   phi_deg          target laser angle  [deg]        -> Move XY
    #   delta_c_mm       target Δc [mm]                    -> Move Z
    #   motion_limit_um  motion quality gate [µm] (0 = off)
    #
    # Each step expands into `replicates` sweep scans, IDs "depth<depth_um>num<k>".
    # Running a step: set the R/phi/Δc fields, Move XYZ, push the parameters
    # into the sweep config (preserving every other tuned field), name the ID,
    # then take the (timed or fixed) sweep scan carrying the motion limit — a
    # scan that fails the limit is dropped like any failed scan (see
    # scan_procedures.take_sweep_scan). All by reusing the existing handlers.
    # ================================================================

    def create_predefined_measurement_group(self):
        group = QGroupBox("Predefined Measurement")
        layout = QVBoxLayout()

        # Source plan entries and the expanded (possibly scrambled) steps.
        self._predef_entries: list = []
        self._predef_steps: list[PlanStep] = []
        self._predef_index: int = 0
        self._predef_combo_updating: bool = False
        # True while a predefined step's sweep is in flight; gates re-entry and
        # tells on_sweep_scan_finished whether to advance the list.
        self._predef_run_active: bool = False

        # Row: import + scramble
        self.predef_import_btn = QPushButton("Import Plan…")
        self.predef_import_btn.setToolTip(
            "Load a .toml measurement plan. A [defaults] table applies to every\n"
            "step; each [[step]] overrides only what differs. Fields:\n"
            "  depth_um, mode(\"timed\"/\"fixed\"), cycles, max_time_s,\n"
            "  replicates, R_mm, phi_deg, delta_c_mm, motion_limit_um\n"
            "See measurement_templates/example_measurement_plan.toml.")
        self.predef_import_btn.clicked.connect(self.on_import_predefined_template)

        self.predef_scramble_btn = QPushButton("Scramble")
        self.predef_scramble_btn.setToolTip(
            "Randomise the order of depths and replicates. Each depth's "
            "replicate numbers still increase monotonically (num1 before "
            "num2, …).")
        self.predef_scramble_btn.clicked.connect(self.on_scramble_predefined)

        io_row = QHBoxLayout()
        io_row.addWidget(self.predef_import_btn)
        io_row.addWidget(self.predef_scramble_btn)
        io_row.addStretch()
        layout.addLayout(io_row)

        # Row: navigation (prev / dropdown / next)
        self.predef_prev_btn = QPushButton("◀")
        self.predef_prev_btn.setFixedWidth(34)
        self.predef_prev_btn.setToolTip("Previous step")
        self.predef_prev_btn.clicked.connect(self.on_predefined_prev)

        self.predef_combo = QComboBox()
        self.predef_combo.currentIndexChanged.connect(self.on_predefined_combo_changed)

        self.predef_next_btn = QPushButton("▶")
        self.predef_next_btn.setFixedWidth(34)
        self.predef_next_btn.setToolTip("Next step")
        self.predef_next_btn.clicked.connect(self.on_predefined_next)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.predef_prev_btn)
        nav_row.addWidget(self.predef_combo, 1)
        nav_row.addWidget(self.predef_next_btn)
        layout.addLayout(nav_row)

        # Row: current-step detail
        self.predef_detail_label = QLabel("No plan loaded.")
        self.predef_detail_label.setWordWrap(True)
        layout.addWidget(self.predef_detail_label)

        # Row: take the current step
        self.predef_take_btn = QPushButton("Take Predefined Measurement")
        self.predef_take_btn.setToolTip(
            "Run the selected step: Move XYZ, set the sweep X/Y, then take "
            "the sweep scan. The selection then advances to the next step.")
        self.predef_take_btn.clicked.connect(self.take_predefined_measurement)
        layout.addWidget(self.predef_take_btn)

        # Row: quality-control progress
        self.predef_progress_btn = QPushButton("Check Progress")
        self.predef_progress_btn.setToolTip(
            "Fast QC pass over the saved scans (no VIPA images, no spectrum "
            "fitting): which planned steps are done vs still to take, how many "
            "cleared the motion limit, and a coverage plot of the real "
            "averaged positions.")
        self.predef_progress_btn.clicked.connect(self.on_check_predefined_progress)
        layout.addWidget(self.predef_progress_btn)

        group.setLayout(layout)
        self._refresh_predefined_ui()
        return group

    # ---- plan parsing / step building (see predefined_plan.py) ----

    # ---- UI refresh / navigation ----

    def _refresh_predefined_ui(self):
        """Rebuild the dropdown and detail label from the current steps."""
        has_steps = bool(self._predef_steps)
        total = len(self._predef_steps)
        if total:
            self._predef_index = max(0, min(self._predef_index, total - 1))

        self._predef_combo_updating = True
        self.predef_combo.clear()
        for i, step in enumerate(self._predef_steps):
            self.predef_combo.addItem(f"{i + 1}. {step.id}")
        if has_steps:
            self.predef_combo.setCurrentIndex(self._predef_index)
        self._predef_combo_updating = False

        for w in (self.predef_combo, self.predef_prev_btn, self.predef_next_btn,
                  self.predef_take_btn, self.predef_scramble_btn):
            w.setEnabled(has_steps)

        if not has_steps:
            self.predef_detail_label.setText("No plan loaded.")
            return

        step = self._predef_steps[self._predef_index]
        limit_txt = (f"{step.motion_limit_um:g} µm"
                     if step.motion_limit_um > 0 else "off")
        self.predef_detail_label.setText(
            f"Step {self._predef_index + 1}/{total} — {step.id}\n"
            f"depth={step.depth_um:g} µm, {step.mode_str()} | "
            f"R={step.R_mm:g} mm, phi={step.phi_deg:g}°, "
            f"Δc={step.delta_c_mm:g} mm | motion limit={limit_txt}")

    def _set_predef_index(self, index: int):
        if not self._predef_steps:
            return
        self._predef_index = max(0, min(index, len(self._predef_steps) - 1))
        self._refresh_predefined_ui()

    def on_predefined_combo_changed(self, index: int):
        if self._predef_combo_updating or index < 0:
            return
        self._set_predef_index(index)

    def on_predefined_prev(self):
        self._set_predef_index(self._predef_index - 1)

    def on_predefined_next(self):
        self._set_predef_index(self._predef_index + 1)

    @staticmethod
    def _measurement_templates_dir() -> str:
        """Default folder for measurement templates:
        src/brillouin_system/measurement_templates."""
        from pathlib import Path
        d = Path(__file__).resolve().parents[2] / "measurement_templates"
        return str(d) if d.is_dir() else ""

    def on_import_predefined_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Measurement Plan", self._measurement_templates_dir(),
            "Measurement plan (*.toml);;All files (*)")
        if not path:
            return
        try:
            entries = parse_plan_toml(path)
        except Exception as e:
            log.exception("[Predefined] Failed to parse plan.")
            QMessageBox.critical(self, "Plan Parse Error", str(e))
            return

        self._predef_entries = entries
        self._predef_steps = expand_plan(entries, scramble=False)
        self._predef_index = 0
        self._refresh_predefined_ui()
        log.info(f"[Predefined] Loaded {len(entries)} depth(s) -> "
                 f"{len(self._predef_steps)} step(s) from {path}")

    def on_scramble_predefined(self):
        if not self._predef_entries:
            return
        self._predef_steps = expand_plan(self._predef_entries, scramble=True)
        self._predef_index = 0
        self._refresh_predefined_ui()
        order = ", ".join(s.id for s in self._predef_steps)
        log.info(f"[Predefined] Scrambled order: {order}")

    def take_predefined_measurement(self):
        """Run the currently-selected step. The selection advances to the
        next step only once the sweep actually completes (see
        on_sweep_scan_finished); a sweep that fails to start or is cancelled
        leaves the selection here so it can be retried."""
        if not self._predef_steps:
            log.warning("[Predefined] No plan loaded.")
            return

        if getattr(self, "_predef_run_active", False):
            log.warning("[Predefined] A step is already running; wait for it "
                        "to finish before starting another.")
            return

        step = self._predef_steps[self._predef_index]
        step_id = step.id
        log.info(f"[Predefined] Running step {self._predef_index + 1}/"
                 f"{len(self._predef_steps)} | ID: {step_id}")

        # 1) Position: Move XYZ using this step's R / phi / Δc.
        self.xy_r_input.setText(f"{step.R_mm:g}")
        self.xy_phi_input.setText(f"{step.phi_deg:g}")
        self.dc_input.setText(f"{step.delta_c_mm:g}")
        self.on_move_xyz_clicked()

        # 2) Push this step's parameters into the sweep config, preserving
        #    every other tuned field. Always set the target depth. A fixed
        #    step also sets n_repeats; a timed step leaves n_repeats alone
        #    (unused). Both kinds set max_time_s from the plan.
        timed = bool(step.timed)
        overrides = {"target_depth_um": float(step.depth_um),
                     "max_time_s": float(step.max_time_s)}
        if not timed:
            overrides["n_repeats"] = int(step.cycles)
        try:
            from dataclasses import replace
            cfg = replace(sweep_scan_config.get(), **overrides)
        except Exception:
            log.exception("[Predefined] Could not clone current sweep config; "
                          "falling back to defaults for this step.")
            cfg = SweepScanConfig(**overrides)
        self.update_sweep_scan_config_requested.emit(cfg)

        # 3) Name the ID and take the sweep scan (reuses take_sweep_scan()).
        #    Mark the run active so on_sweep_scan_finished advances the list
        #    only if this sweep actually completes.
        self._predef_run_active = True
        self.axial_id_input.setText(step_id)
        # Use this step's resolved max time for the budget / auto end-scan-early
        # timer, run timed or fixed as the plan specified, and carry the motion
        # limit so the backend drops the scan if the eye moved too much.
        if not self.take_sweep_scan(max_time_s=cfg.max_time_s, timed=timed,
                                    motion_limit_um=step.motion_limit_um):
            # The request never went out; clear the flag so it can be retried
            # (no completion signal will arrive to clear it otherwise).
            self._predef_run_active = False

    def on_check_predefined_progress(self):
        """Ask the backend to score the current plan against the saved scans.

        Fast by design: it reads only each scan's stored reflection crossings
        and lens positions (no VIPA images, no spectrum fitting). The result
        comes back via predefined_progress_ready -> show_predefined_progress."""
        if not self._predef_steps:
            QMessageBox.information(
                self, "Check Progress",
                "No plan loaded. Import a measurement plan first.")
            return
        self.check_predefined_progress_requested.emit(list(self._predef_steps))

    def show_predefined_progress(self, report: ProgressReport):
        """Display the progress report (non-modal so it can stay open)."""
        dialog = PredefinedProgressDialog(report, parent=self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()

    def update_axial_step_distance(self):
        """
        Compute Scanned Dist (µm) = #steps-1 () * distance (um)
        and update the label in the Axial Scan Cont. group.
        """
        # If the continuous scan widgets aren't created yet, just ignore
        try:
            # Speed (µm/s) from Axial Scan Cont.
            N = self.axial_num_input.text().strip()
            N = int(N)

            step = self.axial_step_input.text().strip()
            step = float(step)

            dist_um = N * step
            self.axial_steps_scanned_dist_label.setText(f"{dist_um:.2f} µm")
        except Exception:
            # If parsing fails, show a neutral placeholder
            self.axial_steps_scanned_dist_label.setText("—")



    def create_log_group(self):
        group = QGroupBox("Log Output")
        layout = QVBoxLayout()
        layout.addWidget(self.log_view)
        group.setLayout(layout)
        return group

    # --- Eye Tracking UI ---

    def create_eye_tracking_group(self):
        group = QGroupBox("Eye Tracking")

        self.eye_glw = GraphicsLayoutWidget()
        self.eye_glw.ci.setContentsMargins(4, 4, 4, 4)
        self.eye_glw.ci.setSpacing(4)

        self.eye_vb = []
        self.eye_img = []

        # NEW: store bottom-row plots
        self.eye_bottom_plots = []

        for i in range(6):
            if i in (2, 4):
                self.eye_glw.ci.nextRow()

            row = i // 2
            col = i % 2

            # Top 2 rows (0..3): keep your existing "image in a ViewBox"
            if i < 4:
                vb = pg.ViewBox(lockAspect=True, enableMenu=False)
                vb.invertY(True)
                img = pg.ImageItem(autoDownsample=True)
                vb.addItem(img)
                vb.setBorder((80, 80, 80))

                self.eye_glw.ci.addItem(vb, row=row, col=col)
                self.eye_vb.append(vb)
                self.eye_img.append(img)

            # Bottom row (4..5): make them true 1D plots with axes
            else:
                p = self.eye_glw.ci.addPlot(row=row, col=col)
                p.showGrid(x=True, y=False, alpha=0.2)
                p.setLabel("bottom", "Distance (mm)")
                p.hideAxis("left")
                p.setYRange(-1, 1, padding=0.0)
                p.setXRange(-3.0, 10.0, padding=0.0)
                p.getViewBox().setBorder((80, 80, 80))
                p.setMenuEnabled(False)

                self.eye_bottom_plots.append(p)

        # Make last row smaller (row index 2)
        grid = self.eye_glw.ci.layout
        grid.setRowStretchFactor(0, 6)
        grid.setRowStretchFactor(1, 6)
        grid.setRowStretchFactor(2, 1)

        self._init_laser_position_map()  # laser map uses self.eye_vb[2] :contentReference[oaicite:1]{index=1}
        self._init_cornea_distance_plot()  # NEW

        layout = QVBoxLayout()
        layout.addWidget(self.eye_glw)
        group.setLayout(layout)
        group.setMinimumWidth(800)
        group.setMinimumHeight(800)
        return group

    def _init_cornea_distance_plot(self):
        """
        Bottom-left plot (index 0 in self.eye_bottom_plots):
          - Laser dot at x=0
          - Grey band indicating cornea region [dc, dc+thickness]
        """
        if not getattr(self, "eye_bottom_plots", None):
            return

        self.cornea_plot = self.eye_bottom_plots[0]  # bottom-left
        # If you want bottom-right too, use self.eye_bottom_plots[1]

        # Laser marker at x=0
        self.laser_1d_dot = pg.ScatterPlotItem(
            [0.0], [0.0],
            size=10,
            brush=pg.mkBrush("r"),
            pen=pg.mkPen("r")
        )
        self.cornea_plot.addItem(self.laser_1d_dot)

        # Cornea band (defaults for now)
        self.cornea_thickness_mm = 0.5
        dc_mm = 2.0
        self.cornea_band = pg.LinearRegionItem(
            values=(dc_mm, dc_mm + self.cornea_thickness_mm),
            orientation=pg.LinearRegionItem.Vertical,
            movable=False,
            brush=pg.mkBrush(150, 150, 150, 80),
            pen=pg.mkPen(150, 150, 150, 200),
        )
        self.cornea_plot.addItem(self.cornea_band)

    def create_show_scan_results(self):
        """
        Simple widget: just a scan selection dropdown and a Show button.
        """
        # ComboBox for selecting axial scan
        self.shift_scan_combo = QComboBox()
        self.shift_scan_combo.setMinimumWidth(160)

        # Show button
        self.shift_show_btn = QPushButton("Show")
        self.shift_show_btn.setFixedWidth(60)
        self.shift_show_btn.clicked.connect(self.on_show_axial_scan_clicked)

        # Horizontal layout
        row = QHBoxLayout()
        row.addWidget(self.shift_scan_combo)
        row.addWidget(self.shift_show_btn)
        row.addStretch()

        # Final layout
        layout = QVBoxLayout()
        layout.addLayout(row)
        layout.addStretch()

        group = QGroupBox("Show Axial Scans")
        group.setLayout(layout)
        return group

    def create_axial_scans_group(self):
        group = QGroupBox("Axial Scans")

        # List widget to display axial scan entries
        self.axial_scans_list = QListWidget()
        self.axial_scans_list.setSelectionMode(QListWidget.MultiSelection)

        # Buttons
        self.save_all_scans_btn = QPushButton("Save All")
        self.save_selected_scan_btn = QPushButton("Save Selected")
        self.remove_selected_scan_btn = QPushButton("Remove Selected")

        self.save_all_scans_btn.clicked.connect(self.save_all_axial_scans)
        self.save_selected_scan_btn.clicked.connect(self.save_selected_axial_scan)
        self.remove_selected_scan_btn.clicked.connect(self.remove_selected_axial_scan)

        # Horizontal layout for buttons
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_all_scans_btn)
        button_row.addWidget(self.save_selected_scan_btn)
        button_row.addWidget(self.remove_selected_scan_btn)

        # Main vertical layout
        layout = QVBoxLayout()
        layout.addWidget(self.axial_scans_list)
        layout.addLayout(button_row)

        group.setLayout(layout)
        return group


    def _fmt_plot_val(self, value, unit="GHz"):
        try:
            value = float(value)
            if np.isfinite(value):
                return f"{value:.3f} {unit}"
        except Exception:
            pass
        return "—"

    def _fmt_plot_mhz(self, value_ghz):
        """A GHz value rendered in MHz (widths read better in MHz)."""
        try:
            value = 1e3 * float(value_ghz)
            if np.isfinite(value):
                return f"{value:.1f} MHz"
        except Exception:
            pass
        return "—"

    # --- inside HiFrontend ---
    def create_andor_display_group(self):
        # White theme + fast options
        pg.setConfigOptions(
            useOpenGL=False,
            antialias=True,
            imageAxisOrder='row-major',
            background='w',
            foreground='k'
        )

        group = QGroupBox("Andor Frame and Fitting")
        lay = QVBoxLayout(group)

        self.glw = GraphicsLayoutWidget()
        lay.addWidget(self.glw)

        ci = self.glw.ci
        ci.setSpacing(12)
        ci.setContentsMargins(8, 8, 8, 8)

        # -------- Row 1: Live image --------
        self.vb_img = self.glw.addViewBox(lockAspect=True, enableMenu=False)
        self.vb_img.invertY(True)
        self.img_item = pg.ImageItem(autoDownsample=True)
        self.vb_img.addItem(self.img_item)
        self.vb_img.setBorder((170, 170, 170))

        # -------- Row 2: Spectrum + fit --------
        self.glw.nextRow()
        self.spec_vb = NotifyingViewBox()
        self.spec_plot = self.glw.addPlot(
            viewBox=self.spec_vb,
            labels={'left': 'Intensity', 'bottom': 'Pixel (X)'}
        )
        self.spec_plot.setMenuEnabled(False)
        self.spec_plot.setClipToView(False)
        self.spec_plot.enableAutoRange(x=True, y=True)
        self.spec_plot.setTitle("Spectrum + Fit", color=(40, 40, 40))
        self.spec_plot.getViewBox().setBorder((170, 170, 170))
        self.spec_vb.userScaled.connect(self._on_spec_user_scaled)

        self.fit_curve = pg.PlotDataItem(
            pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine)
        )
        self.fit_curve.setZValue(0)
        self.spec_plot.addItem(self.fit_curve)

        self.spec_points = pg.PlotDataItem(
            pen=None,
            symbol='o',
            symbolSize=4,
            symbolPen=pg.mkPen('k'),
            symbolBrush='k'
        )
        self.spec_points.setZValue(1)
        self.spec_plot.addItem(self.spec_points)

        self.mask_points = pg.PlotDataItem(
            pen=None,
            symbol='o',
            symbolSize=4,
            symbolPen=pg.mkPen('r'),
            symbolBrush='r'
        )
        self.mask_points.setZValue(2)
        self.spec_plot.addItem(self.mask_points)

        # Important: ignoreBounds=True prevents this overlay from changing plot autoscale
        self.live_fit_text = pg.TextItem(
            text="",
            color=(0, 0, 0),
            anchor=(1, 0)
        )
        _live_fit_font = QFont()
        _live_fit_font.setPointSize(7)
        self.live_fit_text.setFont(_live_fit_font)
        self.live_fit_text.setZValue(10)
        self.spec_plot.addItem(self.live_fit_text, ignoreBounds=True)

        # -------- Row 3: History --------
        self.glw.nextRow()
        self.hist_vb = NotifyingViewBox()
        self.hist_plot = self.glw.addPlot(
            viewBox=self.hist_vb,
            labels={'left': 'GHz', 'bottom': 'Frame'}
        )
        self.hist_plot.setMenuEnabled(False)
        self.hist_plot.setClipToView(False)
        self.hist_plot.enableAutoRange(x=True, y=True)
        self.hist_plot.setTitle("Shift History", color=(40, 40, 40))
        self.hist_plot.getViewBox().setBorder((170, 170, 170))
        self.hist_vb.userScaled.connect(self._on_hist_user_scaled)

        self.hist_curve = self.hist_plot.plot([], [], pen=pg.mkPen('k', width=1))

        # flags/caches
        self._spec_init_done = False
        self._spec_user_zoomed = False
        self._hist_user_zoomed = False
        self.HIST_WINDOW = 200

        return group
    # --- inside HiFrontend ---
    def _on_spec_user_scaled(self):
        self._spec_user_zoomed = True

    def _on_hist_user_scaled(self):
        self._hist_user_zoomed = True

    def create_zaber_manual_movement_group(self):
        group = QGroupBox("Manually Move Zabers")
        layout = QFormLayout()

        # Helper to build each row
        def make_movement_row(label_widget, step_input, *buttons):
            row = QHBoxLayout()
            row.addWidget(step_input)
            for btn in buttons:
                row.addWidget(btn)
            row.addStretch()
            return label_widget, row

        # LENS AXIS
        self.lens_step_input = QLineEdit("100")
        self.lens_step_input.setFixedWidth(60)

        self.lens_back_btn = QPushButton("← Back")
        self.lens_forward_btn = QPushButton("→ Forward")

        self.lens_pos_display = QLabel("Lens 0.00 µm")  # stored label
        layout.addRow(
            *make_movement_row(self.lens_pos_display, self.lens_step_input, self.lens_back_btn, self.lens_forward_btn))

        # Z STAGE AXIS
        self.z_step_input = QLineEdit("100")
        self.z_step_input.setFixedWidth(60)

        self.z_back_btn = QPushButton("← Back")
        self.z_forward_btn = QPushButton("→ Forward")

        self.z_pos_display = QLabel("Stage Z 0.00 µm")  # stored label
        layout.addRow(*make_movement_row(self.z_pos_display, self.z_step_input, self.z_back_btn, self.z_forward_btn))

        # X STAGE AXIS
        self.x_step_input = QLineEdit("100")
        self.x_step_input.setFixedWidth(60)

        self.x_left_btn = QPushButton("← Left")
        self.x_right_btn = QPushButton("→ Right")

        self.x_pos_display = QLabel("Stage X 0.00 µm")  # stored label
        layout.addRow(*make_movement_row(self.x_pos_display, self.x_step_input, self.x_left_btn, self.x_right_btn))

        # Y STAGE AXIS
        self.y_step_input = QLineEdit("100")
        self.y_step_input.setFixedWidth(60)

        self.y_up_btn = QPushButton("↑ Up")
        self.y_down_btn = QPushButton("↓ Down")

        self.y_pos_display = QLabel("Stage Y 0.00 µm")  # stored label
        layout.addRow(*make_movement_row(self.y_pos_display, self.y_step_input, self.y_up_btn, self.y_down_btn))


        # Connect lens movement buttons
        self.lens_back_btn.clicked.connect(lambda: self.move_zaber_lens_by(-1))
        self.lens_forward_btn.clicked.connect(lambda: self.move_zaber_lens_by(+1))

        # Stage Z
        self.z_back_btn.clicked.connect(
            lambda: self.move_zaber_stage_z_requested.emit(-float(self.z_step_input.text())))
        self.z_forward_btn.clicked.connect(
            lambda: self.move_zaber_stage_z_requested.emit(+float(self.z_step_input.text())))

        # Stage X
        self.x_left_btn.clicked.connect(
            lambda: self.move_zaber_stage_x_requested.emit(+float(self.x_step_input.text())))
        self.x_right_btn.clicked.connect(
            lambda: self.move_zaber_stage_x_requested.emit(-float(self.x_step_input.text())))

        # Stage Y
        self.y_up_btn.clicked.connect(lambda: self.move_zaber_stage_y_requested.emit(+float(self.y_step_input.text())))
        self.y_down_btn.clicked.connect(
            lambda: self.move_zaber_stage_y_requested.emit(-float(self.y_step_input.text())))


        # ----------------------------
        # NEW: Move laser to (R, phi) using eye tracking (ABSOLUTE target)
        # ----------------------------
        self.xy_r_input = QLineEdit("0.0")
        self.xy_r_input.setFixedWidth(60)
        self.xy_r_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 4))  # mm

        self.xy_phi_input = QLineEdit("0.0")
        self.xy_phi_input.setFixedWidth(60)
        self.xy_phi_input.setValidator(QDoubleValidator(-3600.0, 3600.0, 3))  # deg

        self.xy_move_btn = QPushButton("Move XY")
        self.xy_move_btn.setFixedWidth(70)
        self.xy_move_btn.clicked.connect(self.on_move_xy_polar_clicked)

        xy_row = QHBoxLayout()
        xy_row.addWidget(QLabel("R [mm]:"))
        xy_row.addWidget(self.xy_r_input)
        xy_row.addSpacing(6)
        xy_row.addWidget(QLabel("phi [deg]:"))
        xy_row.addWidget(self.xy_phi_input)
        xy_row.addSpacing(6)
        xy_row.addWidget(self.xy_move_btn)
        xy_row.addStretch()

        layout.addRow(xy_row)

        # ----------------------------
        # NEW: Move Z to reach Δc target (uses current Δc from eye tracking)
        # ----------------------------
        self.dc_input = QLineEdit("2.0")
        self.dc_input.setFixedWidth(60)
        self.dc_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 4))  # mm

        self.dc_move_btn = QPushButton("Move Z")
        self.dc_move_btn.setFixedWidth(70)
        self.dc_move_btn.clicked.connect(self.on_move_z_by_dc_clicked)

        dc_row = QHBoxLayout()
        dc_row.addWidget(QLabel("Δc [mm]:"))
        dc_row.addWidget(self.dc_input)
        dc_row.addSpacing(6)
        dc_row.addWidget(self.dc_move_btn)
        dc_row.addStretch()

        layout.addRow(dc_row)

        # ----------------------------
        # Move XYZ: run the XY (R, phi) move then the Z (Δc) move, back to
        # back, reusing the existing single-axis handlers unchanged. The
        # backend queues the emitted requests sequentially. This is the move
        # invoked at the start of each predefined-measurement step.
        # ----------------------------
        self.xyz_move_btn = QPushButton("Move XYZ")
        self.xyz_move_btn.setFixedWidth(90)
        self.xyz_move_btn.setToolTip(
            "Move XY to (R, phi) then move Z to Δc, using the fields above.")
        self.xyz_move_btn.clicked.connect(self.on_move_xyz_clicked)

        xyz_row = QHBoxLayout()
        xyz_row.addWidget(self.xyz_move_btn)
        xyz_row.addStretch()

        layout.addRow(xyz_row)

        group.setLayout(layout)
        return group

    def on_move_xyz_clicked(self):
        """Move XY (R, phi) then Z (Δc), back to back. Reuses the existing
        single-axis handlers so their behaviour is unchanged; the backend
        queues the emitted move requests sequentially."""
        self.on_move_xy_polar_clicked()
        self.on_move_z_by_dc_clicked()

    # ---------------- GUI Update Loop ---------------- #
    def _on_andor_mailbox(self):
        display = self.brillouin_signaller.fetch_latest_andor_display()
        if display is None:
            return
        self._display_result_fast(display)  # your fast pyqtgraph updater

    def on_stop_clicked(self):
        log.info("[Brillouin Viewer] STOP clicked.")

        # --- EMERGENCY: close shutters in a separate thread immediately ---
        try:
            threading.Thread(
                target=self.brillouin_signaller.backend.shutter_manager.close_all,
                daemon=True
            ).start()
        except Exception as e:
            log.exception("[STOP] Failed to spawn emergency shutter close thread: %s", e)

        # Existing logic: cancel operations + clean stop of live view
        self.brillouin_signaller.cancel_operations()
        self.close_all_shutters_requested.emit()
        self.stop_live_requested.emit()

    def on_cancel_event_clicked(self):
        log.info("[BrillouinViewer] Cancel button clicked.")
        # self.cancel_requested.emit()
        self.brillouin_signaller.cancel_operations()

    def on_restart_clicked(self):
        log.info("[Brillouin Viewer] Restart clicked.")
        self.stop_live_requested.emit()
        QApplication.processEvents()
        self.start_live_requested.emit()

    def update_gui(self):
        # Update the guis
        self.brillouin_signaller.update_gui()

    def save_all_axial_scans(self):
        self.save_all_axial_scans_requested.emit()

    def save_selected_axial_scan(self):
        selected_items = self.axial_scans_list.selectedItems()
        if not selected_items:
            log.warning("No scans selected.")
            return

        indices = [int(item.text().split(" - ")[0]) for item in selected_items]
        self.save_selected_axial_scans_requested.emit(indices)

    def save_axial_scan_list_to_file(self, scans: list):
        from PyQt5.QtWidgets import QFileDialog
        from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
            dataclass_to_hdf5_native_dict, save_dict_to_hdf5
        )

        if not scans:
            log.info("[Save] No axial scans to save.")
            return

        base_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Axial Scans (base name)",
            filter="All Files (*)"
        )
        if not base_path:
            return

        try:
            # HDF5 only (2026-08-21 decision): the format of record. It is
            # refactor-proof (name-based loading, unknown fields dropped),
            # unlike pickle, which pins module paths forever. Old .pkl files
            # remain loadable everywhere — they are just not written any more.
            h5_path = base_path if base_path.endswith(".h5") else base_path + ".h5"
            native_dict = dataclass_to_hdf5_native_dict(scans)
            save_dict_to_hdf5(h5_path, native_dict)
            log.info(f"[✓] HDF5 saved to: {h5_path}")

        except Exception as e:
            log.exception(f"[Error] Failed to save axial scans: {e}")

    def move_zaber_lens_by(self, direction: int):
        try:
            step = float(self.lens_step_input.text())
            self.move_zaber_eye_lens_requested.emit(direction * step)
        except ValueError:
            log.exception("[Error] Invalid lens step size input.")

    def update_stage_positions(self, x: float, y: float, z: float):
        self.x_pos_display.setText(f"X {x:.2f} µm")
        self.y_pos_display.setText(f"Y {y:.2f} µm")
        self.z_pos_display.setText(f"Z {z:.2f} µm")

    def remove_selected_axial_scan(self):
        selected_items = self.axial_scans_list.selectedItems()
        if not selected_items:
            log.warning("[Warning] No scan selected.")
            return
        indices = [int(item.text().split(" - ")[0]) for item in selected_items]
        self.remove_selected_axial_scans_requested.emit(indices)

    # ---------------- Signal Handles ---------------- #

    def update_system_state_label(self, state):
        if state.name == "IDLE":
            self.state_label.setText("● IDLE")
            self.state_label.setStyleSheet("color: gray; font-weight: bold")
        elif state.name == "BUSY":
            self.state_label.setText("● BUSY")
            self.state_label.setStyleSheet("color: orange; font-weight: bold")
        elif state.name == "FREERUNNING":
            self.state_label.setText("● LIVE")
            self.state_label.setStyleSheet("color: green; font-weight: bold")

    def on_andor_configs_clicked(self):
        dialog = AndorConfigDialog(andor_update_config=self.update_andor_config_settings, parent=self)
        dialog.exec_()

    def update_andor_config_settings(self, andor_config: AndorConfig):
        self.update_andor_config_requested.emit(andor_config)

    def update_calibration_config(self, calibration_config: CalibrationConfig):
        self.update_calibration_config_requested.emit(calibration_config)

    def request_scanning_config_file_update(self, scanning_config: ScanningConfig):
        self.update_scanning_config_requested.emit(scanning_config)

    def update_fitting_configs(self, fitting_configs: FittingConfigs):
        self.update_fitting_configs_requested.emit(fitting_configs)

    def on_fitting_configs_clicked(self):
        dialog = FindPeaksConfigDialog(on_apply=self.update_fitting_configs, parent=self)
        dialog.exec_()

    def on_reference_configs_clicked(self):
        dialog = CalibrationConfigDialog(on_apply=self.update_calibration_config, parent=self)
        dialog.exec_()


    def on_do_live_fitting_toggled(self, state: int):
        self.toggle_do_live_fitting_requested.emit()

    def update_do_live_fitting_checkbox(self, state: bool):
        self.do_live_fitting_checkbox.setChecked(state)

    def update_illumination_ui(self, is_cont: bool):
        if is_cont:
            self.illum_label_cont.setText("● Open")
            self.illum_label_cont.setStyleSheet("color: green; font-weight: bold")
            self.illum_label_pulse.setText("○ Closed")
            self.illum_label_pulse.setStyleSheet("color: gray")
            self.run_gui()  # restart live view
        else:
            self.illum_label_cont.setText("○ Open")
            self.illum_label_cont.setStyleSheet("color: gray")
            self.illum_label_pulse.setText("● Closed")
            self.illum_label_pulse.setStyleSheet("color: green; font-weight: bold")




    # ---------------- Toggle ---------------- #
    def toggle_illumination(self):
        self.toggle_illumination_requested.emit()




    def update_zaber_lens_position(self, pos: float):
        self._zaber_lens_um = pos
        self.lens_pos_display.setText(f"Lens {pos:.2f} µm")
        dx, dy, dz = self._laser_offset.dx, self._laser_offset.dy, self._laser_offset.dz
        self.laser_focus_position = RigCoord(x=0+dx/1000, y=0+dy/1000, z=pos/1000+dz/1000)

    # ---------------- GUI Update Loop ---------------- #


    def run_gui(self):
        self.start_live_requested.emit()


    # --- inside HiFrontend ---
    def _display_result_fast(self, dr):
        # -------- 1) Image: ALWAYS min-max per frame --------
        frame = np.ascontiguousarray(dr.frame)
        if frame.dtype == np.float64:
            frame = frame.astype(np.float32, copy=False)

        fmin = float(np.nanmin(frame)) if frame.size else 0.0
        fmax = float(np.nanmax(frame)) if frame.size else 1.0
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            fmin, fmax = 0.0, 1.0

        self.img_item.setImage(frame, autoLevels=False, levels=(fmin, fmax))

        # -------- 2) Spectrum + Fit --------
        x = np.asarray(getattr(dr, "x_pixels", []))
        y = np.asarray(getattr(dr, "sline", []))

        self.spec_points.setData(x, y)

        if getattr(dr, "is_fitting_available", False):
            xf = np.asarray(getattr(dr, "x_fit_refined", []))
            yf = np.asarray(getattr(dr, "y_fit_refined", []))
            self.fit_curve.setData(xf, yf)
        else:
            self.fit_curve.setData([], [])

        # -------- Live fit values drawn on Spectrum + Fit plot --------
        if getattr(dr, "is_fitting_available", False):
            freq_shift = getattr(dr, "freq_shift_ghz", None)
            left_hwhm = getattr(dr, "hwhm_left_peak", None)
            right_hwhm = getattr(dr, "hwhm_right_peak", None)
            lw_left = getattr(dr, "linewidth_left_peak", None)
            lw_right = getattr(dr, "linewidth_right_peak", None)

            # Compact 3-line overview.
            # 1: L / R / combined shift in GHz ('NA' = cone-corrected)
            # 2: raw HWHMs + deconvolved sample Γ, all MHz
            # 3: the L−R alignment meters (shift lean, width asymmetry)
            def _g(v):
                try:
                    v = float(v)
                    if np.isfinite(v):
                        return f"{v:.4f}"
                except Exception:
                    pass
                return "—"

            def _m(v):
                try:
                    v = 1e3 * float(v)
                    if np.isfinite(v):
                        return f"{v:.0f}"
                except Exception:
                    pass
                return "—"

            def _d(a, b):
                try:
                    d = 1e3 * (float(a) - float(b))
                    if np.isfinite(d):
                        return f"{d:+.1f}"
                except Exception:
                    pass
                return None

            sh_l = getattr(dr, "shift_left_peak", None)
            sh_r = getattr(dr, "shift_right_peak", None)
            na_tag = " NA" if getattr(dr, "na_corrected", False) else ""
            if sh_l is not None and sh_r is not None:
                text = (f"L {_g(sh_l)}  R {_g(sh_r)}  "
                        f"S {_g(freq_shift)} GHz{na_tag}")
            else:
                text = f"Shift {_g(freq_shift)} GHz{na_tag}"

            line2 = f"HWHM {_m(left_hwhm)}/{_m(right_hwhm)}"
            if lw_left is not None and lw_right is not None:
                line2 += f"  Γ {_m(lw_left)}/{_m(lw_right)}"
            text += "\n" + line2 + " MHz"

            meters = []
            ds = _d(sh_l, sh_r)
            if ds is not None:
                meters.append(f"Δshift {ds}")
            dw = _d(left_hwhm, right_hwhm)
            if dw is not None:
                meters.append(f"Δwidth {dw}")
            if meters:
                text += "\nL−R: " + "  ".join(meters) + " MHz"
            self.live_fit_text.setText(text)
            self.live_fit_text.setVisible(True)

            # Keep text in upper-right corner of current view
            vb = self.spec_plot.getViewBox()
            (x_min, x_max), (y_min, y_max) = vb.viewRange()
            self.live_fit_text.setPos(x_max, y_max)

        else:
            self.live_fit_text.setText("")
            self.live_fit_text.setVisible(False)

        mask = getattr(dr, "mask_for_fitting", None)
        if mask is not None and x.size and y.size:
            m = np.asarray(mask, dtype=bool)
            n = min(m.shape[0], x.shape[0], y.shape[0])
            self.mask_points.setData(x[:n][m[:n]], y[:n][m[:n]])
            self.mask_points.setVisible(True)
        else:
            self.mask_points.setData([], [])
            self.mask_points.setVisible(False)

        # one-time autorange (don’t fight user after they zoom)
        if not self._spec_init_done and x.size and y.size:
            vb = self.spec_plot.getViewBox()
            vb.setRange(
                xRange=(float(x.min()), float(x.max())),
                yRange=(float(y.min()), float(y.max())),
                padding=0.05
            )
            self._spec_init_done = True
        elif not self._spec_user_zoomed:
            # allow gentle auto-follow on Y until the user interacts
            self.spec_plot.enableAutoRange(y=True)

        # -------- 3) History --------
        val = getattr(dr, "freq_shift_ghz", None)
        if val is not None:
            self.history_data.append(float(val))
            N = len(self.history_data)

            x_hist = np.arange(N, dtype=float)
            y_hist = np.fromiter(self.history_data, dtype=float)
            self.hist_curve.setData(x_hist, y_hist)

            vb = self.hist_plot.getViewBox()
            if not self._hist_user_zoomed:
                if N <= self.HIST_WINDOW:
                    vb.setXRange(0, max(10, N), padding=0)
                else:
                    left = N - self.HIST_WINDOW
                    right = N
                    vb.setXRange(left, right, padding=0)
                self.hist_plot.enableAutoRange(y=True)

    # ---------------- Handlers ---------------- #

    def update_camera_shutter_button(self, is_open: bool):
        text = "Close" if is_open else "Open"
        self.toggle_camera_shutter_btn.setText(text)

    def populate_camera_ui(self, settings: dict):
        self._andor_exposure_time = settings["exposure"]
        self.exposure_input.setText(str(self._andor_exposure_time))
        self.gain_input.setText(str(settings["gain"]))


    def apply_camera_settings(self):
        try:
            exposure = round(float(self.exposure_input.text()), ndigits=4)
            gain = int(self.gain_input.text())

            settings = {
                "exposure": exposure,
                "gain": gain,
            }

            self.apply_camera_settings_requested.emit(settings)
            log.info("[Brillouin Viewer] Sent new camera settings to worker.")

            self.emit_camera_settings_requested.emit()

        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to apply camera settings: {e}")


    def toggle_reference_mode(self):
        self.toggle_reference_mode_requested.emit()

    def update_reference_ui(self, is_reference_mode: bool):
        if is_reference_mode:
            self.calib_label_meas.setText("○ Meas.")
            self.calib_label_meas.setStyleSheet("color: gray")
            self.calib_label_calib.setText("● Ref.")
            self.calib_label_calib.setStyleSheet("color: green; font-weight: bold")
        else:
            self.calib_label_meas.setText("● Meas.")
            self.calib_label_meas.setStyleSheet("color: green; font-weight: bold")
            self.calib_label_calib.setText("○ Ref.")
            self.calib_label_calib.setStyleSheet("color: gray")


    def update_ref_freq_input(self, freq: float):
        self.ref_freq_input.setText(f"{freq:.3f}")

    # -------------- Functions --------------


    def set_reference_freq(self):
        try:
            freq = float(self.ref_freq_input.text())
            self.update_microwave_freq_requested.emit(freq)
        except ValueError:
            log.exception("[Brillouin Viewer] [Reference] Invalid frequency input.")

    @staticmethod
    def _format_sweep_duration(prefix: str, elapsed: float) -> str:
        return f"{prefix}: {elapsed:.1f} s ({elapsed / 60:.2f} min)"

    def reset_sweep_stats(self):
        """Clear the longest and mean sweep scan duration records."""
        self._sweep_longest_seconds = None
        self._sweep_total_seconds = 0.0
        self._sweep_count = 0
        self.sweep_longest_label.setText("Longest sweep: —")
        self.sweep_mean_label.setText("Mean sweep: —")
        log.info("[Brillouin Viewer] Longest and mean sweep records reset.")

    def on_sweep_scan_finished(self, success: bool, elapsed: float = 0.0):
        """Outcome of a sweep scan (any sweep, predefined or manual).

        elapsed is the sweep duration measured backend-side from the START of
        the sweep scan (it excludes the Move XY / Move Z that position the eye
        beforehand). Only a genuinely completed sweep (success=True) records
        that time and advances the predefined-measurement list; a sweep that
        failed to start or was cancelled (success=False) records nothing and
        leaves the predefined selection where it is, so it can be retried."""
        # The scan is over: cancel the hard max-time backstop if still pending.
        self._cancel_sweep_max_time_timer()

        if success:
            log.info(f"[Brillouin Viewer] Sweep Scan finished | "
                     f"elapsed {elapsed:.1f} s "
                     f"({elapsed / 60:.2f} min)")

            self.sweep_last_label.setText(self._format_sweep_duration(
                "Last sweep", elapsed))

            # Track the longest sweep on record (until the user resets it).
            longest = getattr(self, "_sweep_longest_seconds", None)
            if longest is None or elapsed > longest:
                self._sweep_longest_seconds = elapsed
                self.sweep_longest_label.setText(self._format_sweep_duration(
                    "Longest sweep", elapsed))

            # Running mean over all sweeps since the last reset.
            self._sweep_total_seconds += elapsed
            self._sweep_count += 1
            mean = self._sweep_total_seconds / self._sweep_count
            self.sweep_mean_label.setText(self._format_sweep_duration(
                f"Mean sweep (n={self._sweep_count})", mean))
        else:
            log.info("[Brillouin Viewer] Sweep Scan did not complete "
                     "(failed to start or cancelled); elapsed time not recorded.")

        # Advance the predefined-measurement list only on a completed sweep.
        if getattr(self, "_predef_run_active", False):
            self._predef_run_active = False
            if success:
                if self._predef_index < len(self._predef_steps) - 1:
                    self._set_predef_index(self._predef_index + 1)
                else:
                    log.info("[Predefined] Reached the last step.")
            else:
                log.info("[Predefined] Step did not complete; staying on "
                         f"step {self._predef_index + 1} for retry.")

    def receive_axial_scan_list(self, scan_list: list):

        # Update QListWidget
        self.axial_scans_list.clear()
        self.axial_scans_list.addItems(scan_list)

        self.shift_scan_combo.clear()
        self.shift_scan_combo.addItems(scan_list)

        # A just-finished 'Take Ref. Bkg.' scan: ask the backend to adopt
        # it as the live reflection background (built backend-side).
        if getattr(self, "_pending_ref_bkg_autoload", False) and scan_list:
            last = scan_list[-1]
            if "reflection_background" in str(last):
                self._pending_ref_bkg_autoload = False
                try:
                    self.ref_bkg_label.setText("Ref. bkg: loading…")
                    self.ref_bkg_label.setStyleSheet("color: gray")
                    self.load_ref_bkg_from_scan_requested.emit(
                        int(str(last).split(" - ")[0]))
                except Exception:
                    log.exception("[Brillouin Viewer] Ref. bkg auto-load: "
                                  "could not parse the scan index from "
                                  f"'{last}'")

    def on_show_axial_scan_clicked(self):
        selected_scan = self.shift_scan_combo.currentText()
        if not selected_scan:
            log.info("[Brillouin Viewer] No axial available.")
            return

        i = int(selected_scan.split(" - ")[0])

        self.request_axial_scan_data.emit(i)

    def handle_received_axial_scan_data(self, scan_data: AxialScan):
        """Open the scan in the analyzer viewer — the same viewer the data
        analyzer uses: the scan is re-fitted against its own re-fitted
        calibration under the LIVE configs, frame browser + spectrum fit +
        shift-vs-frame-index profile.

        (The 'Take Ref. Bkg.' auto-load no longer routes through here —
        it goes frontend → signaller → backend via
        load_ref_bkg_from_scan_requested.)"""
        log.info(f"[Brillouin Viewer] Opening scan {scan_data.id} "
                 f"({len(scan_data.measurements)} frames) — re-fitting with "
                 f"the live configs...")
        try:
            from brillouin_system.guis.data_analyzer.show_axial_scan import (
                AxialScanViewer,
            )
            viewer = AxialScanViewer(scan_data)
        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to open scan "
                          f"{scan_data.id}: {e}")
            QMessageBox.critical(
                self, "Cannot Show Scan",
                f"Failed to open scan {scan_data.id}:\n\n"
                f"{type(e).__name__}: {e}")
            return
        # Keep a reference so the window survives; drop closed ones.
        self._open_scan_viewers = [
            v for v in getattr(self, "_open_scan_viewers", [])
            if v.isVisible()]
        self._open_scan_viewers.append(viewer)
        viewer.show()

    def run_calibration(self):
        self.run_calibration_requested.emit()

    def show_calibration_results(self):
        self._show_cali = True
        self._save_cali = False
        self.get_calibration_results_requested.emit()

    def save_calibration_results(self):
        self._show_cali = False
        self._save_cali = True
        self.get_calibration_results_requested.emit()

    def calibration_finished(self):
        self.show_calib_btn.setEnabled(True)
        self.save_calib_btn.setEnabled(True)
        log.info("[Brillouin Viewer] Calibration available")


    def handle_requested_calibration(self,
                                     received_cali: tuple[CalibrationData, CalibrationCalculator, CalibrationConfig]):
        cali_data = received_cali[0]
        cali_calculator = received_cali[1]
        log.info(cali_calculator.get_str_all_models())
        config = received_cali[2]

        if self._show_cali:
            try:
                # The calibration plot shows one px->GHz track; "combined"
                # has no single track, so its plot shows the distance one.
                plot_reference = ("distance" if config.reference == "combined"
                                  else config.reference)
                pixmap = render_calibration_to_pixmap(
                    cali_calculator, reference=plot_reference
                )
                dialog = CalibrationImageDialog(pixmap, parent=self)
                dialog.exec_()
                log.info("[Brillouin Viewer] Calibration plot displayed.")
            except Exception as e:
                log.exception(f"[Brillouin Viewer] Failed to plot calibration: {e}")

        elif self._save_cali:
            if cali_data is None:
                log.info("[Brillouin Viewer] Failed to save data, no data available")
                return

            base_path, _ = QFileDialog.getSaveFileName(
                self, "Save Calibration Data", filter="All Files (*)"
            )
            if not base_path:
                return

            try:
                # HDF5 only (2026-08-21 decision, same as the scan save).
                h5_path = base_path if base_path.endswith(".h5") else base_path + ".h5"
                hdf5_dict = dataclass_to_hdf5_native_dict(cali_data)
                save_dict_to_hdf5(h5_path, hdf5_dict)
                log.info(f"[✓] Calibration data saved as HDF5 to {h5_path}")

            except Exception as e:
                log.exception(f"[Error] Failed to save calibration data: {e}")


        self._show_cali = False
        self._save_cali = True

    def update_laser_coord_calibration(self, laser_offset: LaserOffset):
        self._laser_offset: LaserOffset = laser_offset

    def take_axial_step_scan(self, find_reflection_plane: bool = False, randomize_order: bool = False):
        try:
            id_str = self.axial_id_input.text().strip()
            n_meas = int(self.axial_num_input.text())
            step = float(self.axial_step_input.text())

            # Log info
            log.info(
                f"[Brillouin Viewer] Axial Scan Request | ID: {id_str}, N: {n_meas}, Step: {step} µm"
                f"{', random order' if randomize_order else ''}")


            request = RequestAxialStepScan(
                id=id_str,
                n_measurements=n_meas,
                step_size_um=step,
                find_reflection_plane=find_reflection_plane,
                eye_tracker_results=self.lastest_eye_tracker_results,
                randomize_order=randomize_order,
            )

            self.take_axial_step_scan_requested.emit(request)

        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to initiate axial scan: {e}")


    def take_background_scan(self):
        """A reflection-background capture IS a normal axial scan: N frames
        at the current position (step 0), fixed id 'reflection_background',
        registered and saved like any other scan — no separate pipeline.
        When the scan appears in the registry it is auto-loaded as the
        live reflection background (see receive_axial_scan_list)."""
        try:
            n_meas = int(self.axial_num_input.text())
            self._pending_ref_bkg_autoload = True

            log.info(f"[Brillouin Viewer] Background Scan Request | "
                     f"id: reflection_background, N: {n_meas}, step 0 µm "
                     f"(position at the reflection plane first) — will "
                     f"auto-load as the live reflection background")

            request = RequestAxialStepScan(
                id="reflection_background",
                n_measurements=n_meas,
                step_size_um=0.0,
                find_reflection_plane=False,
                eye_tracker_results=self.lastest_eye_tracker_results,
            )

            self.take_axial_step_scan_requested.emit(request)

        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to initiate background scan: {e}")


    # -------- Reflection background for live sample fitting --------
    # House pattern: the frontend only emits requests; the signaller runs
    # the load/build in the backend thread and reports back via
    # ref_bkg_state (so the GUI never blocks on a calibration re-fit).

    def on_load_ref_bkg_clicked(self):
        """Pick a saved .npz template or an .h5 scan file and ask the
        backend to make it the live reflection background."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Reflection Background", "",
            "Background (*.npz *.h5);;Template (*.npz);;Scan (*.h5)")
        if not path:
            return
        self.ref_bkg_label.setText("Ref. bkg: loading…")
        self.ref_bkg_label.setStyleSheet("color: gray")
        self.load_ref_bkg_from_file_requested.emit(path)

    def update_ref_bkg_label(self, status: str):
        """Backend outcome of a Load/Take Ref. Bkg. request."""
        if status.startswith("ERROR"):
            self.ref_bkg_label.setText("Ref. bkg: failed")
            self.ref_bkg_label.setStyleSheet("color: red")
            QMessageBox.critical(self, "Load Ref. Bkg. Failed", status)
        else:
            self.ref_bkg_label.setText(f"Ref. bkg: {status}")
            self.ref_bkg_label.setStyleSheet("color: green")

    def take_sweep_scan(self, max_time_s: float | None = None,
                        timed: bool = False,
                        motion_limit_um: float | None = None) -> bool:
        """Emit a sweep-scan request. Returns True if the request was sent
        (elapsed time is reported when the scan finishes via
        on_sweep_scan_finished), False if it could not be initiated.

        The max-time budget is enforced backend-side, measured from the START
        of the sweep scan (the Move XY / Move Z that position the eye run as
        earlier requests and are not counted). It is read there from
        SweepScanConfig.max_time_s; max_time_s here is only used to validate a
        timed request up front. timed: run as many cycles as fit in the budget
        instead of a fixed number of cycles (requires max_time_s > 0)."""
        try:
            id_str = self.axial_id_input.text().strip()

            if max_time_s is None:
                try:
                    max_time_s = float(sweep_scan_config.get().max_time_s)
                except Exception:
                    max_time_s = 0.0

            if timed and (not max_time_s or max_time_s <= 0):
                log.warning("[Brillouin Viewer] Timed sweep needs a max time "
                            "> 0 (set it in Sweep Settings). Aborting.")
                return False

            log.info(f"[Brillouin Viewer] {'Timed ' if timed else ''}Sweep "
                     f"Scan Request | ID: {id_str}")

            request = RequestSweepScan(
                id=id_str,
                eye_tracker_results=self.lastest_eye_tracker_results,
                timed=timed,
                motion_limit_um=motion_limit_um,
            )
            self.take_sweep_scan_requested.emit(request)
            return True

        except Exception as e:
            log.exception(f"[Brillouin Viewer] Failed to initiate sweep scan: {e}")
            return False

    def take_timed_sweep_scan(self, max_time_s: float | None = None) -> bool:
        """Run a timed sweep: as many cycles as fit in the max-time budget
        (from Sweep Settings unless max_time_s is given)."""
        return self.take_sweep_scan(max_time_s=max_time_s, timed=True)

    def on_sweep_scan_started(self, max_time_s: float):
        """The backend has begun a sweep scan (after any Move XY / Move Z).
        Arm a hard max-time backstop from this moment: the backend's own
        predictive budget should stop first, but if a cycle overruns the
        estimate this timer guarantees the sweep is ended (and its data saved)
        at the limit. Measured from the sweep start, so the positioning moves
        are not counted."""
        self._arm_sweep_max_time_timer(max_time_s)

    def _arm_sweep_max_time_timer(self, max_time_s: float | None):
        """Start a single-shot timer that ends the sweep early (saving data)
        once max_time_s has elapsed. Any previous timer is cancelled first."""
        self._cancel_sweep_max_time_timer()
        if not max_time_s or max_time_s <= 0:
            return
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._on_sweep_max_time_reached)
        timer.start(int(max_time_s * 1000))
        self._sweep_max_time_timer = timer
        log.info(f"[Brillouin Viewer] Sweep max-time backstop armed at "
                 f"{max_time_s:.1f} s.")

    def _cancel_sweep_max_time_timer(self):
        timer = getattr(self, "_sweep_max_time_timer", None)
        if timer is not None:
            timer.stop()
            self._sweep_max_time_timer = None

    def _on_sweep_max_time_reached(self):
        self._sweep_max_time_timer = None
        log.info("[Brillouin Viewer] Sweep max-time backstop reached - ending "
                 "scan early and saving collected data.")
        self.brillouin_signaller.end_scan_early()

    def on_end_scan_early_clicked(self):
        log.info("[Brillouin Viewer] End Scan Early clicked.")
        self.brillouin_signaller.end_scan_early()


    def clear_measurements(self):
        self._stored_measurements.clear()
        self.measurement_series_label.setText("Stored Series: 0")
        log.info("[Brillouin Viewer] Cleared all stored measurement series.")

    # -- Open Dialog --
    def message_handler(self, title: str, message: str):
        QMessageBox.information(self, title, message)

    # ---- Allied config dialogs ----

    def open_allied_left_dialog(self):
        dlg = AlliedConfigDialog("left", self._apply_allied_left, parent=self)
        dlg.exec_()

    def open_allied_right_dialog(self):
        dlg = AlliedConfigDialog("right", self._apply_allied_right, parent=self)
        dlg.exec_()

    def _apply_allied_left(self, cfg_obj):
        """
        Called by AlliedConfigDialog on Apply for LEFT.
        `cfg_obj` is the updated AlliedConfig for the left camera.
        We emit both left & right configs to the eye-tracker proxy.
        """
        # Push both configs into the eye tracker proxy via our signal
        self.set_et_allied_configs.emit(cfg_obj, None)

    def _apply_allied_right(self, cfg_obj):
        """
        Called by AlliedConfigDialog on Apply for RIGHT.
        """

        self.set_et_allied_configs.emit(None, cfg_obj)

    def open_eye_tracker_config_dialog(self):
        """
        Open the EyeTrackerConfigDialog and on Apply send the config to the
        eye-tracker worker via EyeTrackerProxy.set_et_config.
        """

        def _on_apply(cfg: EyeTrackerConfig):
            """
            Called by EyeTrackerConfigDialog.apply() after updating the
            global ThreadSafeConfig. `cfg` is an EyeTrackerConfig instance.
            """
            try:
                # Emit our signal, which is connected to proxy.set_et_config
                self.set_et_config.emit(cfg)
                log.info("[EyeTracker] Sent new eye tracker config to proxy.")
            except Exception as e:
                log.exception(f"[EyeTracker] Failed to send config to proxy: {e}")

        dlg = EyeTrackerConfigDialog(
            on_apply=_on_apply,
            parent=self,
        )
        dlg.exec_()

    def open_axial_scan_settings_dialog(self):

        def _on_apply(cfg: ScanningConfig):
            try:
                # Emit our signal, which is connected to proxy.set_et_config
                self.update_scanning_config_requested.emit(cfg)
                log.info("[Frontend] Sent new axial scan settings.")
            except Exception as e:
                log.exception(f"[EyeTracker] Failed to send new axial scan settings: {e}")

        dlg = AxialScanningConfigDialog(
            on_apply=_on_apply,
            parent=self,
        )
        dlg.exec_()

    def open_sweep_scan_settings_dialog(self):

        def _on_apply(cfg: SweepScanConfig):
            try:
                self.update_sweep_scan_config_requested.emit(cfg)
                log.info("[Frontend] Sent new sweep scan settings.")
            except Exception as e:
                log.exception(f"[Frontend] Failed to send new sweep scan settings: {e}")

        dlg = SweepScanConfigDialog(
            on_apply=_on_apply,
            parent=self,
        )
        dlg.exec_()

    def _init_laser_position_map(self):
        """
        Initializes the position map display in eye_vb[2].
        Includes:
            - Concentric circles (radius 1..4)
            - Radial lines at 0°, 45°, 90°, 135°
            - Moving laser point
            - Text display (x, y, Δz) in upper-right corner
        """
        vb = self.eye_vb[2]

        # -------------------------------
        # Configure ViewBox
        # -------------------------------
        vb.setAspectLocked(True)  # Keep it square
        vb.setRange(xRange=(-4, 4), yRange=(-4, 4))
        vb.invertY(False)  # make y-axis normal (upward is +y)
        # If you want "y from 4 down to -4", then:
        # vb.invertY(True)

        # -------------------------------
        # Draw concentric circles
        # -------------------------------
        angles = np.linspace(0, 2 * np.pi, 720)
        for r in range(1, 5):
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            circle = pg.PlotDataItem(
                x, y,
                pen=pg.mkPen(150, 150, 150)  # light gray
            )
            vb.addItem(circle)

        # -------------------------------
        # Radial lines: 0°, 45°, 90°, 135°
        # -------------------------------
        for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315, ]:
            theta = np.deg2rad(angle_deg)
            x = [1 * np.cos(theta), 4 * np.cos(theta)]
            y = [1 * np.sin(theta), 4 * np.sin(theta)]

            line = pg.PlotDataItem(
                x, y,
                pen=pg.mkPen(150, 150, 150, style=QtCore.Qt.DashLine)
            )
            vb.addItem(line)

        # -------------------------------
        # Moving laser dot
        # -------------------------------
        self.laser_point = pg.ScatterPlotItem(
            [0.0], [0.0],
            size=12,
            brush=pg.mkBrush('r'),
            pen=pg.mkPen('r')
        )
        vb.addItem(self.laser_point)

        # -------------------------------
        # Text display (top-right)
        # -------------------------------
        self.pos_text = TextItem(
            text="x=\ny=\nz=\nΔc=",
            color=(0, 0, 0),
            anchor=(1, 0)  # right-top anchor
        )
        vb.addItem(self.pos_text)

        # Place it at the exact upper-right of the defined range
        # If invertY(False): top = +4
        # If invertY(True):  top = -4
        top_y = 4 if not vb.yInverted() else -4

        self.pos_text.setPos(4, top_y)

    def update_laser_position_cartesian(self, x: float, y: float):
        if not (np.isfinite(x) and np.isfinite(y)):
            self.clear_laser_position()
            return
        self.laser_point.setData([float(x)], [float(y)])
        self.laser_point.setVisible(True)

    def clear_laser_position(self):
        self.laser_point.setVisible(False)

    def update_laser_position_text_eye_tracker(self, x=None, y=None, z=None, dc=None):
        """
        Update the x, y, Δz display with fixed width fields.
        If a value is None, show blank padded fields.
        """

        def fmt(v):
            return f"{v:6.3f}" if v is not None else " " * 6

        tx = fmt(x)
        ty = fmt(y)
        tz = fmt(z)
        tdc = fmt(dc)

        self.pos_text.setText(
            f"x = {tx}\n"
            f"y = {ty}\n"
            f"z = {tz}\n"
            f"Δc = {tdc}"
        )

    def calibrate_laser_position(self):

        et_result = self._wait_for_eye_result()
        if et_result is None or et_result.pupil3d is None:
            return
        pupil_center = et_result.pupil3d.center_ref

        # convert from mm to um
        x, y, z = pupil_center[0]*1000, pupil_center[1]*1000, pupil_center[2]*1000
        if self._zaber_lens_um is None:
            return
        log.info(f"Pupil Center before moving: {round(float(x)), round(float(y)), round(float(z))}")
        # Assuming Rig COS and zaber_lens share same origin
        # Moving the Rig here (not the lens)
        self.move_zaber_stage_x_requested.emit(x)
        self.move_zaber_stage_y_requested.emit(y)
        self.move_zaber_stage_z_requested.emit(z-self._zaber_lens_um)

        # Wait for the stage moves + a fresh eye result WITHOUT freezing the
        # GUI thread (processEvents keeps the event loop alive).
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.02)
        et_result = self._wait_for_eye_result()
        if et_result is None or et_result.pupil3d is None:
            return
        pupil_center = et_result.pupil3d.center_ref

        # convert from mm to um
        x, y, z = pupil_center[0]*1000, pupil_center[1]*1000, pupil_center[2]*1000
        if self._zaber_lens_um is None:
            return
        log.info(f"Pupil Center after moving: {round(float(x)), round(float(y)), round(float(z))}")

        self.calibrate_laser_camera_position_requested.emit()


    @QtCore.pyqtSlot(object, object, dict)
    def on_eye_frames_ready(self, left, right, meta):
        """
        (left, right, meta)
          left     : np.ndarray (H, W, 3), uint8
          right    : np.ndarray (H, W, 3), uint8
          meta     : Dict: {"ts": last["ts"], "idx": last["idx"], "pupil3D": pupil3D}
        """
        # left/right/rendered: np.ndarray(H, W, 3), uint8

        # pyqtgraph's ImageItem can handle 3-channel images, but a safe option is
        # to convert to grayscale for now:

        self._last_eye_update_monotonic = time.monotonic()
        self.eye_img[0].setImage(left, autoLevels=True)
        self.eye_img[1].setImage(right, autoLevels=True)

        self.lastest_eye_tracker_results = get_eye_tracker_results(
            left=left, right=right, meta=meta, laser_focus_position=self.laser_focus_position
        )

        # Dummy mode: the eye-tracker (dummy stereo cameras) and the reflection
        # finder (simulated cornea) are otherwise independent simulations, so
        # the eye-tracker Δc is unrelated to where the finder's cornea actually
        # is. Left alone, a plan's Move Z chases that bogus Δc and marches the
        # eye lens out of the cornea's search range, so every sweep's initial
        # find fails. Override Δc with the value consistent with the simulated
        # cornea (None on real hardware -> no change) so the whole predefined
        # flow — Move Z, then find, then sweep — stays self-consistent.
        if self.lastest_eye_tracker_results is not None:
            sim_dc = self.brillouin_signaller.backend.simulated_delta_laser_corner_mm()
            if sim_dc is not None:
                from dataclasses import replace
                self.lastest_eye_tracker_results = replace(
                    self.lastest_eye_tracker_results, delta_laser_corner=sim_dc)

        laser_position = self.lastest_eye_tracker_results.laser_position
        if laser_position is not None:
            self.update_laser_position_cartesian(x=laser_position[0], y=laser_position[1])
            self.update_laser_position_text_eye_tracker(
                x=laser_position[0],
                y=laser_position[1],
                z=laser_position[2],
                dc=self.lastest_eye_tracker_results.delta_laser_corner
            )

            self._update_cornea_band(self.lastest_eye_tracker_results.delta_laser_corner)

        else:
            self.clear_laser_position()
            self.update_laser_position_text_eye_tracker(None, None, None, None)

            # IMPORTANT: also hide cornea when laser_position missing
            self._update_cornea_band(None)


    def _wait_for_eye_result(self, timeout_s: float = 1, max_age_s: float = 0.5) -> EyeTrackerResults | None:
        """
        Wait (briefly) for a recent EyeTrackerResults with a valid laser_position.
        max_age_s: how recent the result must be.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            r = self.lastest_eye_tracker_results
            if r is not None and r.laser_position is not None:
                age = time.monotonic() - self._last_eye_update_monotonic
                if age <= max_age_s:
                    return r
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        return None

    def _update_cornea_band(self, dc_mm):
        # If plot isn't initialized yet, do nothing
        if not hasattr(self, "cornea_band"):
            return

        if dc_mm is None:
            self.cornea_band.setVisible(False)
            return

        self.cornea_band.setRegion((dc_mm, dc_mm + self.cornea_thickness_mm))
        self.cornea_band.setVisible(True)

    def on_move_xy_polar_clicked(self):
        """
        Move stage so the laser ends up at (R, phi) in laser coordinates.
        R [mm], phi [deg] describe the ABSOLUTE desired laser position.
        """
        try:
            res = self._wait_for_eye_result(timeout_s=0.8, max_age_s=0.3)
            if res is None:
                log.warning("[Zaber XY Polar] No recent eye-tracker result.")
                return

            x_laser = float(res.laser_position[0])
            y_laser = float(res.laser_position[1])

            r_mm = float(self.xy_r_input.text())
            phi_deg = float(self.xy_phi_input.text())

            phi = np.deg2rad(phi_deg)
            x_target = r_mm * np.cos(phi)
            y_target = r_mm * np.sin(phi)

            dx_laser = x_target - x_laser
            dy_laser = y_target - y_laser

            dx_um = dx_laser * 1000.0
            dy_um = dy_laser * 1000.0

            # Limit move to max 3.5 mm
            max_move_um = 3500.0
            move_mag_um = float(np.hypot(dx_um, dy_um))
            if move_mag_um > max_move_um:
                scale = max_move_um / move_mag_um
                dx_um *= scale
                dy_um *= scale
                log.warning(
                    f"[Zaber XY Polar] Requested move {move_mag_um / 1000:.3f} mm exceeds "
                    f"limit 3.5 mm. Clamping to 3.5 mm (scale={scale:.3f})."
                )

            # SIGN NOTE:
            # If moving the stage +X makes the laser move -X,
            # change emit(dx_um) -> emit(-dx_um) (same for Y).
            self.move_zaber_stage_x_requested.emit(-dx_um)
            self.move_zaber_stage_y_requested.emit(dy_um)

        except Exception as e:
            log.exception(f"[Zaber XY Polar] Failed: {e}")

    def on_move_z_by_dc_clicked(self):
        """
        Move eye-lens Z so that Δc reaches the user-specified value (mm).
        Assumes res.delta_laser_corner is in mm.
        """
        try:
            res = self._wait_for_eye_result(timeout_s=0.8, max_age_s=0.3)
            if res is None:
                log.warning("[Zaber Z Δc] No recent eye-tracker result.")
                return

            dc_current_mm = res.delta_laser_corner
            if dc_current_mm is None:
                log.warning("[Zaber Z Δc] delta_laser_corner is None; cannot compute Z move.")
                return

            dc_target_mm = float(self.dc_input.text())
            dz_mm = dc_target_mm - float(dc_current_mm)
            dz_um = dz_mm * 1000.0

            # Limit movement (pick your limit)
            max_move_um = 2000  # 2 mm
            dz_um = float(np.clip(dz_um, -max_move_um, max_move_um))

            log.info(
                "[Zaber Z Δc] dc_current=%.3f mm target=%.3f mm -> dz=%.3f mm (clamped to %.3f mm)",
                dc_current_mm, dc_target_mm, dz_um / 1000.0, max_move_um / 1000.0,
            )

            # Sign may need flipping depending on geometry
            self.move_zaber_eye_lens_requested.emit(-dz_um)

        except Exception as e:
            log.exception(f"[Zaber Z Δc] Invalid input or update error: {e}")

    # ---- Fitting button (placeholder) ----

    def shutdown_eye_tracker(self):
        self.request_eye_shutdown.emit()
        time.sleep(5)
        self.eye_thread.quit()
        if not self.eye_thread.wait(10000):
            log.error("Eye tracker thread did not stop within 10 s — continuing anyway.")

    def on_restart_eye_clicked(self):
        # Close old eye tracker
        self.shutdown_eye_tracker()
        # Create fresh controller + thread
        self.start_eye_tracker()


    def find_reflection_plane(self):
        self.find_reflection_plane_request.emit()


    def find_reflection_plane_backwards(self):
        self.find_reflection_plane_backwards_request.emit()


    def closeEvent(self, event):
        print("GUI shutdown initiated...")
        event.ignore()

        if include_eye_tracking:
            self.shutdown_eye_tracker()

        self.stop_live_requested.emit()
        self.shutdown_requested.emit()  # no sleep

    def _finalize_close(self):
        print("Backend shutdown complete. Closing GUI...")
        self.brillouin_signaller_thread.quit()
        self.brillouin_signaller_thread.wait(3000)

        QApplication.quit()


def main():
    # Set rounding policy before constructing QApplication (Qt ≥ 5.14)
    try:
        if hasattr(QtWidgets.QApplication, "setHighDpiScaleFactorRoundingPolicy"):
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
            )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        * { font-size: 8pt; }
    """)
    viewer = HiFrontend()
    viewer.show()
    exit_code = app.exec_()
    sys.exit(exit_code)



if __name__ == "__main__":
    # IMPORTANT: On Windows, spawn the writer process ONLY here.
    start_logging()
    install_crash_hooks()

    # For debugging:
    # enable_console_fallback(force=True)

    main()
