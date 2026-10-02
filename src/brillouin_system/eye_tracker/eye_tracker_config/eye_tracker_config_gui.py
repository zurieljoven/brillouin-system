# eye_tracker_config_gui.py

from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QApplication,
    QMessageBox,
    QCheckBox,
    QComboBox,
)
from PyQt5.QtGui import QIntValidator

from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import (
    EyeTrackerConfig,
    PUPIL_FIT_TOML_PATH,
    load_eye_tracker_config,
    save_config_section,
    eye_tracker_config,
)
from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig


class EyeTrackerConfigDialog(QDialog):
    def __init__(
        self,
        cfg_holder: ThreadSafeConfig | None = None,
        on_apply=None,
        parent=None,
    ):
        """
        cfg_holder:
            ThreadSafeConfig[EyeTrackerConfig]. If None, the global
            `eye_tracker_config` is used.

        on_apply:
            Optional callback taking a single EyeTrackerConfig, called
            after cfg_holder has been updated (but before saving).
        """
        super().__init__(parent)
        self.setWindowTitle("Eye Tracker Configuration")

        self.cfg_holder: ThreadSafeConfig = cfg_holder or eye_tracker_config
        self.on_apply = on_apply

        self.inputs: dict[str, object] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._group_main(self.inputs))
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self._load()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _group_main(self, inputs: dict) -> QGroupBox:
        g = QGroupBox("Settings")
        v = QVBoxLayout()

        def add_row(label: str, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget, 1)
            v.addLayout(h)

        # Thresholds
        le_left_thr = QLineEdit()
        le_left_thr.setValidator(QIntValidator(0, 255))
        inputs["binary_threshold_left"] = le_left_thr
        add_row("Binary threshold (left)", le_left_thr)

        le_right_thr = QLineEdit()
        le_right_thr.setValidator(QIntValidator(0, 255))
        inputs["binary_threshold_right"] = le_right_thr
        add_row("Binary threshold (right)", le_right_thr)

        # NEW: fill_n_vetical_dark_pixels
        le_fill_left = QLineEdit()
        le_fill_left.setValidator(QIntValidator(0, 10000))
        inputs["fill_n_vetical_dark_pixels_left"] = le_fill_left
        add_row("Fill N vertical dark pixels (left)", le_fill_left)

        le_fill_right = QLineEdit()
        le_fill_right.setValidator(QIntValidator(0, 10000))
        inputs["fill_n_vetical_dark_pixels_right"] = le_fill_right
        add_row("Fill N vertical dark pixels (right)", le_fill_right)

        le_hfill_left = QLineEdit()
        le_hfill_left.setValidator(QIntValidator(0, 10000))
        inputs["fill_n_horizontal_dark_pixels_left"] = le_hfill_left
        add_row("Fill N horizontal dark pixels (left)", le_hfill_left)

        le_hfill_right = QLineEdit()
        le_hfill_right.setValidator(QIntValidator(0, 10000))
        inputs["fill_n_horizontal_dark_pixels_right"] = le_hfill_right
        add_row("Fill N horizontal dark pixels (right)", le_hfill_right)

        # Masking radii
        le_mask_left = QLineEdit()
        le_mask_left.setValidator(QIntValidator(1, 10000))
        inputs["masking_radius_left"] = le_mask_left
        add_row("Masking radius (left)", le_mask_left)

        le_mask_right = QLineEdit()
        le_mask_right.setValidator(QIntValidator(1, 10000))
        inputs["masking_radius_right"] = le_mask_right
        add_row("Masking radius (right)", le_mask_right)

        # Masking center (left) cx, cy
        le_mask_center_left_cx = QLineEdit()
        le_mask_center_left_cx.setValidator(QIntValidator(-5000, 5000))
        inputs["masking_center_left_cx"] = le_mask_center_left_cx

        le_mask_center_left_cy = QLineEdit()
        le_mask_center_left_cy.setValidator(QIntValidator(-5000, 5000))
        inputs["masking_center_left_cy"] = le_mask_center_left_cy

        row_left_center = QHBoxLayout()
        row_left_center.addWidget(QLabel("Masking center (left) cx, cy"))
        row_left_center.addWidget(le_mask_center_left_cx)
        row_left_center.addWidget(le_mask_center_left_cy)
        v.addLayout(row_left_center)

        # Masking center (right) cx, cy
        le_mask_center_right_cx = QLineEdit()
        le_mask_center_right_cx.setValidator(QIntValidator(-5000, 5000))
        inputs["masking_center_right_cx"] = le_mask_center_right_cx

        le_mask_center_right_cy = QLineEdit()
        le_mask_center_right_cy.setValidator(QIntValidator(-5000, 5000))
        inputs["masking_center_right_cy"] = le_mask_center_right_cy

        row_right_center = QHBoxLayout()
        row_right_center.addWidget(QLabel("Masking center (right) cx, cy"))
        row_right_center.addWidget(le_mask_center_right_cx)
        row_right_center.addWidget(le_mask_center_right_cy)
        v.addLayout(row_right_center)

        # Ellipse fitting / overlay
        de = QCheckBox("Run ellipse fitting")
        inputs["do_ellipse_fitting"] = de
        add_row("Ellipse fitting", de)

        ov = QCheckBox("Overlay ellipse on output")
        inputs["overlay_ellipse"] = ov
        add_row("Overlay ellipse", ov)

        # Frame returned
        cb = QComboBox()
        frame_options = ["original", "binary", "floodfilled", "contour"]
        cb.addItems(frame_options)
        inputs["frame_returned"] = cb
        add_row("Frame returned", cb)

        g.setLayout(v)
        return g

    def _buttons(self) -> QHBoxLayout:
        h = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)

        h.addStretch()
        h.addWidget(apply_btn)
        h.addWidget(save_btn)
        h.addWidget(close_btn)
        return h

    # ------------------------------------------------------------------ #
    # Data <-> UI
    # ------------------------------------------------------------------ #

    def _set_fields(self, cfg: EyeTrackerConfig):
        # Thresholds
        self.inputs["binary_threshold_left"].setText(str(cfg.binary_threshold_left))
        self.inputs["binary_threshold_right"].setText(str(cfg.binary_threshold_right))

        # NEW: fill
        self.inputs["fill_n_vetical_dark_pixels_left"].setText(
            str(cfg.fill_n_vetical_dark_pixels_left)
        )
        self.inputs["fill_n_vetical_dark_pixels_right"].setText(
            str(cfg.fill_n_vetical_dark_pixels_right)
        )
        self.inputs["fill_n_horizontal_dark_pixels_left"].setText(
            str(cfg.fill_n_horizontal_dark_pixels_left)
        )
        self.inputs["fill_n_horizontal_dark_pixels_right"].setText(
            str(cfg.fill_n_horizontal_dark_pixels_right)
        )

        # Masking
        self.inputs["masking_radius_left"].setText(str(cfg.masking_radius_left))
        self.inputs["masking_radius_right"].setText(str(cfg.masking_radius_right))

        lcx, lcy = cfg.masking_center_left
        rcx, rcy = cfg.masking_center_right

        self.inputs["masking_center_left_cx"].setText(str(lcx))
        self.inputs["masking_center_left_cy"].setText(str(lcy))
        self.inputs["masking_center_right_cx"].setText(str(rcx))
        self.inputs["masking_center_right_cy"].setText(str(rcy))

        # Ellipse fitting
        self.inputs["do_ellipse_fitting"].setChecked(bool(cfg.do_ellipse_fitting))
        self.inputs["overlay_ellipse"].setChecked(bool(cfg.overlay_ellipse))

        # Frame returned
        frame_options = ["original", "binary", "floodfilled", "contour"]
        try:
            idx = frame_options.index(cfg.frame_returned)
        except ValueError:
            idx = 0
        self.inputs["frame_returned"].setCurrentIndex(idx)

    def _collect(self) -> dict:
        def _intval(key: str, default: int) -> int:
            text = self.inputs[key].text()
            return int(text) if text else default

        return {
            "binary_threshold_left": _intval("binary_threshold_left", 20),
            "binary_threshold_right": _intval("binary_threshold_right", 20),

            # NEW: fill defaults
            "fill_n_vetical_dark_pixels_left": _intval("fill_n_vetical_dark_pixels_left", 10),
            "fill_n_vetical_dark_pixels_right": _intval("fill_n_vetical_dark_pixels_right", 10),
            "fill_n_horizontal_dark_pixels_left": _intval("fill_n_horizontal_dark_pixels_left", 0),
            "fill_n_horizontal_dark_pixels_right": _intval("fill_n_horizontal_dark_pixels_right", 0),

            "masking_radius_left": _intval("masking_radius_left", 500),
            "masking_radius_right": _intval("masking_radius_right", 500),

            # Fixed defaults to match dataclass (0,0)
            "masking_center_left": (
                _intval("masking_center_left_cx", 0),
                _intval("masking_center_left_cy", 0),
            ),
            "masking_center_right": (
                _intval("masking_center_right_cx", 0),
                _intval("masking_center_right_cy", 0),
            ),

            "do_ellipse_fitting": bool(self.inputs["do_ellipse_fitting"].isChecked()),
            "overlay_ellipse": bool(self.inputs["overlay_ellipse"].isChecked()),
            "frame_returned": self.inputs["frame_returned"].currentText(),
        }

    def _load(self):
        """Load current config from the ThreadSafeConfig into the widgets."""
        cfg = self.cfg_holder.get()
        self._set_fields(cfg)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def apply(self):
        """Apply settings to in-memory config (but do not save TOML)."""
        try:
            kwargs = self._collect()
            self.cfg_holder.update(**kwargs)

            if self.on_apply:
                self.on_apply(self.cfg_holder.get())

            QMessageBox.information(self, "Applied", "Settings applied (not saved).")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply config:\n{e}")

    def save(self):
        """Apply then persist to TOML."""
        try:
            self.apply()
            save_config_section(PUPIL_FIT_TOML_PATH, "eye_tracker", self.cfg_holder)
            QMessageBox.information(self, "Saved", f"Saved to {PUPIL_FIT_TOML_PATH.name}.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")


if __name__ == "__main__":
    import sys

    holder = ThreadSafeConfig(load_eye_tracker_config(PUPIL_FIT_TOML_PATH, "eye_tracker"))
    app = QApplication(sys.argv)
    dlg = EyeTrackerConfigDialog(cfg_holder=holder)
    dlg.show()
    sys.exit(app.exec_())
