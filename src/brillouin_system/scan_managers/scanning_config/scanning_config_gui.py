from __future__ import annotations

from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig
from brillouin_system.scan_managers.scanning_config.scanning_config import (
    AXIAL_SCANNING_TOML_PATH,
    ScanningConfig,
    axial_scanning_config,
    load_axial_scanning_config,
    save_config_section,
)


class AxialScanningConfigDialog(QDialog):
    def __init__(self, cfg_holder: ThreadSafeConfig | None = None, on_apply=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axial Scanning Configuration")
        self.setMinimumSize(430, 500)

        self.cfg_holder: ThreadSafeConfig = cfg_holder or axial_scanning_config
        self.on_apply = on_apply

        self.inputs: dict[str, QLineEdit] = {}

        layout = QVBoxLayout()
        layout.addWidget(self._group_axial_scanning())
        layout.addLayout(self._buttons())
        self.setLayout(layout)

        self.load_values()

    # ------------------------------------------------------------------ #
    # UI helpers
    # ------------------------------------------------------------------ #

    def _add_row(self, layout: QVBoxLayout, label: str, key: str, widget: QLineEdit):
        self.inputs[key] = widget
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(widget, 1)
        layout.addLayout(row)

    def _group_axial_scanning(self) -> QGroupBox:
        g = QGroupBox("Reflection Finder")
        v = QVBoxLayout()

        le_sample_rate = QLineEdit()
        le_sample_rate.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "NI sample rate [Hz]", "ni_sample_rate_hz", le_sample_rate)

        le_speed = QLineEdit()
        le_speed.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Speed [µm/s]", "speed_um_s", le_speed)

        le_max_distance = QLineEdit()
        le_max_distance.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Max distance [µm]", "max_distance_um", le_max_distance)

        le_hi_sigma = QLineEdit()
        le_hi_sigma.setValidator(QIntValidator(0, 1_000_000))
        self._add_row(v, "High threshold [nσ]", "threshold_high_n_sigma", le_hi_sigma)

        le_lo_sigma = QLineEdit()
        le_lo_sigma.setValidator(QIntValidator(0, 1_000_000))
        self._add_row(v, "Low threshold [nσ]", "threshold_low_n_sigma", le_lo_sigma)

        le_bg_acqui = QLineEdit()
        le_bg_acqui.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Background acquisition [s]", "bg_acqui_s", le_bg_acqui)

        le_debounce = QLineEdit()
        le_debounce.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Debounce [s]", "debounce_s", le_debounce)

        le_z_poll = QLineEdit()
        le_z_poll.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Z poll [s]", "z_poll_s", le_z_poll)

        le_alpha = QLineEdit()
        le_alpha.setValidator(QDoubleValidator(0.0, 1.0, 6))
        self._add_row(v, "Alpha", "alpha", le_alpha)

        le_chunk_size = QLineEdit()
        le_chunk_size.setValidator(QIntValidator(1, 1_000_000))
        self._add_row(v, "Chunk size", "chunk_size", le_chunk_size)

        le_idle_sleep = QLineEdit()
        le_idle_sleep.setValidator(QDoubleValidator(0.0, 1e12, 6))
        self._add_row(v, "Idle sleep [s]", "idle_sleep_s", le_idle_sleep)

        le_z_offset = QLineEdit()
        le_z_offset.setValidator(QDoubleValidator(-1e12, 1e12, 6))
        self._add_row(v, "Z offset [µm]", "z_offset_um", le_z_offset)

        le_min_above = QLineEdit()
        le_min_above.setValidator(QIntValidator(1, 1_000_000))
        self._add_row(v, "Min samples above threshold", "min_samples_above", le_min_above)

        for label, key in (("Homodyne sample rate [Hz/ch]", "homodyne_sample_rate_hz"),
                           ("Homodyne speed [µm/s]", "homodyne_speed_um_s"),
                           ("Homodyne max distance [µm]", "homodyne_max_distance_um"),
                           ("Homodyne threshold [nσ]", "homodyne_n_sigma"),
                           ("Homodyne stop after [µm]", "homodyne_stop_after_um"),
                           ("Homodyne band low [Hz]", "homodyne_band_lo_hz"),
                           ("Homodyne band high [Hz]", "homodyne_band_hi_hz"),
                           ("Homodyne RMS window [s]", "homodyne_rms_window_s")):
            le = QLineEdit()
            le.setValidator(QDoubleValidator(0.0, 1e12, 6))
            self._add_row(v, label, key, le)

        le_homodyne_chunk = QLineEdit()
        le_homodyne_chunk.setValidator(QIntValidator(1, 1_000_000))
        self._add_row(v, "Homodyne chunk size", "homodyne_chunk_size", le_homodyne_chunk)

        g.setLayout(v)
        return g

    def _buttons(self) -> QHBoxLayout:
        h = QHBoxLayout()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)

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

    def load_values(self) -> None:
        cfg: ScanningConfig = self.cfg_holder.get()

        for k, w in self.inputs.items():
            w.setText(str(getattr(cfg, k)))

    def _update_config_from_inputs(self) -> None:
        def _req(key: str) -> str:
            return self.inputs[key].text().strip()

        self.cfg_holder.update(
            ni_sample_rate_hz=float(_req("ni_sample_rate_hz")),
            speed_um_s=float(_req("speed_um_s")),
            max_distance_um=float(_req("max_distance_um")),
            threshold_high_n_sigma=int(_req("threshold_high_n_sigma")),
            threshold_low_n_sigma=int(_req("threshold_low_n_sigma")),
            bg_acqui_s=float(_req("bg_acqui_s")),
            debounce_s=float(_req("debounce_s")),
            z_poll_s=float(_req("z_poll_s")),
            alpha=float(_req("alpha")),
            chunk_size=max(1, int(_req("chunk_size"))),
            idle_sleep_s=float(_req("idle_sleep_s")),
            z_offset_um=float(_req("z_offset_um")),
            min_samples_above=max(1, int(_req("min_samples_above"))),
            homodyne_sample_rate_hz=float(_req("homodyne_sample_rate_hz")),
            homodyne_speed_um_s=float(_req("homodyne_speed_um_s")),
            homodyne_max_distance_um=float(_req("homodyne_max_distance_um")),
            homodyne_n_sigma=float(_req("homodyne_n_sigma")),
            homodyne_stop_after_um=float(_req("homodyne_stop_after_um")),
            homodyne_band_lo_hz=float(_req("homodyne_band_lo_hz")),
            homodyne_band_hi_hz=float(_req("homodyne_band_hi_hz")),
            homodyne_rms_window_s=float(_req("homodyne_rms_window_s")),
            homodyne_chunk_size=max(1, int(_req("homodyne_chunk_size"))),
        )

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def apply_settings(self) -> None:
        try:
            self._update_config_from_inputs()
            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())
            print("[Axial Scanning Config] Settings applied.")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", str(e))

    def save_config(self) -> None:
        try:
            self._update_config_from_inputs()
            save_config_section(AXIAL_SCANNING_TOML_PATH, "axial_scanning", self.cfg_holder)
            if callable(self.on_apply):
                self.on_apply(self.cfg_holder.get())
            print("[Axial Scanning Config] Settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))


if __name__ == "__main__":
    import sys

    def example_send(cfg: ScanningConfig):
        print(cfg)

    holder = ThreadSafeConfig(load_axial_scanning_config(AXIAL_SCANNING_TOML_PATH))

    app = QApplication(sys.argv)
    dlg = AxialScanningConfigDialog(holder, example_send)
    dlg.exec_()
