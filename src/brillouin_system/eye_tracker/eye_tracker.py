
import numpy as np

from brillouin_system.devices.cameras.allied.allied_config.allied_config import AlliedConfig
from brillouin_system.devices.cameras.allied.own_subprocess.dual_camera_proxy import DualCameraProxy
from brillouin_system.eye_tracker.eye_tracker_config.eye_tracker_config import EyeTrackerConfig, eye_tracker_config
from brillouin_system.eye_tracker.eye_tracker_dataclasses import EyeTrackerSettings, EyeTrackerResultsForGui
from brillouin_system.eye_tracker.eye_tracker_helpers import scale_image, ensure_uint8, draw_ellipse_rgb, gray_to_rgb, \
    make_black_image
from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter import EllipseFitter
from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import PupilEllipse
from brillouin_system.eye_tracker.pupil_fitting.pupil3D import Pupil3D
from brillouin_system.eye_tracker.pupil_fitting.pupil_detector import PupilDetector

from brillouin_system.eye_tracker.stereo_imaging.init_stereo_cameras import stereo_cameras
from brillouin_system.eye_tracker.stereo_imaging.init_se3 import left_to_ref




IMG_SIZE = (512, 512)
ELLIPSE_OVERLAY_COLOR = (0, 255, 0)  # RGB: green
ELLIPSE_OVERLAY_THICKNESS = 3



class EyeTracker:
    def __init__(self, use_dummy=False):

        self._config: EyeTrackerConfig | None = None

        self.pupil_detector = PupilDetector(stereo_cameras=stereo_cameras, left_to_ref=left_to_ref)
        self.ellipse_fitter = EllipseFitter()

        init_config = eye_tracker_config.get()
        self.set_config(config=init_config)

        self.dual_cam_proxy = DualCameraProxy(dtype="uint8", slots=8, use_dummy=use_dummy)
        self.dual_cam_proxy.start()

    def set_config(self, config: EyeTrackerConfig):
        self._config = config

        self.ellipse_fitter.set_config(
            binary_threshold_left=config.binary_threshold_left,
            binary_threshold_right=config.binary_threshold_right,
            fill_n_vetical_dark_pixels_left=config.fill_n_vetical_dark_pixels_left,
            fill_n_vetical_dark_pixels_right=config.fill_n_vetical_dark_pixels_right,
            masking_radius_left=config.masking_radius_left,
            masking_radius_right=config.masking_radius_right,
            masking_center_left=config.masking_center_left,
            masking_center_right=config.masking_center_right,
            frame_to_be_returned=config.frame_returned,
            fill_n_horizontal_dark_pixels_left=config.fill_n_horizontal_dark_pixels_left,
            fill_n_horizontal_dark_pixels_right=config.fill_n_horizontal_dark_pixels_right,
        )

    def _get_settings(self) -> EyeTrackerSettings:
        return EyeTrackerSettings(
            config=self._config,
            stereo_calibration=self.pupil_detector.stereo.st_cal,
        )


    def _get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        left_img, right_img, meta = self.dual_cam_proxy.get_frames()
        return left_img, right_img, float(meta["ts"])  # ts from time.time()


    def _get_ellipses(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple[PupilEllipse, PupilEllipse]:
        pupil_eL: PupilEllipse = self.ellipse_fitter.find_pupil_left(left_img)
        pupil_eR: PupilEllipse = self.ellipse_fitter.find_pupil_right(right_img)
        return pupil_eL, pupil_eR

    def get_pupil3D(self, eL: Ellipse2D | None, eR: Ellipse2D | None) -> Pupil3D | None:
        # return self.pupil_detector.triangulate_center_using_cones(eL=eL, eR=eR)
        return self.pupil_detector.triangulate_center(eL=eL, eR=eR)

    def _get_cam_imgs_for_display(self, pupil_eL: PupilEllipse, pupil_eR: PupilEllipse) -> tuple[np.ndarray, np.ndarray]:
        """
        Return display-ready left and right images according to current config.

        Parameters
        ----------
        pupil_eL, pupil_eR : objects from EllipseFitter
            Contain .original_img, .binary_img, .floodfilled_img, etc.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (left_img, right_img) scaled for display.
        """

        # select source images
        left_img = pupil_eL.pupil_img
        right_img = pupil_eR.pupil_img

        # Convert to uint8
        left_img = ensure_uint8(left_img)
        right_img = ensure_uint8(right_img)

        # Ensure RGB
        left_img = gray_to_rgb(left_img)
        right_img = gray_to_rgb(right_img)

        if self._config.overlay_ellipse:
            if pupil_eL.ellipse is not None:
                left_img = draw_ellipse_rgb(img_rgb=left_img,
                                           ellipse=pupil_eL.ellipse,
                                           color_rgb = ELLIPSE_OVERLAY_COLOR,  # green
                                           thickness = ELLIPSE_OVERLAY_THICKNESS, # int
                                           )

            if pupil_eR.ellipse is not None:
                right_img = draw_ellipse_rgb(img_rgb=right_img,
                                           ellipse=pupil_eR.ellipse,
                                           color_rgb = ELLIPSE_OVERLAY_COLOR,  # green
                                           thickness = ELLIPSE_OVERLAY_THICKNESS, # int
                                           )



        return left_img, right_img

    def get_eye_tracker_results_from_original_imgs(self,
                                                   left_img: np.ndarray,
                                                   right_img: np.ndarray) -> EyeTrackerResultsForGui:
        # Ensure they are all rgb
        if self._config.do_ellipse_fitting:
            pupil_eL, pupil_eR = self._get_ellipses(left_img, right_img)
            pupil_3D = self.get_pupil3D(pupil_eL.ellipse, pupil_eR.ellipse)
            cam_left_img, cam_right_img = self._get_cam_imgs_for_display(pupil_eL, pupil_eR)

        else:
            pupil_3D=None
            cam_left_img = gray_to_rgb(ensure_uint8(left_img))
            cam_right_img = gray_to_rgb(ensure_uint8(right_img))

        # Ensure all images are correctly scaled
        # ensure correct scaling and rgb (for shared memory space) for GUI display
        cam_left_img = scale_image(cam_left_img, IMG_SIZE)
        cam_right_img = scale_image(cam_right_img, IMG_SIZE)

        cam_left_img = ensure_uint8(cam_left_img)
        cam_right_img = ensure_uint8(cam_right_img)

        return EyeTrackerResultsForGui(
            pupil3D = pupil_3D,
            cam_left_img=cam_left_img,
            cam_right_img=cam_right_img,
        )

    def get_results_for_gui(self) -> EyeTrackerResultsForGui:

        try:
            left_img, right_img, ts = self._get_frames()
        except:
            return EyeTrackerResultsForGui(
                pupil3D=None,
                cam_left_img=make_black_image(max_size=IMG_SIZE, channels=3),
                cam_right_img=make_black_image(max_size=IMG_SIZE, channels=3),
            )

        # Ensure they are all rgb
        et_results = self.get_eye_tracker_results_from_original_imgs(left_img=left_img, right_img=right_img)


        return et_results

    def set_allied_vision_configs(self, cfg_left: AlliedConfig, cfg_right: AlliedConfig):
        self.dual_cam_proxy.set_configs(cfg_left=cfg_left, cfg_right=cfg_right)


    def shutdown(self):
        self.dual_cam_proxy.shutdown()
