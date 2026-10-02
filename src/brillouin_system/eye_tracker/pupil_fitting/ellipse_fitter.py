import numpy as np

from brillouin_system.eye_tracker.pupil_fitting.ellipse_fitter_helpers import PupilEllipse, \
    find_pupil_ellipse_with_flooding, PupilImgType, make_img_black_outside_ring_around_center

RETURN_FRAME_MAPPING = {
    "original": PupilImgType.ORIGINAL,
    "binary": PupilImgType.BINARY,
    "floodfilled": PupilImgType.FLOODFILLED,
    "contour": PupilImgType.CONTOUR,
}

def map_return_frame(frame_to_be_returned: str) -> PupilImgType:
    """
    Map a string to a PupilImgType.
    Falls back to ORIGINAL if input is unknown.
    """
    return RETURN_FRAME_MAPPING.get(frame_to_be_returned.lower(), PupilImgType.ORIGINAL)

class EllipseFitter:
    """
    Holds left/right configs (from TOML via your config accessors) and provides:
      - find_pupil_left(image)
      - find_pupil_right(image)
    Both call a single internal algorithm _find_pupil(image, cfg).

    Configs are ALWAYS used (no ad-hoc overrides here). Call .refresh() to re-pull
    latest values if you edit the TOML or change them via your config GUI.
    """

    def __init__(self) -> None:
        self._binary_threshold_left: int = 20
        self._binary_threshold_right: int = 20
        self._masking_radius_left: int = 500
        self._masking_radius_right: int = 500
        self._fill_n_vetical_dark_pixels_left: int = 0
        self._fill_n_vetical_dark_pixels_right: int = 0
        self._fill_n_horizontal_dark_pixels_left: int = 0
        self._fill_n_horizontal_dark_pixels_right: int = 0
        self._masking_center_left: tuple[int, int] = (0, 0)
        self._masking_center_right: tuple[int, int] = (0, 0)
        self._frame_to_be_returned: PupilImgType = PupilImgType.ORIGINAL

    def set_config(
            self,
            binary_threshold_left: int,
            binary_threshold_right: int,
            fill_n_vetical_dark_pixels_left: int,
            fill_n_vetical_dark_pixels_right: int,
            masking_radius_left: int,
            masking_radius_right: int,
            masking_center_left: tuple[int, int],
            masking_center_right: tuple[int, int],
            frame_to_be_returned: str,
            fill_n_horizontal_dark_pixels_left: int = 0,
            fill_n_horizontal_dark_pixels_right: int = 0,
    ) -> None:
        """
        Directly sets all internal configuration fields.
        This does NOT pull values from the EyeTrackerConfig dataclass.
        frame_to_be_returned: "original", "binary", "floodfilled", "contour"
        """

        self._binary_threshold_left = binary_threshold_left
        self._binary_threshold_right = binary_threshold_right

        self._fill_n_vetical_dark_pixels_left = fill_n_vetical_dark_pixels_left
        self._fill_n_vetical_dark_pixels_right = fill_n_vetical_dark_pixels_right
        self._fill_n_horizontal_dark_pixels_left = fill_n_horizontal_dark_pixels_left
        self._fill_n_horizontal_dark_pixels_right = fill_n_horizontal_dark_pixels_right

        self._masking_radius_left = masking_radius_left
        self._masking_radius_right = masking_radius_right

        self._masking_center_left = masking_center_left
        self._masking_center_right = masking_center_right

        self._frame_to_be_returned: PupilImgType = map_return_frame(frame_to_be_returned)

    # ---- Public API ----



    def find_pupil_left(self, image: np.ndarray) -> PupilEllipse:

        image = make_img_black_outside_ring_around_center(img=image,
                                                          ring_radius=self._masking_radius_left,
                                                          center=self._masking_center_left,
                                                          make_copy=False)

        return find_pupil_ellipse_with_flooding(img=image,
                                                threshold=self._binary_threshold_left,
                                                fill_n_vetical_dark_pixels=self._fill_n_vetical_dark_pixels_left,
                                                fill_n_horizontal_dark_pixels=self._fill_n_horizontal_dark_pixels_left,
                                                frame_to_be_returned=self._frame_to_be_returned)




    def find_pupil_right(self, image: np.ndarray) -> PupilEllipse:

        image = make_img_black_outside_ring_around_center(img=image,
                                                          ring_radius=self._masking_radius_right,
                                                          center=self._masking_center_right,
                                                          make_copy=False)

        return find_pupil_ellipse_with_flooding(img=image,
                                                threshold=self._binary_threshold_right,
                                                fill_n_vetical_dark_pixels=self._fill_n_vetical_dark_pixels_right,
                                                fill_n_horizontal_dark_pixels=self._fill_n_horizontal_dark_pixels_right,
                                                frame_to_be_returned=self._frame_to_be_returned)

