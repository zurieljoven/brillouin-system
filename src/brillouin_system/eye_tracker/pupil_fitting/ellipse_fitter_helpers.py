import cv2
import numpy as np
from dataclasses import dataclass
from brillouin_system.eye_tracker.pupil_fitting.ellipse2D import Ellipse2D
from enum import Enum, auto

from brillouin_system.eye_tracker.pupil_fitting.fill_vertical_gaps import fill_vertical_gaps_binary_fast, fill_horizontal_gaps_binary_fast


class PupilImgType(Enum):
    """Stages of the pupil detection pipeline."""
    ORIGINAL = auto()       # raw input or normalized grayscale
    BINARY = auto()         # after thresholding (pupil dark → white)
    FLOODFILLED = auto()    # after background flood-fill removal
    CONTOUR = auto()        # largest connected component / final contour mask

    def __str__(self):
        return self.name.lower()




@dataclass
class PupilEllipse:
    """
    Container for pupil-detection results.

    Attributes:
        pupil_img: Returned image at the requested stage.
        pupil_img_type: Which stage this image represents.
        ellipse: Fitted ellipse parameters or None if no valid fit.
                 Fields: (cx, cy, major, minor, angle_deg)
    """
    pupil_img: np.ndarray
    pupil_img_type: PupilImgType
    ellipse: Ellipse2D | None


def img_to_be_returned(
        pupil_img_type: PupilImgType,
        original_frame: np.ndarray | None,
        binary_img: np.ndarray | None,
        floodfilled_img: np.ndarray | None,
        contour_img: np.ndarray | None,
) -> tuple[np.ndarray, PupilImgType]:
    """
    Returns the requested image type from the pupil detection pipeline,
    with graceful fallback to the most processed available image.

    Returns:
        (image, actual_type): tuple containing the selected image and
        the PupilImgType that corresponds to it.
    """
    img_map = {
        PupilImgType.ORIGINAL: original_frame,
        PupilImgType.BINARY: binary_img,
        PupilImgType.FLOODFILLED: floodfilled_img,
        PupilImgType.CONTOUR: contour_img,
    }

    # Try the explicitly requested image first
    img = img_map.get(pupil_img_type)
    if img is not None:
        return img, pupil_img_type

    # Fallback order: most processed → least
    fallback_order = [
        (contour_img, PupilImgType.CONTOUR),
        (floodfilled_img, PupilImgType.FLOODFILLED),
        (binary_img, PupilImgType.BINARY),
        (original_frame, PupilImgType.ORIGINAL),
    ]

    for candidate, candidate_type in fallback_order:
        if candidate is not None:
            return candidate, candidate_type

    raise ValueError("No valid image available to return (all inputs were None).")

def _ensure_u8(img: np.ndarray) -> np.ndarray:
    """Normalize any numeric single- or 3-channel image to uint8 (keeps shape)."""
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    m, M = np.min(img), np.max(img)
    if M > m:
        scaled = (img - m) * (255.0 / (M - m))
        return np.ascontiguousarray(scaled.astype(np.uint8))
    # constant image
    return np.zeros_like(img, dtype=np.uint8)



def extract_roi(img, roi_center_xy, roi_width_height):
    """
    Return a cropped ROI around (cx, cy) with given (width, height).

    - img:        np.ndarray (H, W) or (H, W, C)
    - roi_center_xy: (cx, cy) in pixel coordinates
    - roi_width_height: (w, h) in pixels

    Behavior:
    - Computes a centered rectangular ROI.
    - Clips ROI to image bounds.

    Returns:
    Returns:
        roi: img[y1_clip:y2_clip, x1_clip:x2_clip]
        offset_xy: (x1_clip, y1_clip)
    """
    H, W = img.shape[:2]
    cx, cy = roi_center_xy
    rw, rh = roi_width_height

    # Half extents
    half_w = rw // 2
    half_h = rh // 2

    # Intended ROI bounds (inclusive)
    x1 = cx - half_w
    x2 = cx + half_w
    y1 = cy - half_h
    y2 = cy + half_h

    # Clip ROI to image bounds for slicing (note: upper bound is exclusive)
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(W, x2 + 1)   # +1 because slice end is exclusive
    y2_clip = min(H, y2 + 1)

    # Handle degenerate case: ROI completely outside image → empty slice
    if x1_clip >= x2_clip or y1_clip >= y2_clip:
        return img[0:0, 0:0]   # empty array of correct ndim

    roi = img[y1_clip:y2_clip, x1_clip:x2_clip]

    # Darken the outermost pixels of the original image
    img[0, :] = 0          # top row
    img[-1, :] = 0         # bottom row
    img[:, 0] = 0          # left column
    img[:, -1] = 0         # right column

    return roi, x1_clip, y1_clip


def make_img_black_outside_ring_around_center(img: np.ndarray,
                                              ring_radius: int,
                                              center: tuple[int, int] = (0,0,),
                                              make_copy: bool = False) -> np.ndarray:
    """
    Zero (black) all pixels whose distance from the image center is > ring_radius.
    Keeps whatever is inside the circle, blackens everything outside.

    Parameters
    ----------
    img : np.ndarray
        Grayscale or color image.
    ring_radius : int
        Radius (in pixels) of the kept circular region around the center.
    center: tuple[int, int],
        cx, cy (horizontal: left to right, vertical: bottom to top)
    make_copy: bool
        makes a copy of the image if True

    Returns
    -------
    np.ndarray
        New image with pixels outside the circle set to 0.
    """
    h, w = img.shape[:2]
    cy, cx = h // 2 - center[1], w // 2 + center[0]

    # Build distance grid
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    # Mask for points INSIDE the circle
    mask = dist2 <= ring_radius ** 2

    if make_copy:
        out = img.copy()
    else:
        out = img

    if img.ndim == 2:
        # Grayscale
        out[~mask] = 0
    else:
        # Color: broadcast mask over channels
        out[~mask, :] = 0

    return out

def find_pupil_ellipse_with_flooding(
    img: np.ndarray,
    threshold: int = 20,
    fill_n_vetical_dark_pixels: int = 0,
    fill_n_horizontal_dark_pixels: int = 0,
    frame_to_be_returned: PupilImgType = PupilImgType.ORIGINAL
) -> PupilEllipse:
    """
    Fast pupil ellipse detection via binary inversion, flood fill,
    largest component, and ellipse fit.

    Returns a PupilEllipse whose `pupil_img` is whichever stage you request.
    """
    img = _ensure_u8(img)

    # Step 1: Binary inverse threshold — pupil dark → white
    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    # Only keep a copy of the binary if it's the requested output
    binary = bw.copy() if frame_to_be_returned == PupilImgType.BINARY else None

    # Step 2: Flood fill from corners to remove background (in-place on bw)
    h, w = bw.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(bw, mask, (0, 0), 128)
    # cv2.floodFill(bw, mask, (w - 1, 0), 128)
    # cv2.floodFill(bw, mask, (0, h - 1), 128)
    # cv2.floodFill(bw, mask, (w - 1, h - 1), 128)
    bw[bw == 128] = 0  # remove background marked by 128

    # Step 2.2: Fill gaps (eye lashes might be in the way)
    if fill_n_vetical_dark_pixels != 0:
        bw = fill_vertical_gaps_binary_fast(img=bw, n_vertical_pixels=fill_n_vetical_dark_pixels, make_copy=False)
    if fill_n_horizontal_dark_pixels != 0:
        bw = fill_horizontal_gaps_binary_fast(img=bw, n_horizontal_pixels=fill_n_horizontal_dark_pixels, make_copy=False)

    # Step 3: Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8, ltype=cv2.CV_32S)
    ellipse_obj = None
    contour_img = None

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        # Build a mask for the largest component (0/255)
        contour_img = np.zeros_like(bw, dtype=np.uint8)
        contour_img[labels == largest_label] = 255

        # Step 4: Fit ellipse from the largest component
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
                (cx, cy), (major, minor), angle_deg = cv2.fitEllipse(cnt)
                ellipse_obj = Ellipse2D(cx, cy, major, minor, angle_deg)

    # Decide which image to return
    out_img, out_type = img_to_be_returned(
        pupil_img_type=frame_to_be_returned,
        original_frame=img,
        binary_img=binary,
        floodfilled_img=bw,
        contour_img=contour_img,
    )

    return PupilEllipse(
        pupil_img=out_img,
        pupil_img_type=out_type,
        ellipse=ellipse_obj,
    )


