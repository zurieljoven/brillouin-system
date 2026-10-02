import cv2
import numpy as np

def fill_vertical_gaps_binary_fast(img: np.ndarray, n_vertical_pixels: int = 10, make_copy: bool = False) -> np.ndarray:
    """
    (0/255 in, 0/255 out)
    Fill pixels that are 0 but have a 255 within <=n above AND within <=n below in the same column.
    """
    if n_vertical_pixels <= 0:
        return img.copy() if make_copy else img

    bw255 = img
    if bw255.dtype != np.uint8:
        raise TypeError("binary must be uint8 with values 0/255")

    k = np.ones((n_vertical_pixels + 1, 1), np.uint8)

    above = cv2.dilate(
        bw255, k,
        anchor=(0, n_vertical_pixels),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    below = cv2.dilate(
        bw255, k,
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    fill = (bw255 == 0) & (above != 0) & (below != 0)

    out = bw255.copy() if make_copy else bw255
    out[fill] = 255
    return out


def fill_horizontal_gaps_binary_fast(img: np.ndarray, n_horizontal_pixels: int = 10, make_copy: bool = False) -> np.ndarray:
    """
    (0/255 in, 0/255 out)
    Fill pixels that are 0 but have a 255 within <=n to the left AND within <=n to the right in the same row.
    """
    if n_horizontal_pixels <= 0:
        return img.copy() if make_copy else img

    bw255 = img
    if bw255.dtype != np.uint8:
        raise TypeError("binary must be uint8 with values 0/255")

    k = np.ones((1, n_horizontal_pixels + 1), np.uint8)

    left = cv2.dilate(
        bw255, k,
        anchor=(n_horizontal_pixels, 0),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    right = cv2.dilate(
        bw255, k,
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    fill = (bw255 == 0) & (left != 0) & (right != 0)

    out = bw255.copy() if make_copy else bw255
    out[fill] = 255
    return out
