from cv2 import cv2
import numpy as np
from typing import *


def _calculate_final_size(
        original_width: int,
        original_height: int,
        desired_width: Optional[int],
        desired_height: Optional[int]) -> Tuple[int, int]:
    if desired_width is None:
        r = desired_height / float(original_height)
        dim = (int(original_width * r), desired_height)
    elif desired_height is None:
        r = desired_width / float(original_width)
        dim = (desired_width, int(original_height * r))
    else:
        original_aspect_ratio = original_width / original_height
        desired_aspect_ratio = desired_width / desired_height
        if original_aspect_ratio > desired_aspect_ratio:
            final_width = desired_width
            final_height = final_width / original_width * original_height
        else:
            final_height = desired_height
            final_width = final_height / original_height * original_width
        dim = (int(final_width), int(final_height))
    return dim


def resize_image(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_aspect_ratio: bool = True,
        interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    if not keep_aspect_ratio and all([width, height]):
        return cv2.resize(image, (width, height), interpolation=interpolation)

    if width is None and height is None:
        return image

    (h, w) = image.shape[:2]
    dim = _calculate_final_size(w, h, width, height)

    return cv2.resize(image, dim, interpolation=interpolation)


def display_image(image: np.ndarray, width: int = 600, height: int = 600):
    sized_image = resize_image(image, width=width, height=height)
    cv2.imshow("image", sized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_hough_line_segments(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return line_image
