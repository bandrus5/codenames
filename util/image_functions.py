import cv2
import numpy as np


def resize_image(
        image: np.ndarray,
        width: int = None,
        height: int = None,
        keep_aspect_ratio: bool = True) -> np.ndarray:
    if not keep_aspect_ratio and all([width, height]):
        return cv2.resize(image, (width, height))

    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim)


def display_image(image: np.ndarray):
    sized_image = resize_image(image, width=600)
    cv2.imshow("image", sized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
