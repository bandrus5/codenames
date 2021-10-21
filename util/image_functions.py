from cv2 import cv2
import numpy as np
from typing import *
import os
from util.type_defs import *
from matplotlib import pyplot as plt

is_verbose = False


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


def convert_to_Int2D_3C(image: Union[Int2D_1C, Int2D_3C, Float2D_1C, Float2D_3C]) -> Int2D_3C:
    converted_img = image.reshape((image.shape[0], image.shape[1], -1))
    if converted_img.shape[2] < 3:
        converted_img = np.concatenate([converted_img] * 3, axis=2)
    if converted_img.dtype != np.uint8:
        converted_img = (converted_img * 255).astype(np.uint8)
    return converted_img


def combine_images(images: List[Union[Int2D_1C, Int2D_3C, Float2D_3C, Float2D_1C]], image_width: int) -> Int2D_3C:
    if not isinstance(images, List):
        images = [images]
    reshaped_imgs = []
    for image in images:
        reshaped = convert_to_Int2D_3C(image)
        reshaped = resize_image(reshaped, width=image_width)
        reshaped_imgs.append(reshaped)
    max_height = 0
    for image in reshaped_imgs:
        max_height = np.maximum(max_height, image.shape[0])
    padded_imgs = []
    for image in reshaped_imgs:
        padding = (max_height - image.shape[0]) / 2
        image = cv2.copyMakeBorder(image, int(np.ceil(padding)), int(np.floor(padding)), 0, 0, cv2.BORDER_CONSTANT,
                                   (0, 0, 0))
        padded_imgs.append(image)
    combined_img = np.concatenate(padded_imgs, axis=1)
    return combined_img


def display_image(image: np.ndarray, width: int = 600, height: int = 600):
    sized_image = resize_image(image, width=width, height=height)
    cv2.imshow("image", sized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def verbose_display(
        images_to_display: List[Union[Int2D_1C, Int2D_3C, Float2D_3C, Float2D_1C]], display_size: int = 512):
    if is_verbose:
        combined_img = combine_images(images_to_display, display_size)
        display_image(combined_img, width=display_size * len(images_to_display))


def save_image(image: np.ndarray, location: str):
    if not os.path.exists(os.path.dirname(location)):
        print(f'{os.path.dirname(location)} does not exist, creating directory')
        os.makedirs(os.path.dirname(location))
    cv2.imwrite(location, image)


def draw_hough_line_segments(image: Union[Int2D_3C, Int2D_1C], lines: np.ndarray, color: Color = np.array([0, 0, 255])) -> Int2D_3C:
    line_image = image.copy()
    line_image = convert_to_Int2D_3C(line_image)
    color = color.tolist()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), color, 1)
    return line_image


def _draw_histogram_channel(histogram_img: Int2D_3C, histogram_for_channel: Float1D, color: Color, max_value: float):
    color = color.tolist()
    h, w = histogram_img.shape[:2]
    multiplier = h / max_value
    for i in range(1, w):
        pt1 = (i - 1, int(h - multiplier * histogram_for_channel[i - 1]))
        pt2 = (i, int(h - multiplier * histogram_for_channel[i]))
        cv2.line(histogram_img, pt1, pt2, color, thickness=1)


def draw_histogram(histogram: Histogram) -> Int2D_3C:
    bin_count = histogram[0].shape[0]
    histogram_height = bin_count
    histogram_img = np.zeros((histogram_height, bin_count, 3), dtype=np.uint8)
    max_value = 0
    for channel in range(3):
        max_value = np.maximum(max_value, np.max(histogram[channel]))

    for channel in range(3):
        color = np.array([0, 0, 0], dtype=np.uint8)
        color[channel] += 255
        _draw_histogram_channel(histogram_img, histogram[channel], color, max_value)
    return histogram_img
