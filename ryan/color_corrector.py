import numpy as np
import numpy.typing as npt
from util.image_functions import *
from ryan.color_extractor import ColorExtractor
from cv2 import cv2
from util.type_defs import *


class ColorCorrector:
    bin_count = 256

    @staticmethod
    def correct_color(input_img: Int2D_3C, return_extras: bool = False, mask: Optional[Int2D_1C] = None):
        original_histogram = ColorCorrector.calc_channel_histograms(input_img, mask)
        max_channel_values = ColorCorrector.get_max_channel_values(original_histogram, 5)
        channels = []
        for channel in range(3):
            channel_img = np.zeros(input_img.shape[:2], dtype=np.uint8)
            scale = (ColorCorrector.bin_count / max_channel_values[channel]) * ColorCorrector.bin_count
            cv2.normalize(input_img[..., channel], channel_img, alpha=0, beta=scale, norm_type=cv2.NORM_MINMAX)
            channels.append(channel_img.astype(np.uint8))
        corrected_img = np.stack(channels, axis=2)
        if return_extras:
            return corrected_img, max_channel_values, original_histogram
        else:
            return corrected_img

    @staticmethod
    def calc_channel_histograms(input_img: Int2D_3C, mask: Optional[Int2D_1C] = None) -> Histogram:
        histogram = []
        for channel in range(3):
            channel_histogram = cv2.calcHist([input_img], [channel], mask, [ColorCorrector.bin_count], [0, 255])
            channel_histogram /= channel_histogram.sum()
            channel_histogram = channel_histogram.reshape(-1)
            histogram.append(channel_histogram)
        return histogram

    @staticmethod
    def get_max_channel_values(histogram: Histogram, max_clip_percent: float) -> Int1D:
        max_clip_percent /= 100
        max_channel_values = np.zeros(3, dtype=np.uint8)
        for channel in range(3):
            clip_percent = 0.0
            bin_index = int(histogram[channel].shape[0])
            while clip_percent < max_clip_percent:
                bin_index -= 1
                clip_percent += histogram[channel][bin_index]
            bin_index += 1
            max_channel_values[channel] = bin_index
        return max_channel_values
