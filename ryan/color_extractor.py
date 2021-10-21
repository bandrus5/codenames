import numpy as np
from cv2 import cv2
import numpy.typing as npt
from util.type_defs import *


class ColorExtractor:
    @staticmethod
    def extract_color(input_img: Int2D_3C, color: Color, threshold: int, use_only_value: bool = False) -> Int2D_1C:
        if use_only_value:
            hue_weight = 0
            saturation_weight = 0
            value_weight = 1
        else:
            hue_weight = 3
            saturation_weight = 2
            value_weight = 0
        total_weight = hue_weight + saturation_weight + value_weight
        hue_distance_img = ColorExtractor._distance_from_color_in_channel(input_img, color, 'h')
        saturation_distance_img = ColorExtractor._distance_from_color_in_channel(input_img, color, 's')
        value_distance_img = ColorExtractor._distance_from_color_in_channel(input_img, color, 'v')
        distance_img = (hue_weight * hue_distance_img +
                        saturation_weight * saturation_distance_img +
                        value_weight * value_distance_img) / total_weight
        # self.verbose_display([hue_distance_img, saturation_distance_img, value_distance_img, distance_img, input_img], 256)
        distance_img = (distance_img * 255).astype(np.uint8)
        # self.verbose_display([distance_img, median_distance_img, input_img])
        # return distance_img
        if use_only_value:
            threshold_type = cv2.THRESH_BINARY | cv2.THRESH_OTSU
        else:
            threshold_type = cv2.THRESH_BINARY
        _, threshold_img = cv2.threshold(distance_img, threshold, 255, threshold_type)
        # self.verbose_display([threshold_img, input_img])
        return threshold_img

    @staticmethod
    def _channel_from_channel_name(channel_name: str):
        channel = 0
        if channel_name == 'h' or channel_name == 'hue':
            channel = 0
        elif channel_name == 's' or channel_name == 'saturation':
            channel = 1
        elif channel_name == 'v' or channel_name == 'value':
            channel = 2
        return channel

    @staticmethod
    def _distance_from_color_in_channel(input_img: Int2D_3C, color: Color, channel_name: str = 'h') -> Float2D_1C:
        channel = ColorExtractor._channel_from_channel_name(channel_name)
        h, w = input_img.shape[:2]
        conversion = cv2.COLOR_BGR2HSV
        converted_img = cv2.cvtColor(input_img.astype(np.uint8), conversion).astype(np.float32)
        converted_color = cv2.cvtColor(color.reshape((1, 1, 3)).astype(np.uint8), conversion).astype(np.float32)
        difference_img = (converted_img[:, :, channel] - converted_color[:, :, channel])

        difference_img = np.abs(difference_img)
        if conversion == cv2.COLOR_BGR2HSV and channel == 0:
            difference_img = np.minimum(difference_img, 180 - difference_img)
        distance_img = difference_img
        distance_img = distance_img / np.max(distance_img)
        distance_img = 1 - distance_img
        return distance_img
