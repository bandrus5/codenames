from cv2 import cv2
import numpy as np
from util.type_defs import *
from ryan.color_extractor import ColorExtractor
from util.image_functions import verbose_display


class CardIdentifier:
    @staticmethod
    def identify_card(card_img: Int2D_3C, red_color: Color, blue_color: Color, yellow_color: Color) -> str:
        red_distance = ColorExtractor.distance_from_color(card_img, red_color)
        blue_distance = ColorExtractor.distance_from_color(card_img, blue_color)
        yellow_distance = ColorExtractor.distance_from_color(card_img, yellow_color, use_only_value=True)
        red_avg = np.mean(red_distance)
        blue_avg = np.mean(blue_distance)
        yellow_avg = np.mean(yellow_distance)
        print(f"Red: {red_avg}")
        print(f"Blu: {blue_avg}")
        print(f"Yel: {yellow_avg}")
        card_color = max((red_avg, "red"), (blue_avg, "blue"), (yellow_avg, "yellow"), key=lambda x: x[0])[1]
        print(f"Card is: {card_color}")
        verbose_display([card_img, red_distance, blue_distance, yellow_distance])
        return card_color
