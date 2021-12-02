from cv2 import cv2
import numpy as np
from duncan.east_text_detection import TextRecognizer
from util.type_defs import *
from ryan.color_extractor import ColorExtractor
from util.image_functions import verbose_display


class CardIdentifier:
    def __init__(self, text_recognizer):
        self.text_recognizer = text_recognizer

    def identify_card(self, card_img: Int2D_3C, red_color: Color, blue_color: Color, yellow_color: Color) -> str:
        red_distance = ColorExtractor.distance_from_color(card_img, red_color)
        blue_distance = ColorExtractor.distance_from_color(card_img, blue_color)
        yellow_distance = ColorExtractor.distance_from_color(card_img, yellow_color, use_only_value=True)
        red_avg = np.mean(red_distance)
        blue_avg = np.mean(blue_distance)
        yellow_avg = np.mean(yellow_distance)
        print(f"Red: {red_avg}")
        print(f"Blu: {blue_avg}")
        print(f"Yel: {yellow_avg}")
        card_value = max((red_avg, "r"), (blue_avg, "b"), (yellow_avg, "y"), key=lambda x: x[0])[1]

        if card_value == "y":
            card_text = self.text_recognizer.read_card(card_img)
            if card_text is not None:
                card_value = card_text
        print(f"Card is: {card_value}")
        verbose_display([card_img, red_distance, blue_distance, yellow_distance])
        return card_value
