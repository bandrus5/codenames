from cv2 import cv2
import numpy as np
from util.type_defs import *
from util.image_functions import verbose_display


class CardCropper:
    card_size_mm = np.array([67,
                             43])  # https://www.reddit.com/r/boardgames/comments/exyr27/comment/fge7epo/?utm_source=share&utm_medium=web2x&context=3
    px_per_inch = 150
    mm_per_inch = 25.4

    @staticmethod
    def crop_card(card_bounds: Contour, original_img: Int2D_3C) -> Int2D_3C:
        width = np.linalg.norm(card_bounds[1] - card_bounds[0])
        height = np.linalg.norm(card_bounds[3] - card_bounds[0])
        card_size = CardCropper.card_size_mm.copy()
        if height > width:
            card_size = np.roll(card_size, 1)
        card_size = card_size / CardCropper.mm_per_inch * CardCropper.px_per_inch
        card_size = card_size.astype(np.int32)

        card = np.zeros([card_size[0], card_size[1], 3], dtype=np.uint8)
        cropped_card_bounds = np.array([
            [0, 0],
            [card.shape[0], 0],
            [card.shape[0], card.shape[1]],
            [0, card.shape[1]]])
        M, _ = cv2.findHomography(card_bounds, cropped_card_bounds)
        card = cv2.warpPerspective(original_img, M, (cropped_card_bounds[2,0], cropped_card_bounds[2,1]))
        return card
