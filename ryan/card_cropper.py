from cv2 import cv2
import numpy as np
from util.type_defs import *
from util.image_functions import verbose_display


class CardCropper:
    card_size_mm = np.array([67,  # Width x Height. X, Y.
                             43])  # https://www.reddit.com/r/boardgames/comments/exyr27/comment/fge7epo/?utm_source=share&utm_medium=web2x&context=3
    px_per_inch = 150
    mm_per_inch = 25.4

    @staticmethod
    def crop_cards(cards_quads: List[Contour], crop_img: Int2D_3C) -> List[Int2D_3C]:
        cropped_cards = []
        for card_quad in cards_quads:
            cropped_card = CardCropper.crop_card(card_quad, crop_img)
            cropped_cards.append(cropped_card)
        return cropped_cards

    @staticmethod
    def crop_card(card_quad: Contour, original_img: Int2D_3C) -> Int2D_3C:
        height = np.linalg.norm(card_quad[1] - card_quad[0])
        width = np.linalg.norm(card_quad[3] - card_quad[0])
        card_size = CardCropper.card_size_mm.copy()
        should_rotate_card = height > width
        if should_rotate_card:
            card_size = np.roll(card_size, 1)
        card_size = card_size / CardCropper.mm_per_inch * CardCropper.px_per_inch
        card_size = card_size.astype(np.int32)

        card = np.zeros([card_size[1], card_size[0], 3], dtype=np.uint8)
        cropped_card_bounds = np.array([  # Cols, Rows. Y, X
            [0, 0],  # TL
            [0, card.shape[0]],  # TR
            [card.shape[1], card.shape[0]],  # BR
            [card.shape[1], 0]  # BL
        ])
        M, _ = cv2.findHomography(card_quad, cropped_card_bounds)
        card = cv2.warpPerspective(original_img, M, (cropped_card_bounds[2, 0], cropped_card_bounds[2, 1]))
        if should_rotate_card:
            card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return card
