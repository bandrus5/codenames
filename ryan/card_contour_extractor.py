from cv2 import cv2
from typing import *
from util.image_functions import verbose_display
import util.image_functions
import numpy as np
from util.type_defs import *


class CardContourExtractor:
    _card_contour = np.array([[[0, 0]], [[60, 0]], [[60, 94]], [[0, 95]]])
    # _card_contour = np.array([[[416, 559]], [[357, 561]], [[360, 655]], [[420, 650]]])

    @staticmethod
    def extract_card_contours(threshold_img: Int2D_1C, original_img: Int2D_3C) -> List[Contour]:
        contours = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = CardContourExtractor._simplify_contours(contours, 0.1)
        # contours = CardContourExtractor._filter_contours(contours)
        if util.image_functions.is_verbose:
            CardContourExtractor._display_contours(original_img, contours, display_all_at_once=False)
        return contours

    @staticmethod
    def _simplify_contours( contours: List[Contour], max_difference: float = 0.1) -> List[Contour]:
        simplified_contours = []
        for contour in contours:
            epsilon = max_difference * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            simplified_contours.append(simplified_contour)
        return simplified_contours

    @staticmethod
    def _display_contours(input_img: Int2D_3C, contours: List[Contour], display_all_at_once: bool = True):
        if display_all_at_once:
            background_img = input_img.copy()
        for i, contour in enumerate(contours):
            if not display_all_at_once:
                background_img = input_img.copy()
                line_count = contour.shape[0]
                similarity = CardContourExtractor._get_contour_card_similarity(contour)
                area = cv2.contourArea(contour)
                print(f"{i} - Count: {line_count:3} Similarity: {similarity:.5f} Area: {area:.2f}")
            cv2.drawContours(background_img, [contour], -1, (0, 255, 0), 2)
            if not display_all_at_once:
                verbose_display([background_img])
        if display_all_at_once:
            verbose_display([background_img])


    # def _find_best_contours(self, input_img: np.ndarray, keep_count: int, original_img: np.ndarray) -> List[np.ndarray]:
    #
    #     contours = sorted(simplified_contours, key=RyanImageProcessor._get_contour_card_similarity, reverse=False)
    #     if len(contours) <= keep_count:
    #         keep_count = len(contours)
    #     contours = contours[:keep_count]

    @staticmethod
    def _get_contour_card_similarity(contour: Contour) -> float:
        card_similarity = cv2.matchShapes(CardContourExtractor._card_contour, contour, cv2.CONTOURS_MATCH_I1, 0)
        return card_similarity
        # segment_count = contour.shape[0]
        # contour_area = cv2.contourArea(contour)
        # return contour_area
        # similarity = contour_area - segment_count
        # return similarity

    @staticmethod
    def _filter_contours(contours: List[Contour]) -> List[Contour]:
        pass

