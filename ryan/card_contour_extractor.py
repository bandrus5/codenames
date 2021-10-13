from cv2 import cv2
from typing import *
import numpy as np


class CardContourExtractor:
    _card_contour = np.array([[[0, 0]], [[60, 0]], [[60, 94]], [[0, 95]]])
    # _card_contour = np.array([[[416, 559]], [[357, 561]], [[360, 655]], [[420, 650]]])

    @staticmethod
    def extract_card_contours(threshold_img: np.ndarray, original_img: np.ndarray) -> List[np.ndarray]:
        contours = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = CardContourExtractor._simplify_contours(contours, 0.1)
        return contours

    @staticmethod
    def _simplify_contours( contours: List[np.ndarray], max_difference: float = 0.1) -> List[np.ndarray]:
        simplified_contours = []
        for contour in contours:
            epsilon = max_difference * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            simplified_contours.append(simplified_contour)
        return simplified_contours

    # def _find_best_contours(self, input_img: np.ndarray, keep_count: int, original_img: np.ndarray) -> List[np.ndarray]:
    #
    #     contours = sorted(simplified_contours, key=RyanImageProcessor._get_contour_card_similarity, reverse=False)
    #     if len(contours) <= keep_count:
    #         keep_count = len(contours)
    #     contours = contours[:keep_count]
    #     if self.is_verbose:
    #         contour_img = original_img.copy()
    #         for contour in contours:
    #             contour_img = original_img.copy()
    #             print(f"Similarity: {self._get_contour_card_similarity(contour)}")
    #             print(f"Lines: {contour.shape[0]}")
    #             print(f"Area: {cv2.contourArea(contour)}")
    #             cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
    #             self.verbose_display([contour_img])
    #     return contours

    @staticmethod
    def _get_contour_card_similarity(contour) -> float:
        card_similarity = cv2.matchShapes(CardContourExtractor._card_contour, contour, cv2.CONTOURS_MATCH_I1, 0)
        return card_similarity
        # segment_count = contour.shape[0]
        # contour_area = cv2.contourArea(contour)
        # return contour_area
        # similarity = contour_area - segment_count
        # return similarity

