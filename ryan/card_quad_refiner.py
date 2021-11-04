from cv2 import cv2
import numpy as np
from util.image_functions import verbose_display
from ryan.card_contour_extractor import CardContourExtractor
from ryan.card_grid_recognizer import CardGridRecognizer
from util.type_defs import *


class CardQuadRefiner:
    KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Watershed
    SURE_CARD_VALUE = 2
    SURE_BACKGROUND_VALUE = 1
    UNKNOWN_VALUE = 0

    # GrabCut
    # SURE_CARD_VALUE = cv2.GC_FGD
    # SURE_BACKGROUND_VALUE = cv2.GC_BGD
    # UNKNOWN_VALUE = cv2.GC_PR_FGD

    @staticmethod
    def refine_cards_quads(cards_contours: List[Contour], input_img: Int2D_3C) -> List[Contour]:
        refined_contours = []
        for card_contour in cards_contours:
            refined_contour = CardQuadRefiner.refine_card_quad(card_contour, input_img)
            refined_contours.append(refined_contour)
        return refined_contours

    # @staticmethod
    # def grab_cut(mask_img: Int2D_1C, input_img: Int2D_3C):
    #     bg_model = np.zeros((1, 65), dtype=np.float)
    #     fg_model = np.zeros((1, 65), dtype=np.float)
    #     mask_img, _, _ = cv2.grabCut(input_img, mask_img, None, bg_model, fg_model, 1, cv2.GC_INIT_WITH_MASK)
    #     return mask_img

    @staticmethod
    def refine_card_quad(card_quad: Contour, input_img: Int2D_3C) -> Contour:
        initial_quad_img = CardContourExtractor.display_contours(input_img.copy(), [card_quad], return_result=True,
                                                                 thickness=10, color=(0, 255, 0))
        cv2.imwrite("viz/12_card_quad.png", initial_quad_img)

        mask_img = CardQuadRefiner._create_initial_mask(card_quad, input_img.shape[:2])
        # refined_mask_img = CardQuadRefiner.grab_cut(mask_img, input_img)
        refined_mask_img = cv2.watershed(input_img, mask_img.astype(np.int32))

        refined_mask_img = (refined_mask_img == CardQuadRefiner.SURE_CARD_VALUE).astype(np.uint8)
        refined_card_quad, contours = CardGridRecognizer.find_largest_quad(refined_mask_img, input_img,
                                                                           simplify_epsilon=0.1)

        if len(refined_card_quad) == 4:
            accepted_quad, rejected_quad = refined_card_quad, card_quad
        else:
            accepted_quad, rejected_quad = card_quad, refined_card_quad

        # refined_quad_img = CardContourExtractor.display_contours(input_img.copy(), [rejected_quad], return_result=True,
        #                                                          thickness=2, color=(0, 0, 255), number_points=True)
        refined_quad_img = CardContourExtractor.display_contours(input_img.copy(), [accepted_quad], return_result=True,
                                                                 thickness=10, color=(0, 255, 0))
        display_mask_img = np.stack([np.zeros(mask_img.shape), np.zeros(mask_img.shape), mask_img * 85],
                                    axis=2).astype(np.uint8)
        display_mask_img = cv2.addWeighted(display_mask_img, 0.5, input_img, 0.5, 1)
        cv2.imwrite("viz/13_markers.png", display_mask_img)
        cv2.imwrite("viz/14_refined_mask.png", refined_mask_img * 255)
        cv2.imwrite("viz/15_refined_quad.png", refined_quad_img)
        verbose_display([display_mask_img, refined_mask_img * 127, refined_quad_img])
        return accepted_quad

    @staticmethod
    def _create_initial_mask(card_contour: Contour, image_size: Tuple[int, int]) -> Int2D_1C:
        contour_img = np.zeros(image_size, dtype=np.uint8)
        cv2.drawContours(contour_img, [card_contour], -1, 1, -1)
        sure_card = cv2.erode(contour_img, CardQuadRefiner.KERNEL, iterations=25)

        sure_background = cv2.dilate(contour_img, CardQuadRefiner.KERNEL, iterations=25)
        sure_background = cv2.absdiff(sure_background, 1)

        unknown = cv2.bitwise_or(sure_background, sure_card)
        unknown = cv2.absdiff(unknown, 1)

        mask_img = (sure_card * CardQuadRefiner.SURE_CARD_VALUE +
                    sure_background * CardQuadRefiner.SURE_BACKGROUND_VALUE +
                    unknown * CardQuadRefiner.UNKNOWN_VALUE)
        return mask_img
