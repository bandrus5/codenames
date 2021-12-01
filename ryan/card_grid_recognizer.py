from cv2 import cv2
from util.type_defs import *
from util.image_functions import verbose_display, verbose_save
from util.image_functions import draw_hough_line_segments
from ryan.card_contour_extractor import CardContourExtractor
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import util.image_functions


class CardGridRecognizer:
    @staticmethod
    def get_cards_quads(threshold_img: Int2D_1C, original_img: Int2D_3C, row_total: int = 5, col_total: int = 5) -> \
    List[Contour]:
        grid_quad = CardGridRecognizer._find_grid_quad(threshold_img, original_img)
        point_count = grid_quad.shape[0]
        if point_count != 4:
            print(f"Error finding grid quad! Only {point_count} points were found instead of 4")
            return []

        cards_quads = []
        for row in range(row_total):
            for col in range(col_total):
                card_quad = CardGridRecognizer._get_card_quad(grid_quad, row, col, row_total, col_total)
                cards_quads.append(card_quad)
        draw_img = original_img.copy()
        draw_img = CardContourExtractor.display_contours(draw_img, cards_quads, return_result=True, thickness=10)
        for i, card in enumerate(cards_quads):
            point = CardGridRecognizer._point_between_quad(card, 0.5, 0.2)
            cv2.putText(draw_img, str(i), point, cv2.FONT_HERSHEY_PLAIN, 3, color=(0, 0, 255), thickness=6)
        verbose_save("grid", draw_img)
        verbose_display(draw_img)
        return cards_quads

    @staticmethod
    def _point_between_points(point_a: Point, point_b: Point, percent_of_a: float) -> Point:
        between_point = point_a * percent_of_a + point_b * (1 - percent_of_a)
        between_point = between_point.astype(np.int32)
        return between_point

    @staticmethod
    def _point_between_quad(quad: Contour, percent_from_left: float, percent_from_top: float) -> Point:
        quad = quad.reshape(-1, 2)
        top = CardGridRecognizer._point_between_points(quad[0], quad[1], 1 - percent_from_left)
        bottom = CardGridRecognizer._point_between_points(quad[3], quad[2], 1 - percent_from_left)
        point = CardGridRecognizer._point_between_points(top, bottom, 1 - percent_from_top)
        point = point.astype(np.int32)
        return point

    @staticmethod
    def _get_card_quad(grid_quad: Contour, card_row: int, card_col: int, row_total: int = 5,
                       col_total: int = 5) -> Contour:
        card_contour = []
        for i in range(0, 2):
            for j in range(0, 2):
                percent_from_left = (card_row + i) / row_total
                percent_from_top = (card_col + j) / col_total
                point = CardGridRecognizer._point_between_quad(grid_quad, percent_from_left, percent_from_top)
                card_contour.append(point)
        card_contour = [card_contour[0], card_contour[2], card_contour[3], card_contour[1]]
        card_contour = np.array(card_contour)
        return card_contour

    @staticmethod
    def _find_grid_quad(threshold_img: Int2D_1C, original_img: Int2D_3C) -> Contour:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshold_img1 = cv2.dilate(threshold_img, kernel, iterations=30, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        verbose_save("dilate", threshold_img1)
        threshold_img2 = cv2.erode(threshold_img1, kernel, iterations=60, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        threshold_img3 = cv2.dilate(threshold_img2, kernel, iterations=30, borderType=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        verbose_save("erode", threshold_img3)
        grid_quad, contours = CardGridRecognizer.find_largest_quad(threshold_img3, original_img)

        contour_img = CardContourExtractor.display_contours(original_img, contours, return_result=True, thickness=10)
        verbose_save("contour", contour_img)
        simplified_contour_img = CardContourExtractor.display_contours(original_img, [grid_quad],
                                                                       return_result=True, thickness=10)
        verbose_save("simplified", simplified_contour_img)
        combined_img = verbose_display(
            [threshold_img, threshold_img1, threshold_img2, threshold_img3, contour_img, simplified_contour_img],
            display_size=400)
        return grid_quad

    @staticmethod
    def make_topleft_point_first(contour: Contour) -> Contour:
        squared_distances = [x[0] ** 2 + x[1] ** 2 for x in contour]
        topleft_index = np.argmin(squared_distances)
        arranged_contour = np.roll(contour, -topleft_index, axis=0)
        return arranged_contour

    @staticmethod
    def draw_centers(input_img: Int2D_3C, centers: IntArray, size: int, color: Color):
        input_img = input_img.copy()
        color = color.tolist()
        for center in centers:
            center = (center[0], center[1])
            cv2.circle(input_img, center, size, color, -1)
        return input_img

    @staticmethod
    def find_largest_quad(threshold_img: Int2D_1C, original_img: Int2D_3C, simplify_epsilon: float = 0.1) \
            -> Tuple[Contour, List[Contour]]:
        contours = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [c.reshape(-1, 2) for c in contours]
        simple_contours = [CardContourExtractor.simplify_contour(contour, simplify_epsilon) for contour in contours]
        simple_contours = [c.reshape(-1, 2) for c in simple_contours]
        areas = [cv2.contourArea(contour) for contour in simple_contours]
        largest_contour = simple_contours[np.argmax(areas)]
        # CardContourExtractor.display_contours(original_img, [largest_contour], number_points=True)
        largest_quad = CardGridRecognizer.make_topleft_point_first(largest_contour)
        # CardContourExtractor.display_contours(original_img, [largest_quad], number_points=True)
        return largest_quad, contours
