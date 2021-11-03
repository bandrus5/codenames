from cv2 import cv2
from util.type_defs import *
from util.image_functions import verbose_display
from util.image_functions import draw_hough_line_segments
from ryan.card_contour_extractor import CardContourExtractor
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import util.image_functions


class CardGridRecognizer:
    @staticmethod
    def get_cards_bounds(threshold_img: Int2D_1C, original_img: Int2D_3C, row_total: int = 5, col_total: int = 5) -> List[Contour]:
        grid_border = CardGridRecognizer._find_grid_border(threshold_img, original_img)
        point_count = grid_border.shape[0]
        if point_count != 4:
            print(f"Error finding grid border! Only {point_count} points were found instead of 4")
            return []

        cards_bounds = []
        for row in range(row_total):
            for col in range(col_total):
                card_bounds = CardGridRecognizer._get_bounds_of_card(grid_border, row, col, row_total, col_total)
                cards_bounds.append(card_bounds)
        draw_img = original_img.copy()
        for i, card in enumerate(cards_bounds):
            point = CardGridRecognizer._point_between_border(card, 0.2, 0.5)
            cv2.putText(draw_img, str(i), point, cv2.FONT_HERSHEY_PLAIN, 4, color=(0, 0, 255), thickness=6)
        draw_img = CardContourExtractor.display_contours(draw_img, cards_bounds,return_result=True, thickness=10)
        cv2.imwrite("viz/10_grid.png", draw_img)
        verbose_display(draw_img)
        return cards_bounds

    @staticmethod
    def _point_between_points(point_a: Point, point_b: Point, percent_of_a: float) -> Point:
        between_point = point_a * percent_of_a + point_b * (1 - percent_of_a)
        between_point = between_point.astype(np.int32)
        return between_point

    @staticmethod
    def _point_between_border(border: Contour, percent_from_left: float, percent_from_top: float) -> Point:
        border = border.reshape(-1, 2)
        top = CardGridRecognizer._point_between_points(border[0], border[1], 1 - percent_from_left)
        bottom = CardGridRecognizer._point_between_points(border[3], border[2], 1 - percent_from_left)
        point = CardGridRecognizer._point_between_points(top, bottom, 1 - percent_from_top)
        point = point.astype(np.int32)
        return point

    @staticmethod
    def _get_bounds_of_card(grid_border: Contour, card_row: int, card_col: int, row_total: int = 5,
                           col_total: int = 5) -> Contour:
        card_contour = []
        for i in range(0, 2):
            for j in range(0, 2):
                percent_from_left = (card_row + i) / row_total
                percent_from_top = (card_col + j) / col_total
                point = CardGridRecognizer._point_between_border(grid_border, percent_from_left, percent_from_top)
                card_contour.append(point)
        card_contour[2], card_contour[3] = card_contour[3], card_contour[2]
        card_contour = np.array(card_contour)
        return card_contour

    @staticmethod
    def _find_grid_border(threshold_img: Int2D_1C, original_img: Int2D_3C) -> Contour:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshold_img1 = cv2.dilate(threshold_img, kernel, iterations=30, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite("viz/07_dilate.png", threshold_img1)
        threshold_img2 = cv2.erode(threshold_img1, kernel, iterations=60, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        threshold_img3 = cv2.dilate(threshold_img2, kernel, iterations=30, borderType=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        cv2.imwrite("viz/07_erode.png", threshold_img3)
        contours = cv2.findContours(threshold_img3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [c.reshape(-1, 2) for c in contours]
        contour_img = CardContourExtractor.display_contours(original_img, contours, return_result=True, thickness=10)
        cv2.imwrite("viz/08_contour.png", contour_img)
        contours = [CardContourExtractor.simplify_contour(contour, 0.1) for contour in contours]
        contours = [c.reshape(-1, 2) for c in contours]
        areas = [cv2.contourArea(contour) for contour in contours]
        largest_contour = contours[np.argmax(areas)]
        simplified_contour_img = CardContourExtractor.display_contours(original_img, [largest_contour],
                                                                       return_result=True, thickness=10)
        cv2.imwrite("viz/09_simplified.png", simplified_contour_img)
        combined_img = verbose_display(
            [threshold_img, threshold_img1, threshold_img2, threshold_img3, contour_img, simplified_contour_img],
            display_size=400)
        grid_border = CardGridRecognizer.make_topleft_point_first(largest_contour)
        return grid_border

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

    # @staticmethod
    # def grid_objective(x, centers):
    #     x = x.reshape(-1, 2)
    #     distance = 0
    #     for pt in x:
    #         pt_distance = np.linalg.norm(centers - pt)
    #         distance += pt_distance
    #     return distance
    #     pt0 = x.reshape(-1, 2)[0]
    #     pt1 = x.reshape(-1, 2)[1]
    #     pt0_distance = np.linalg.norm(centers - pt0)
    #     # return pt0_distance
    #     pt1_distance = np.linalg.norm(centers - pt1)
    #     pt_distance = np.linalg.norm(pt0 - pt1)
    #     distance = pt0_distance + pt1_distance - pt_distance * 2
    #     return distance

    # @staticmethod
    # def optimize_with_progress(function, initial, args, iterations):
    #     results = []
    #     results.append(initial)
    #     result = None
    #     for i in range(iterations):
    #         result = minimize(function, x0=results[-1], args=args, options={'maxiter': 1})
    #         results.append(result.x)
    #     return result, results

    # @staticmethod
    # def optimize_test(input_img: Int2D_3C):
    #     centers = np.array(
    #         [[234, 719], [460, 718], [384, 717], [309, 717], [540, 714], [313, 608], [233, 608], [388, 606], [543, 605],
    #          [462, 603], [312, 502], [386, 500], [234, 500], [545, 495], [462, 495], [309, 388], [232, 387], [385, 387],
    #          [463, 384], [544, 383], [231, 279], [383, 277], [307, 276], [461, 275], [545, 273]])
    #     initial = np.array([[0, 0], [100, 0], [100, 100]])
    #     result, results = CardGridRecognizer.optimize_with_progress(CardGridRecognizer.grid_objective, initial, centers, 10)
    #
    #     result_img = CardGridRecognizer.draw_centers(input_img, centers, 20, np.array([0, 255, 0]))
    #     for i, iteration_result in enumerate(results):
    #         for j, point in enumerate(iteration_result.reshape([-1, 2])):
    #             point = (int(point[0]), int(point[1]))
    #             brightness = int((i / len(results)) * 255)
    #             color = [0, 0, 0]
    #             color[j] = brightness
    #             color = tuple(color)
    #
    #             cv2.circle(result_img, point, 10, color, -1)
    #     verbose_display(result_img)

    # @staticmethod
    # def optimize_test2(original_img: Int2D_3C):
    #     card_centers = np.array(
    #         [[234, 719], [460, 718], [384, 717], [309, 717], [540, 714], [313, 608], [233, 608], [388, 606], [543, 605],
    #          [462, 603], [312, 502], [386, 500], [234, 500], [545, 495], [462, 495], [309, 388], [232, 387], [385, 387],
    #          [463, 384], [544, 383], [231, 279], [383, 277], [307, 276], [461, 275], [545, 273]])
    #     count_img = np.zeros(original_img.shape, dtype=np.uint8)
    #     for i in range(len(card_centers)):
    #         for j in range(i + 1, len(card_centers)):
    #             center_i = card_centers[i]
    #             center_j = card_centers[j]
    #             center_offset = center_j - center_i
    #             for x in range(-1, 2):
    #                 new_center = center_i + x * center_offset
    #                 if new_center[0] >= 0 and new_center[1] >= 0 and new_center[0] < count_img.shape[0] and new_center[1] < count_img.shape[1]:
    #                     count_img[tuple(new_center)] += 1
    #     count_img *= 128
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     count_img = cv2.dilate(count_img, kernel, iterations=9)
    #     verbose_display(count_img)
