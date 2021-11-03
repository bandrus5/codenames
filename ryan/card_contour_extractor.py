from cv2 import cv2
from util.image_functions import verbose_display
import util.image_functions
import numpy as np
import matplotlib.pyplot as plt
from util.type_defs import *
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

ContourGroup = Tuple[List[Contour], FloatArray]


class CardContourExtractor:
    colors = [(0, 255, 0),
              (0, 0, 255),
              (255, 0, 0),
              (255, 0, 255),
              (0, 255, 255),
              (255, 255, 0),
              (0, 0, 128),
              (0, 128, 0),
              (128, 0, 0),
              (128, 0, 128),
              (0, 128, 128),
              (128, 128, 0)]
    _card_contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    # _card_contour = np.array([[[0, 0]], [[60, 0]], [[60, 94]], [[0, 95]]])
    # _card_contour = np.array([[[416, 559]], [[357, 561]], [[360, 655]], [[420, 650]]])

    @staticmethod
    def extract_card_contours(threshold_img: Int2D_1C, original_img: Int2D_3C) -> ContourGroup:
        contours = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [CardContourExtractor.simplify_contour(contour, 0.1) for contour in contours]
        contours = [c.reshape(-1, 2) for c in contours]
        contour_groups = CardContourExtractor._group_contours(contours)
        # Put group with cards at beginning
        card_group_index = CardContourExtractor._find_card_group(contour_groups)
        contour_groups[0], contour_groups[card_group_index] = contour_groups[card_group_index], contour_groups[0]
        card_group_index = 0
        if util.image_functions.is_verbose:
            CardContourExtractor._plot_group_stats(contour_groups)
            contour_img = original_img.copy()
            for i, (contour_group, group_stats) in enumerate(contour_groups):
                cv2.drawContours(contour_img, contour_group, -1, CardContourExtractor.colors[i], 2)
            verbose_display([contour_img])
        return contour_groups[0]

    @staticmethod
    def _find_card_group(contour_groups: List[ContourGroup]) -> int:
        largest_group_area = 0
        index_of_largest_group = -1
        for i, (contour_group, group_stats) in enumerate(contour_groups):
            group_area = 0
            for similarity, area in group_stats:
                group_area += area
            if group_area > largest_group_area:
                largest_group_area = group_area
                index_of_largest_group = i
        return index_of_largest_group

    @staticmethod
    def simplify_contour(contour: Contour, max_difference: float = 0.1) -> Contour:
        epsilon = max_difference * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        return simplified_contour

    @staticmethod
    def display_contours(input_img: Int2D_3C, contours: List[Contour], display_all_at_once: bool = True,
                         return_result: bool = False, number_points: bool = False, thickness: int = 2) -> Optional[Int2D_3C]:
        if display_all_at_once:
            background_img = input_img.copy()
            for i, contour in enumerate(contours):
                if not display_all_at_once:
                    background_img = input_img.copy()
                    line_count = contour.shape[0]
                    similarity = CardContourExtractor._get_contour_card_similarity(contour)
                    area = cv2.contourArea(contour)
                    print(f"{i} - Count: {line_count:3} Similarity: {similarity:.5f} Area: {area:.2f}")
                cv2.drawContours(background_img, [contour], -1, (0, 255, 0), thickness)
                if number_points:
                    for i, point in enumerate(contour):
                        cv2.putText(background_img, str(i), point, cv2.FONT_HERSHEY_PLAIN, 2, color=(0, 0, 255))
                if not display_all_at_once and not return_result:
                    verbose_display([background_img])
            if display_all_at_once and not return_result:
                verbose_display([background_img])
        if return_result:
            return background_img

    @staticmethod
    def _get_contour_card_similarity(contour: Contour) -> float:
        card_similarity = cv2.matchShapes(CardContourExtractor._card_contour, contour, cv2.CONTOURS_MATCH_I1, 0)
        return card_similarity

    @staticmethod
    def _group_contours(contours: List[Contour]) -> List[ContourGroup]:
        contours = list(filter(lambda x: x.shape[0] >= 4, contours))
        similarities = np.array(
            [1000 * CardContourExtractor._get_contour_card_similarity(contour) for contour in contours],
            dtype=np.float32)
        areas = np.array([cv2.contourArea(contour) for contour in contours], dtype=np.float32)
        data = np.stack([similarities, areas], axis=1)
        scaled_data = StandardScaler().fit_transform(data)

        result = DBSCAN(eps=1).fit(scaled_data)  # eps=1200
        labels = result.labels_

        contours_array = np.array(contours)
        contour_groups = []
        unique_labels = sorted(list(set(labels)))
        for i, label in enumerate(unique_labels):
            contour_group = contours_array[labels == label]
            group_stats = data[labels == label]
            contour_groups.append((contour_group, group_stats))
        return contour_groups

    @staticmethod
    def _plot_group_stats(contour_groups: List[ContourGroup]):
        for i, (contour_group, group_stats) in enumerate(contour_groups):
            color = CardContourExtractor.colors[i]
            color = (color[2] / 255, color[1] / 255, color[0] / 255)
            plt.scatter(group_stats[:, 0], group_stats[:, 1], c=color)
        plt.xlabel("Difference from card shape")
        plt.ylabel("Area")
        plt.show()

    @staticmethod
    def get_centroid(contour: Contour, as_tuple: bool = False, as_int: bool = True) -> Union[Tuple[int, int], IntArray]:
        m = cv2.moments(contour)
        x = m['m10']
        y = m['m01']
        m0 = m['m00']
        if m0 != 0:
            x /= m0
            y /= m0
        if as_int:
            x = int(x)
            y = int(y)

        if as_tuple:
            centroid = (x, y)
        else:
            centroid = np.array([x, y], dtype=np.int)

        return centroid
