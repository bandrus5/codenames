import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.image_functions import display_image, save_image
from util.image_functions import draw_hough_line_segments
import math

def cartesian_to_homogeneous(point):
    return np.array([point[0], point[1], 1])

def homogeneous_to_cartesian(point):
    return (round(point[0]), round(point[1]))

class BerkeleyImageProcessor(ImageProcessorInterface):

    def __init__(self, flags=None):
        if flags:
            try:
                self.difficulty = int(flags.difficulty)
                self.background = int(flags.background)
            except:
                pass
        self.kernel = np.ones((5,5),np.uint8)

    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        input_image = cv2.resize(input_image, (0,0), fx = 0.5, fy = 0.5)
        red_image_gray = self._simple_color_scale(input_image, 0, True)
        red_image_gray_hc = cv2.convertScaleAbs(red_image_gray, alpha=4.0)
        _, red_image_threshold = cv2.threshold(red_image_gray_hc, 150, 255, cv2.THRESH_BINARY)
        red_image_open = cv2.morphologyEx(red_image_threshold, cv2.MORPH_OPEN, self.kernel)
        red_image_eroded = cv2.erode(red_image_open, self.kernel, iterations = 4)
        blue_image_gray = self._simple_color_scale(input_image, 2, True)
        blue_image_gray_hc = cv2.convertScaleAbs(blue_image_gray, alpha=4.0)
        _, blue_image_threshold = cv2.threshold(blue_image_gray_hc, 150, 255, cv2.THRESH_BINARY)
        blue_image_open = cv2.morphologyEx(blue_image_threshold, cv2.MORPH_OPEN, self.kernel)
        blue_image_eroded = cv2.erode(blue_image_open, self.kernel, iterations = 4)

        recomposed = self._recompose(input_image, red_image_eroded, blue_image_eroded)

        if self.difficulty and self.background:
            self._save_image(input_image, f'{self.background}{self.difficulty}truth')
            # self._save_image(recomposed, f'{self.background}{self.difficulty}recomposed')

        recomposed_bw = self._recompose(input_image, red_image_eroded, blue_image_eroded, bw=True)
        # self._find_key_grid_line_based_hough(input_image, recomposed, recomposed_bw)
        self._point_based_ransac(input_image, recomposed_bw)

        return GameState(cards=None, key=None, first_turn=None)

    def _save_image(self, image, fn):
        save_image(image, f'berkeley/results/{fn}.png')

    def _simple_color_scale(self, image, color_index, grayscale=False):
        new_image = np.copy(image)
        means = np.round(np.mean(new_image, axis=2))

        new_image[:, :, color_index] = np.maximum(0, new_image[:, :, color_index] - means)

        for i in range(image.shape[2]):
            if i == color_index:
                continue
            new_image[:, :, i] = new_image[:, :, color_index] if grayscale else 0

        return new_image

    def _recompose(self, image, red_image, blue_image, bw=False):
        new_image = np.copy(image)
        new_image[:,:,0] = red_image[:,:,0]
        new_image[:,:,1] = 0
        new_image[:,:,2] = blue_image[:,:,2]

        if bw:
            new_image[:,:,0] = np.maximum(new_image[:,:,0],new_image[:,:,2])
            new_image[:,:,1] = np.maximum(new_image[:,:,0],new_image[:,:,2])
            new_image[:,:,2] = np.maximum(new_image[:,:,0],new_image[:,:,2])
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        return new_image

    def _has_similar_transform(self, src1, dest1, src2, dest2, thresh):
        for ind in [0, 1]:
            dist1 = dest1[ind] - src1[ind]
            dist2 = dest2[ind] - src2[ind]
            if abs(dist2 - dist1) > thresh[ind]:
                return False
        return True

    def _distance_within_threshold(self, point1, point2, thresh):
        for ind in [0, 1]:
            if abs(point1[ind] - point2[ind]) > thresh[ind]:
                return False
        return True

    def _score_homography(self, homography, prototype_points, image_points, thresh):
        score = 0
        for point in prototype_points:
            homogeneous_point = cartesian_to_homogeneous(point)
            translated_point = np.matmul(homography, homogeneous_point)
            if np.isnan(np.sum(translated_point)):
                continue
            for point2 in image_points:
                if self._distance_within_threshold(homogeneous_to_cartesian(translated_point), point2, thresh):
                    score += 1
                    continue
        return score

    def _point_based_ransac(self, input_image, recomposed_bw):
        input_image = np.copy(input_image)
        key_card_prototype = [(30, 197), (364, 197), (197, 364), (197, 30)] + [(row, col) for row in [90, 144, 197, 251, 305] for col in [90, 144, 197, 251, 305]]
        contours = cv2.findContours(recomposed_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        shape_centers = []
        height, width = recomposed_bw.shape[:2]
        new_visualization = np.zeros((height, width, 3))
        for c in contours:
            m = cv2.moments(c)
            try:
                cX = int(m['m10'] / m['m00'])
                cY = int(m['m01'] / m['m00'])
            except ZeroDivisionError:
                continue
            shape_centers.append((cX, cY))
            cv2.circle(new_visualization, (cX, cY), 7, (255, 255, 255), -1)


        x_match_thresh = math.ceil(width * 0.0025)
        y_match_thresh = math.ceil(height * 0.0025)

        x_dist_thresh = math.ceil(width * 0.1)
        y_dist_thresh = math.ceil(height * 0.1)

        parallelograms = []
        for A in shape_centers:
            for B in shape_centers:
                if B == A or B[1] < A[1]:
                    continue
                for C in shape_centers:
                    if C in [A, B] or not self._distance_within_threshold(A, C, (x_dist_thresh, y_dist_thresh)):
                        continue
                    for D in shape_centers:
                        if D in [A, B, C] or D[0] < A[0]:
                            continue
                        if self._has_similar_transform(A, B, D, C, (x_match_thresh, y_match_thresh)) and self._has_similar_transform(A, D, B, C, (x_match_thresh, y_match_thresh)):
                            parallelograms.append((A, B, C, D))

        prototype_axis_aligned_rectangles = [((90, 90), (90, 144), (144, 144), (144, 90)), ((90, 90), (90, 144), (197, 144), (197, 90)), ((90, 90), (90, 144), (251, 144), (251, 90)), ((90, 90), (90, 144), (305, 144), (305, 90)), ((90, 90), (90, 197), (144, 197), (144, 90)), ((90, 90), (90, 197), (197, 197), (197, 90)), ((90, 90), (90, 197), (251, 197), (251, 90)), ((90, 90), (90, 197), (305, 197), (305, 90)), ((90, 90), (90, 251), (144, 251), (144, 90)), ((90, 90), (90, 251), (197, 251), (197, 90)), ((90, 90), (90, 251), (251, 251), (251, 90)), ((90, 90), (90, 251), (305, 251), (305, 90)), ((90, 90), (90, 305), (144, 305), (144, 90)), ((90, 90), (90, 305), (197, 305), (197, 90)), ((90, 90), (90, 305), (251, 305), (251, 90)), ((90, 90), (90, 305), (305, 305), (305, 90)), ((90, 144), (90, 197), (144, 197), (144, 144)), ((90, 144), (90, 197), (197, 197), (197, 144)), ((90, 144), (90, 197), (251, 197), (251, 144)), ((90, 144), (90, 197), (305, 197), (305, 144)), ((90, 144), (90, 251), (144, 251), (144, 144)), ((90, 144), (90, 251), (197, 251), (197, 144)), ((90, 144), (90, 251), (251, 251), (251, 144)), ((90, 144), (90, 251), (305, 251), (305, 144)), ((90, 144), (90, 305), (144, 305), (144, 144)), ((90, 144), (90, 305), (197, 305), (197, 144)), ((90, 144), (90, 305), (251, 305), (251, 144)), ((90, 144), (90, 305), (305, 305), (305, 144)), ((90, 197), (90, 251), (144, 251), (144, 197)), ((90, 197), (90, 251), (197, 251), (197, 197)), ((90, 197), (90, 251), (251, 251), (251, 197)), ((90, 197), (90, 251), (305, 251), (305, 197)), ((90, 197), (90, 305), (144, 305), (144, 197)), ((90, 197), (90, 305), (197, 305), (197, 197)), ((90, 197), (90, 305), (251, 305), (251, 197)), ((90, 197), (90, 305), (305, 305), (305, 197)), ((90, 251), (90, 305), (144, 305), (144, 251)), ((90, 251), (90, 305), (197, 305), (197, 251)), ((90, 251), (90, 305), (251, 305), (251, 251)), ((90, 251), (90, 305), (305, 305), (305, 251)), ((144, 90), (144, 144), (197, 144), (197, 90)), ((144, 90), (144, 144), (251, 144), (251, 90)), ((144, 90), (144, 144), (305, 144), (305, 90)), ((144, 90), (144, 197), (197, 197), (197, 90)), ((144, 90), (144, 197), (251, 197), (251, 90)), ((144, 90), (144, 197), (305, 197), (305, 90)), ((144, 90), (144, 251), (197, 251), (197, 90)), ((144, 90), (144, 251), (251, 251), (251, 90)), ((144, 90), (144, 251), (305, 251), (305, 90)), ((144, 90), (144, 305), (197, 305), (197, 90)), ((144, 90), (144, 305), (251, 305), (251, 90)), ((144, 90), (144, 305), (305, 305), (305, 90)), ((144, 144), (144, 197), (197, 197), (197, 144)), ((144, 144), (144, 197), (251, 197), (251, 144)), ((144, 144), (144, 197), (305, 197), (305, 144)), ((144, 144), (144, 251), (197, 251), (197, 144)), ((144, 144), (144, 251), (251, 251), (251, 144)), ((144, 144), (144, 251), (305, 251), (305, 144)), ((144, 144), (144, 305), (197, 305), (197, 144)), ((144, 144), (144, 305), (251, 305), (251, 144)), ((144, 144), (144, 305), (305, 305), (305, 144)), ((144, 197), (144, 251), (197, 251), (197, 197)), ((144, 197), (144, 251), (251, 251), (251, 197)), ((144, 197), (144, 251), (305, 251), (305, 197)), ((144, 197), (144, 305), (197, 305), (197, 197)), ((144, 197), (144, 305), (251, 305), (251, 197)), ((144, 197), (144, 305), (305, 305), (305, 197)), ((144, 251), (144, 305), (197, 305), (197, 251)), ((144, 251), (144, 305), (251, 305), (251, 251)), ((144, 251), (144, 305), (305, 305), (305, 251)), ((197, 90), (197, 144), (251, 144), (251, 90)), ((197, 90), (197, 144), (305, 144), (305, 90)), ((197, 90), (197, 197), (251, 197), (251, 90)), ((197, 90), (197, 197), (305, 197), (305, 90)), ((197, 90), (197, 251), (251, 251), (251, 90)), ((197, 90), (197, 251), (305, 251), (305, 90)), ((197, 90), (197, 305), (251, 305), (251, 90)), ((197, 90), (197, 305), (305, 305), (305, 90)), ((197, 144), (197, 197), (251, 197), (251, 144)), ((197, 144), (197, 197), (305, 197), (305, 144)), ((197, 144), (197, 251), (251, 251), (251, 144)), ((197, 144), (197, 251), (305, 251), (305, 144)), ((197, 144), (197, 305), (251, 305), (251, 144)), ((197, 144), (197, 305), (305, 305), (305, 144)), ((197, 197), (197, 251), (251, 251), (251, 197)), ((197, 197), (197, 251), (305, 251), (305, 197)), ((197, 197), (197, 305), (251, 305), (251, 197)), ((197, 197), (197, 305), (305, 305), (305, 197)), ((197, 251), (197, 305), (251, 305), (251, 251)), ((197, 251), (197, 305), (305, 305), (305, 251)), ((251, 90), (251, 144), (305, 144), (305, 90)), ((251, 90), (251, 197), (305, 197), (305, 90)), ((251, 90), (251, 251), (305, 251), (305, 90)), ((251, 90), (251, 305), (305, 305), (305, 90)), ((251, 144), (251, 197), (305, 197), (305, 144)), ((251, 144), (251, 251), (305, 251), (305, 144)), ((251, 144), (251, 305), (305, 305), (305, 144)), ((251, 197), (251, 251), (305, 251), (305, 197)), ((251, 197), (251, 305), (305, 305), (305, 197)), ((251, 251), (251, 305), (305, 305), (305, 251)), ((30, 197), (364, 197), (197, 364), (197, 30))]

        # for A in key_card_prototype:
        #     for B in key_card_prototype:
        #         if B == A or B[1] < A[1]:
        #             continue
        #         for C in key_card_prototype:
        #             if C in [A, B]:
        #                 continue
        #             for D in key_card_prototype:
        #                 if D in [A, B, C] or D[0] < A[0]:
        #                     continue
        #                 # if self._has_similar_transform(A, B, D, C, (1, 1)) and self._has_similar_transform(A, D, B, C, (1, 1)):
        #                 #     prototype_parallelograms.append((A, B, C, D))
        #                 if A[0] == B[0] and C[0] == D[0] and A[1] == D[1] and B[1] == C[1]:
        #                     prototype_axis_aligned_rectangles.append((A, B, C, D))

        best_homography = None
        best_score = -1
        for oA, oB, oC, oD in parallelograms:
            for pA, pB, pC, pD in prototype_axis_aligned_rectangles:
                homography = cv2.findHomography(np.array([pA, pB, pC, pD]), np.array([oA, oB, oC, oD]), method=cv2.RHO)[0]
                if homography is None or homography.size == 0:
                    continue
                score = self._score_homography(homography, key_card_prototype, shape_centers, (x_match_thresh, y_match_thresh))
                if score > best_score:
                    best_score = score
                    best_homography = homography
        print(best_score)


        if best_score > -1:
            for point in key_card_prototype:
                translated_point = homogeneous_to_cartesian(np.matmul(best_homography, cartesian_to_homogeneous(point)))
                cv2.circle(input_image, translated_point, 7, (0, 255, 0), -1)

            if self.difficulty and self.background:
                self._save_image(input_image, f'{self.background}{self.difficulty}predicted_location')
