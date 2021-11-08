import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.image_functions import display_image, save_image
from util.image_functions import draw_hough_line_segments
import math

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
        self._point_based_ransac(recomposed_bw)

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


    def _point_based_ransac(self, recomposed_bw):
        key_card_prototype = [(30, 197), (364, 197), (197, 364), (197, 30)] + [(row, col) for row in [90, 144, 197, 251, 305] for col in [90, 144, 197, 251, 305]]
        contours = cv2.findContours(recomposed_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        shape_centers = []
        for c in contours:
            m = cv2.moments(c)
            try:
                cX = int(m['m10'] / m['m00'])
                cY = int(m['m01'] / m['m00'])
            except ZeroDivisionError:
                continue
            shape_centers.append((cX, cY))
        # FIXME this function only works if the src and destination have the same number of points. Figure out how to wrap this with repeated random subsets to get the best possible match.
        ret = cv2.findHomography(np.array(key_card_prototype), np.array(shape_centers), method=cv2.RHO)
        print('Success')
        exit()



    def _find_key_grid_line_based_hough(self, input_image, recomposed, recomposed_bw):
        contours = cv2.findContours(recomposed_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        shape_centers = []
        for c in contours:
            m = cv2.moments(c)
            try:
                cX = int(m['m10'] / m['m00'])
                cY = int(m['m01'] / m['m00'])
            except ZeroDivisionError:
                continue
            shape_centers.append((cX, cY))
            cv2.drawContours(recomposed, [c], -1, (0, 255, 0), 2)
            cv2.circle(recomposed, (cX, cY), 7, (255, 255, 255), -1)
        if self.difficulty and self.background:
            self._save_image(recomposed, f'{self.background}{self.difficulty}contours')

        height, width = input_image.shape[:2]

        angle_votes = np.zeros(180)

        for shape_center1 in shape_centers:
            x1, y1 = shape_center1
            for shape_center2 in shape_centers:
                x2, y2 = shape_center2
                if (x1 + y1) > (x2 + y2) or (x1 == x2 and y1 == y2):
                    continue

                deltaX = x2 - x1
                deltaY = y2 - y1

                if abs(deltaX) > width / 20 or abs(deltaY) > height / 20:
                    continue

                if deltaX == 0:
                    angle_votes[89] += 1
                    angle_votes[90] += 3
                    angle_votes[91] += 1
                else:
                    projected_angle = round(math.degrees(math.atan2(deltaY, deltaX)))
                    angle_votes[(projected_angle - 1) % 180] += 1
                    angle_votes[projected_angle % 180] += 3
                    angle_votes[(projected_angle + 1) % 180] += 1

        first_alignment = np.argmax(angle_votes)
        for i in range(-40, 41):
            angle_votes[(first_alignment + i) % 180] = 0
        second_alignment = np.argmax(angle_votes)

        if first_alignment < second_alignment:
            second_alignment, first_alignment = first_alignment, second_alignment

        first_alignment_size_votes = np.zeros(min(width, height))
        second_alignment_size_votes = np.zeros(min(width, height))
        center_x_votes = np.zeros(width)
        center_y_votes = np.zeros(height)
        for shape_center1 in shape_centers:
            x1, y1 = shape_center1
            for shape_center2 in shape_centers:
                x2, y2 = shape_center2
                if (x1 + y1) > (x2 + y2) or (x1 == x2 and y1 == y2):
                    continue

                deltaX = x2 - x1
                deltaY = y2 - y1

                if abs(deltaX) > width / 20 or abs(deltaY) > height / 20:
                    continue
                if deltaX == 0:
                    projected_angle = 90
                else:
                    projected_angle = round(math.degrees(math.atan2(deltaY, deltaX)))
                for delta in range(-3, 4):
                    if (projected_angle + delta) % 180 == first_alignment:
                        distance = math.dist(shape_center1, shape_center2)
                        projected_card_size = round(distance * 7)
                        projected_center_x = round(np.mean([x1, x2]))
                        projected_center_y = round(np.mean([y1, y2]))
                        for i in range(-10, 11):
                            vote_magnitude = 10 - abs(i)
                            if 0 < projected_card_size + i < first_alignment_size_votes.shape[0]:
                                first_alignment_size_votes[projected_card_size + i] += vote_magnitude
                            if 0 < projected_center_x + i < width:
                                center_x_votes[projected_center_x + i] += vote_magnitude
                            if 0 < projected_center_y + i < height:
                                center_y_votes[projected_center_y + i] += vote_magnitude

                    if (projected_angle + delta) % 180 == second_alignment:
                        distance = math.dist(shape_center1, shape_center2)
                        projected_card_size = round(distance * 7)
                        projected_center_x = round(np.mean([x1, x2]))
                        projected_center_y = round(np.mean([y1, y2]))
                        for i in range(-10, 11):
                            vote_magnitude = 10 - abs(i)
                            if 0 < projected_card_size + i < second_alignment_size_votes.shape[0]:
                                second_alignment_size_votes[projected_card_size + i] += vote_magnitude
                            if 0 < projected_center_x + i < width:
                                center_x_votes[projected_center_x + i] += vote_magnitude
                            if 0 < projected_center_y + i < height:
                                center_y_votes[projected_center_y + i] += vote_magnitude

        first_alignment_size = np.argmax(first_alignment_size_votes)
        second_alignment_size = np.argmax(second_alignment_size_votes)

        ax = 500
        ay = 500
        bx = round(math.cos(math.radians(first_alignment)) * first_alignment_size) + ax
        by = round(math.sin(math.radians(first_alignment)) * first_alignment_size) + ay
        cx = round(math.cos(math.radians(second_alignment)) * second_alignment_size) + ax
        cy = round(math.sin(math.radians(second_alignment)) * second_alignment_size) + ay
        dx = bx + cx - ax
        dy = by + cy - ay

        current_center_x = round(np.mean([ax, dx]))
        current_center_y = round(np.mean([ay, dy]))

        x_shift = np.argmax(center_x_votes) - current_center_x
        y_shift = np.argmax(center_y_votes) - current_center_y

        ax += x_shift
        bx += x_shift
        cx += x_shift
        dx += x_shift
        ay += y_shift
        by += y_shift
        cy += y_shift
        dy += y_shift

        a, b, c, d = (ax, ay), (bx, by), (cx, cy), (dx, dy)

        cv2.line(input_image, a, b, (0, 255, 0), 4)
        cv2.line(input_image, b, d, (0, 255, 0), 4)
        cv2.line(input_image, c, d, (0, 255, 0), 4)
        cv2.line(input_image, c, a, (0, 255, 0), 4)
        cv2.circle(input_image, (np.argmax(center_x_votes), np.argmax(center_y_votes)), 7, (255, 255, 255), -1)
        for point, label in zip([a, b, c, d], ['A', 'B', 'C', 'D']):
            cv2.putText(input_image, label, point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        if self.difficulty and self.background:
            self._save_image(input_image, f'{self.background}{self.difficulty}card_projection')
