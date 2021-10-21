import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.image_functions import display_image, save_image
from util.image_functions import draw_hough_line_segments

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
            self._save_image(recomposed, f'{self.background}{self.difficulty}recomposed')

        recomposed_bw = self._recompose(input_image, red_image_eroded, blue_image_eroded, bw=True)

        self._find_key_grid(input_image, recomposed, recomposed_bw)

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

    def _find_key_grid(self, input_image, recomposed, recomposed_bw):
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
        votes = np.zeros((height, width))
        for shape_center1 in shape_centers:
            x1, y1 = shape_center1
            for shape_center2 in shape_centers:
                x2, y2 = shape_center2
                if (x1 + y1) <= (x2 + y2): # Don't duplicate work
                    continue
                if abs(x2 - x1) > width / 5 or abs(y2 - y1) > height / 5:
                    continue

                deltaX = x2 - x1
                deltaY = y2 - y1

                for i in range(4):
                    _deltaX = deltaX * i
                    _deltaY = deltaY * i

                    if 0 < x2 + _deltaX < width and 0 < y2 + _deltaY < height:
                        votes[y2 + _deltaY][x2 + _deltaX] += 3
                        for j in range(-10, 11):
                            for k in range(-10, 11):
                                if 0 < x2 + _deltaX + j < width and 0 < y2 + _deltaY + k < height:
                                    votes[y2 + _deltaY + k][x2 + _deltaX + j] += 1

                    if 0 < x1 - _deltaX < width and 0 < y1 - _deltaY < height:
                        votes[y1 - _deltaY][x1 - _deltaX] += 3
                        for j in range(-10, 11):
                            for k in range(-10, 11):
                                if 0 < x1 - _deltaX + j < width and 0 < y1 - _deltaY + k < height:
                                    votes[y1 - _deltaY + k][x1 - _deltaX + j] += 1

                perpDeltaX = deltaY
                perpDeltaY = -deltaX

                for i in [-4, -3, -2, -1, 1, 2, 3, 4]:
                    _deltaX = perpDeltaX * i
                    _deltaY = perpDeltaY * i

                    if 0 < x2 + _deltaX < width and 0 < y2 + _deltaY < height:
                        votes[y2 + _deltaY][x2 + _deltaX] += 3
                        for j in range(-10, 11):
                            for k in range(-10, 11):
                                if 0 < x2 + _deltaX + j < width and 0 < y2 + _deltaY + k < height:
                                    votes[y2 + _deltaY + k][x2 + _deltaX + j] += 1

                    if 0 < x1 + _deltaX < width and 0 < y1 + _deltaY < height:
                        votes[y1 + _deltaY][x1 + _deltaX] += 3
                        for j in range(-10, 11):
                            for k in range(-10, 11):
                                if 0 < x1 + _deltaX + j < width and 0 < y1 + _deltaY + k < height:
                                    votes[y1 + _deltaY + k][x1 + _deltaX + j] += 1


        votes_image = np.zeros((height, width, 3))
        votes_image[:,:,0] = (votes > 20) * 255
        votes_image[:,:,1] = (votes > 20) * 255
        votes_image[:,:,2] = (votes > 20) * 255

        if self.difficulty and self.background:
            self._save_image(votes_image, f'{self.background}{self.difficulty}votes')
