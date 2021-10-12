import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.image_functions import display_image, save_image
from util.image_functions import draw_hough_line_segments
from tqdm import tqdm

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
        red_image_eroded = cv2.erode(red_image_open, self.kernel, iterations = 2)
        blue_image_gray = self._simple_color_scale(input_image, 2, True)
        blue_image_gray_hc = cv2.convertScaleAbs(blue_image_gray, alpha=4.0)
        _, blue_image_threshold = cv2.threshold(blue_image_gray_hc, 150, 255, cv2.THRESH_BINARY)
        blue_image_open = cv2.morphologyEx(blue_image_threshold, cv2.MORPH_OPEN, self.kernel)
        blue_image_eroded = cv2.erode(blue_image_open, self.kernel, iterations = 2)

        recomposed = self._recompose(input_image, red_image_eroded, blue_image_eroded)

        if self.difficulty and self.background:
            print('Saving to', self.difficulty, self.background)
            self._save_image(input_image, f'{self.background}{self.difficulty}truth')
            # self._save_image(red_image_threshold, f'{self.background}{self.difficulty}red_thresh')
            # self._save_image(blue_image_threshold, f'{self.background}{self.difficulty}blue_thresh')
            # self._save_image(red_image_open, f'{self.background}{self.difficulty}red_open')
            # self._save_image(blue_image_open, f'{self.background}{self.difficulty}blue_open')
            # self._save_image(red_image_eroded, f'{self.background}{self.difficulty}red_erode')
            # self._save_image(blue_image_eroded, f'{self.background}{self.difficulty}blue_erode')
            self._save_image(recomposed, f'{self.background}{self.difficulty}recomposed')
        else:
            print('Not saving to', self.difficulty, self.background)

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

    def _recompose(self, image, red_image, blue_image):
        new_image = np.copy(image)
        new_image[:, :, 0] = red_image[:,:,0]
        new_image[:, :, 1] = 0
        new_image[:, :, 2] = blue_image[:,:,2]
        return new_image
