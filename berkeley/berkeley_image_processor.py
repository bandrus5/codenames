import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.image_functions import display_image, save_image
from util.image_functions import draw_hough_line_segments
from tqdm import tqdm

class BerkeleyImageProcessor(ImageProcessorInterface):
    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        # red_image = self._simple_color_scale(input_image, 0)
        red_image_gray = self._simple_color_scale(input_image, 0, True)
        # blue_image = self._simple_color_scale(input_image, 2)
        blue_image_gray = self._simple_color_scale(input_image, 2, True)

        display_image(input_image)
        # self._save_image(input_image, '13truth')
        # display_image(red_image)
        display_image(red_image_gray)
        # self._save_image(red_image_gray, '13red')
        # display_image(blue_image)
        display_image(blue_image_gray)
        # self._save_image(blue_image_gray, '13blue')

        return GameState(cards=None, key=None, first_turn=None)

    def _save_image(self, image, fn):
        save_image(image, f'berkeley/results/{fn}.png')

    def _simple_color_scale(self, image, color_index, grayscale=False):
        new_image = np.copy(image)
        means = np.round(np.mean(new_image, axis=2))

        for w in tqdm(range(image.shape[0])):
            for h in range(image.shape[1]):
                new_val = new_image[w][h][color_index] - means[w][h]
                new_image[w][h][color_index] = max(0, new_val)
                for v in range(image.shape[2]):
                    if v != color_index:
                        new_image[w][h][v] = new_image[w][h][color_index] if grayscale else 0
        return new_image
