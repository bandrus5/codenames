import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *


class RyanImageProcessor(ImageProcessorInterface):
    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        gray = input_image[:,:,2] # cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        self.verbose_display(gray)
        blurred = cv2.GaussianBlur(gray, (91, 91), 0)
        self.verbose_display(blurred)
        return GameState(cards=None, key=None, first_turn=None)




