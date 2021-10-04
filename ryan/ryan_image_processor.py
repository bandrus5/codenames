import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.image_functions import display_image


class RyanImageProcessor(ImageProcessorInterface):
    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        return GameState(cards=None, key=None, first_turn=None)




