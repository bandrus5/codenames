import numpy as np
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *


class RyanImageProcessor(ImageProcessorInterface):
    def extract_state_from_image(self, input_image: np.ndarray) -> GameState:
        pass

