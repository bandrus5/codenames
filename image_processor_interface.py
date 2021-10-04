import numpy as np
from data_labelling.game_state import *


class ImageProcessorInterface:
    def extract_state_from_image(self, input_image: np.ndarray) -> GameState:
        pass
