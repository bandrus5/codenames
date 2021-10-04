import numpy as np
from data_labelling.game_state import *


class ProcessorInterface:
    def process(self, input_image: np.ndarray) -> GameState:
        pass
