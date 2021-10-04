import numpy as np
from data_labelling.game_state import *
from util.image_functions import display_image


class ImageProcessorInterface:
    def __init__(self, is_verbose: bool = False, verbose_display_size: int = 1000):
        self.is_verbose = is_verbose
        self.verbose_display_size = verbose_display_size

    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        pass

    def verbose_display(self, image):
        if self.is_verbose:
            display_image(image, self.verbose_display_size)
