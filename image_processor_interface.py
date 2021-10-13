import cv2
import numpy as np
from data_labelling.game_state import *
from util.image_functions import display_image
from util.image_functions import combine_images


class ImageProcessorInterface:
    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        pass
