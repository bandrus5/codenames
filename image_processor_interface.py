import numpy as np
from data_labelling.game_state import *
from util.image_functions import display_image


class ImageProcessorInterface:
    def __init__(self, is_verbose: bool = False, verbose_display_size: int = 1000):
        self.is_verbose = is_verbose
        self.verbose_display_size = verbose_display_size

    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        pass

    def verbose_display(self, images_to_display: List[np.ndarray], override_display_size: Optional[int] = None):
        if self.is_verbose:
            reshaped_imgs = []
            for image in images_to_display:
                reshaped = image.reshape((image.shape[0], image.shape[1], -1))
                if reshaped.shape[2] < 3:
                    reshaped = np.concatenate([reshaped] * 3, axis=2)
                if reshaped.dtype != np.uint8:
                    reshaped = (reshaped * 255).astype(np.uint8)
                reshaped_imgs.append(reshaped)
            combined_img = np.concatenate(reshaped_imgs, axis=1)
            if override_display_size is not None:
                display_size = override_display_size
            else:
                display_size = self.verbose_display_size
            display_image(combined_img, width=display_size * len(images_to_display))
