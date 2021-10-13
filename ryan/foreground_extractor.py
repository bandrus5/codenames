import numpy as np
from typing import *
from cv2 import cv2


class ForegroundExtractor:
    @staticmethod
    def extract_foreground(input_img: np.ndarray, background_margin: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = input_img.shape[:2]
        mask = np.zeros(input_img.shape[:2], dtype=np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        rect = (background_margin, background_margin, w - 2 * background_margin, h - 2 * background_margin)
        cv2.grabCut(input_img, mask, rect, bg_model, fg_model, 1, cv2.GC_INIT_WITH_RECT)
        mask_img = np.where(mask == cv2.GC_PR_FGD, 1, 0).astype(np.uint8)
        masked_img = input_img * np.concatenate([mask_img[..., np.newaxis]] * 3, axis=2).astype(np.uint8)
        return mask_img, masked_img
