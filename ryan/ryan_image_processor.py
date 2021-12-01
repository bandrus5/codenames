import numpy as np
import numpy.typing as npt
from cv2 import cv2
import os
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from ryan.foreground_extractor import ForegroundExtractor
from ryan.color_extractor import  ColorExtractor
from ryan.card_identifier import CardIdentifier
from ryan.card_quad_refiner import CardQuadRefiner
from ryan.card_contour_extractor import CardContourExtractor
from ryan.color_corrector import  ColorCorrector
from ryan.card_grid_recognizer import CardGridRecognizer
from ryan.card_cropper import CardCropper
from util.image_functions import resize_image
from util.image_functions import *
from util.type_defs import *


class RyanImageProcessor(ImageProcessorInterface):
    kernel = np.ones((3, 3), dtype=np.uint8)
    # red_card_color = np.array([15, 50, 180])
    red_card_color = np.array([0, 0, 255])
    blue_card_color = np.array([100, 60, 30])
    # blue_card_color = np.array([255, 0, 0])
    # yellow_card_color = np.array([130, 160, 190])
    yellow_card_color = np.array([180, 210, 230])
    white_card_color = np.array([255, 255, 255])

    def __init__(self, flags):
        CardIdentifier.init(flags.tesseract_path)

    # yellow_card_color = np.array([0, 255, 255])

    def extract_state_from_image(self, input_image: Int2D_3C) -> Optional[GameState]:
        verbose_save("input", input_image, is_first_save=True)
        resized_size = 1000
        resized_img = resize_image(input_image, width=resized_size, height=resized_size)
        verbose_save("resized", resized_img)
        blurred_img = cv2.GaussianBlur(resized_img, (23, 23), 0)
        verbose_save("blurred", blurred_img)
        # mask, masked_img = ForegroundExtractor.extract_foreground(blurred_img, 20)
        # mask = None
        # masked_img = blurred_img
        # balanced_img, _, histogram = ColorCorrector.correct_color(masked_img, True, mask)
        # balanced_hist = ColorCorrector.calc_channel_histograms(balanced_img, mask)
        # verbose_display([masked_img, draw_histogram(histogram), balanced_img, draw_histogram(balanced_hist)], 512)

        balanced_img = blurred_img
        red_img = ColorExtractor.extract_color(balanced_img, self.red_card_color, threshold=200)
        blue_img = ColorExtractor.extract_color(balanced_img, self.blue_card_color, threshold=200)
        yellow_img = ColorExtractor.extract_color(balanced_img, self.white_card_color, threshold=150, use_only_value=True)
        verbose_display([red_img, blue_img, yellow_img, balanced_img], 512)

        threshold_img = cv2.bitwise_or(yellow_img, blue_img)
        verbose_save("combined", threshold_img)
        # card_group_contours, card_group_stats = CardContourExtractor.extract_card_contours(threshold_img, resized_img)
        cards_quads = CardGridRecognizer.get_cards_quads(threshold_img, resized_img)
        refined_cards_quads = CardQuadRefiner.refine_cards_quads(cards_quads, resized_img)
        cropped_cards = CardCropper.crop_cards(refined_cards_quads, resized_img)
        for cropped_card in cropped_cards:
            card_value = CardIdentifier.identify_card(cropped_card, RyanImageProcessor.red_card_color, RyanImageProcessor.blue_card_color, RyanImageProcessor.yellow_card_color)
        self._display_cards(cropped_cards)


        return GameState(cards=None, key=None, first_turn=None)

    def _display_cards(self, cropped_cards: List[Int2D_3C]):
        bordered_cards = []
        for cropped_card in cropped_cards:
            cropped_card = cv2.copyMakeBorder(cropped_card, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            bordered_cards.append(cropped_card)
        combined_img = verbose_display(bordered_cards,display_size=100)
        verbose_save("cards", combined_img)

    def _filter_color(self, color_img: Int2D_1C, original_img: Int2D_3C):
        erode_img = cv2.erode(color_img, None, iterations=3)
        dilate_img = cv2.dilate(erode_img, None, iterations=3)
        return dilate_img







    # def _get_dominant_colors(self, image):
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    #     flags = cv2.KMEANS_RANDOM_CENTERS
    #     _, labels, palette = cv2.kmeans(image.reshape(-1, 3).astype(np.float32), 5, None, criteria, 10, flags)
    #     _, counts = np.unique(labels, return_counts=True)
    #
    #     if self.is_verbose:
    #         dominant_color_vis = RyanImageProcessor._visualize_dominant_colors(palette, counts)
    #         self.verbose_display(dominant_color_vis)
    #
    #     return palette, counts
    #
    # @staticmethod
    # def _visualize_dominant_colors(palette, counts):
    #     size = 64
    #     canvas = np.zeros((size, size * len(counts), 3), dtype=np.uint8)
    #     for i, palette_color in enumerate(palette):
    #         color = tuple(palette_color.astype(np.int32).tolist())
    #         pt1 = (i * size, 0)
    #         pt2 = ((i + 1) * size, size)
    #         cv2.rectangle(canvas, pt1, pt2, color, -1)
    #     return canvas
    #
    # def _quantitize_image(self, image):
    #     h, w = image.shape[:2]
    #     from sklearn.cluster import MiniBatchKMeans
    #     lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #     lab_vector = lab_img.reshape(-1, 3)
    #     clusters = MiniBatchKMeans(n_clusters=5)
    #     labels = clusters.fit_predict(lab_vector)
    #     quantization = clusters.cluster_centers_.astype("uint8")[labels]
    #     quantization_img = quantization.reshape((h, w, 3))
    #     quant_bgr_img = cv2.cvtColor(quantization_img, cv2.COLOR_LAB2BGR)
    #
    #     verbose_display(quant_bgr_img)
    #     return quant_bgr_img