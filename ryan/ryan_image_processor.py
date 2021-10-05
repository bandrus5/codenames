import numpy as np
from cv2 import cv2
from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.image_functions import resize_image
from util.image_functions import draw_hough_line_segments


class RyanImageProcessor(ImageProcessorInterface):
    kernel = np.ones((3, 3), dtype=np.uint8)
    red_card_color = np.array([10, 100, 200])
    blue_card_color = np.array([140, 70, 0])
    normal_card_color = np.array([170, 170, 180])


    def extract_state_from_image(self, input_image: np.ndarray) -> Optional[GameState]:
        resized_img = resize_image(input_image, width=1000, height=1000)
        # gray_img = self._distance_from_color(resized_img, self.blue_card_color)
        gray_img = resized_img[:, :, 2]
        self.verbose_display(gray_img)
        blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
        self.verbose_display(blurred_img)
        # _, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 501, -5)
        self.verbose_display(threshold_img)
        erode_img = cv2.erode(threshold_img, None, iterations=3)
        self.verbose_display(erode_img)
        dilate_img = cv2.dilate(erode_img, None, iterations=3)
        self.verbose_display(dilate_img)
        edges_img = cv2.Canny(dilate_img, 25, 25, apertureSize=5)
        self.verbose_display(edges_img)
        contours = self._find_best_contours(dilate_img, 30, resized_img)

        # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=50, maxLineGap=5)
        # self.verbose_display(draw_hough_line_segments(resized, lines))


        # for contour in contours:
        #     perimeter = cv2.arcLength(contour, True)
        #     approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        return GameState(cards=None, key=None, first_turn=None)

    def _distance_from_color(self, input_img, color):
        h, w = input_img.shape[:2]
        hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        a = hsv_img - np.array([20, 0, 0])
        self.verbose_display(cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_HSV2BGR))
        b = hsv_img - np.array([0, 20, 0])
        self.verbose_display(cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_HSV2BGR))
        c = hsv_img - np.array([0, 0, 20])
        self.verbose_display(cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_HSV2BGR))
        hsv_color = cv2.cvtColor(color.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        distance_img = np.abs(hsv_img[:, :, 0].astype(np.float) - hsv_color[:, :, 0])
        # distance_img = np.linalg.norm((hsv_img[:, :, 0] - hsv_color[:, :, 0]).reshape(h, w, 1), axis=2)
        clipped_img = np.clip(distance_img, 0, 255).astype(np.uint8)
        return 255 - clipped_img

    def _find_best_contours(self, input_img: np.ndarray, keep_count: int, original_img: np.ndarray) -> List[np.ndarray]:
        contours = cv2.findContours(input_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=RyanImageProcessor._get_contour_card_similarity, reverse=True)
        if len(contours) <= keep_count:
            keep_count = len(contours)
        contours = contours[:keep_count]
        if self.is_verbose:
            contour_img = original_img.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            self.verbose_display(contour_img)
        return contours

    @staticmethod
    def _get_contour_card_similarity(contour) -> float:
        segment_count = contour.shape[0]
        contour_area = cv2.contourArea(contour)
        similarity = contour_area - segment_count
        return similarity

    def _get_dominant_colors(self, image):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(image.reshape(-1, 3).astype(np.float32), 5, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        if self.is_verbose:
            dominant_color_vis = RyanImageProcessor._visualize_dominant_colors(palette, counts)
            self.verbose_display(dominant_color_vis)

        return palette, counts

    @staticmethod
    def _visualize_dominant_colors(palette, counts):
        size = 64
        canvas = np.zeros((size, size * len(counts), 3), dtype=np.uint8)
        for i, palette_color in enumerate(palette):
            color = tuple(palette_color.astype(np.int32).tolist())
            pt1 = (i * size, 0)
            pt2 = ((i + 1) * size, size)
            cv2.rectangle(canvas, pt1, pt2, color, -1)
        return canvas

    def _quantitize_image(self, image):
        h, w = image.shape[:2]
        from sklearn.cluster import MiniBatchKMeans
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_vector = lab_img.reshape(-1, 3)
        clusters = MiniBatchKMeans(n_clusters=5)
        labels = clusters.fit_predict(lab_vector)
        quantization = clusters.cluster_centers_.astype("uint8")[labels]
        quantization_img = quantization.reshape((h, w, 3))
        quant_bgr_img = cv2.cvtColor(quantization_img, cv2.COLOR_LAB2BGR)

        self.verbose_display(quant_bgr_img)
        return quant_bgr_img




