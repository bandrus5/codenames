import os, sys
import numpy as np
import cv2
import time
from imutils.object_detection import non_max_suppression
from util.type_defs import *
from util.image_functions import verbose_display
from berkeley.word_validator import WordValidator
import pytesseract


class TextRecognizer:
    def __init__(self, tesseract_path):
        if tesseract_path is not None and tesseract_path != "":
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        self.word_validator = WordValidator()

    # Somehow I found the value of `gamma=0.4` to be the best in my case
    @staticmethod
    def _adjust_gamma(image, gamma=0.4):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    # noise removal
    @staticmethod
    def _remove_noise(image):
        return cv2.medianBlur(image, 5)

    def _detect_text_regions(self, image):
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        orig = image.copy()

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height: Should be multiple of 32
        (newW, newH) = (320, 320)

        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))

        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)

        start = time.time()

        self.net.setInput(blob)

        (scores, geometry) = self.net.forward(layerNames)

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                # Set minimum confidence as required
                if scoresData[x] < 0.5:
                    continue
                # compute the offset factor as our resulting feature maps will
                #  x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            print(startX, startY, endX, endY)

            # draw the bounding box on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        print(time.time() - start)
        return boxes

    def _read_potential_words(self, card_img: Int2D_3C) -> List[str]:
        image_copy = card_img.copy()
        verbose_display(image_copy)

        image_copy = image_copy.astype(np.uint8)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        image_copy = 255 - cv2.adaptiveThreshold(image_copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21,
                                                 10)
        image_copy = TextRecognizer._adjust_gamma(image_copy)
        image_copy = TextRecognizer._remove_noise(image_copy)

        boxes = self._detect_text_regions(image_copy)
        image_copy = TextRecognizer._remove_noise(image_copy)

        thresh = 255 - cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        if len(image_copy.shape) == 2:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        (H, W) = image_copy.shape[:2]
        (newW, newH) = (320, 320)

        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image_copy = cv2.resize(image_copy, (newW, newH))
        (H, W) = image_copy.shape[:2]

        card_words = []

        for (x, y, w, h) in boxes:
            x = int(x * rW)
            y = int(y * rH)
            w = int(w * rW) + 10
            h = int(h * rH) + 10
            print("==========")
            ROI = thresh[y:h, x:w]
            if ROI.shape[0] > 0 and ROI.shape[1] > 0:
                data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
                print(data)
                card_words.append(data)
                verbose_display(ROI)

        return card_words

    def read_card(self, card_img: Int2D_3C) -> str:
        potential_words = self._read_potential_words(card_img)
        best_word = None
        best_score = float('inf')
        for potential_word in potential_words:
            words, score = self.word_validator.autocorrect_word(potential_word)
            if score < best_score and len(words) > 0:
                best_score = score
                best_word = words[0]
        return best_word



# # 1756 HOLE
# # 1758 LIMOUSINE
# # 1765 PASS
# # 1767 HORSESHOE
# # 1774 ROBIN
#
# image = cv2.imread("IMG_1778.jpeg", 0)
# image_copy = image.copy()
# cv2.imshow('image', image)
# cv2.waitKey(0)
# # maybe do a softer threshold operation before gamma adjustment
# image_copy = 255 - cv2.threshold(image_copy, 160, 255, cv2.THRESH_BINARY_INV)[1]
# image_copy = adjust_gamma(image_copy)
# image_copy = 255 - cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# image_copy = remove_noise(image_copy)
#
#
#
#
#
#
#
#
# cv2.imshow('image', image_copy)
# cv2.waitKey(0)
#
# boxes = east_detect(image_copy)
# print(boxes)
# image = remove_noise(image)
#
# #cv2.imshow('img', image)
# #cv2.waitKey(0)
#
# #out_image = east_detect(image)
#
# thresh = 255 - cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
#
#
# if len(image.shape) == 2:
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
# (H, W) = image.shape[:2]
# (newW, newH) = (320, 320)
#
# rW = W / float(newW)
# rH = H / float(newH)
#
# print("rW and rH:")
# print(rW, rH)
#
# # resize the image and grab the new image dimensions
# image = cv2.resize(image, (newW, newH))
# (H, W) = image.shape[:2]
#
#
#
# (x,y,w,h) = boxes[0]
# x = int(x * rW)
# y = int(y * rH)
# w = int(w * rW) + 50
# h = int(h * rH) + 50
# print("x,y,w,h")
# print(x,y,w,h)
# print("=====")
# ROI  = thresh[y:h, x:w]
# #ROI = thresh[1:850, 1:850]
# cv2.imshow('ROI', ROI)
# cv2.waitKey(0)
#
# data = pytesseract.image_to_string(ROI, lang = 'eng', config = '--psm 6')
# print(data)
# #im = pytesseract.image_to_data(ROI, lang = 'eng', config = '--psm 6')
# #print(im)
#
# #cv2.imwrite("sample_output.jpg", out_image)
