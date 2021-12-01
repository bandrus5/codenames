from cv2 import cv2
import numpy as np
from util.type_defs import *
from ryan.color_extractor import ColorExtractor
from util.image_functions import verbose_display

import os, sys
import cv2
import time
from imutils.object_detection import non_max_suppression
import pytesseract

# Somehow I found the value of `gamma=0.4` to be the best in my case
def adjust_gamma(image, gamma=0.4):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

def east_detect(image):
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
    
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    start = time.time()
    
    net.setInput(blob)
    
    (scores, geometry) = net.forward(layerNames)
    
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
        
    	# draw the bounding box on the image
    	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    
    return boxes




class CardIdentifier:
    @staticmethod
    def init(tesseract_path):
        if tesseract_path is not None and tesseract_path != "":
            pytesseract.pytesseract.tesseract_cmd = tesseract_path


    @staticmethod
    def identify_card(card_img: Int2D_3C, red_color: Color, blue_color: Color, yellow_color: Color) -> str:
        red_distance = ColorExtractor.distance_from_color(card_img, red_color)
        blue_distance = ColorExtractor.distance_from_color(card_img, blue_color)
        yellow_distance = ColorExtractor.distance_from_color(card_img, yellow_color, use_only_value=True)
        red_avg = np.mean(red_distance)
        blue_avg = np.mean(blue_distance)
        yellow_avg = np.mean(yellow_distance)
        print(f"Red: {red_avg}")
        print(f"Blu: {blue_avg}")
        print(f"Yel: {yellow_avg}")
        card_color = max((red_avg, "red"), (blue_avg, "blue"), (yellow_avg, "yellow"), key=lambda x: x[0])[1]

        if card_color == "yellow":
            image_copy = card_img.copy()
            cv2.imshow("image_copy", image_copy)
            cv2.waitKey(0)

            #image_copy = remove_noise(image_copy)
            #image_copy = 255 - cv2.threshold(image_copy, 160, 255, cv2.THRESH_BINARY_INV)[1]
            image_copy = image_copy.astype(np.uint8)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            image_copy = 255 - cv2.adaptiveThreshold(image_copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            #cv2.imshow("adaptive", image_copy)
            #cv2.waitKey(0)
            image_copy = adjust_gamma(image_copy)
            #image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            #image_copy = 255 - cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            image_copy = remove_noise(image_copy)

            boxes = east_detect(image_copy)
            #print(boxes)
            image_copy = remove_noise(image_copy)

            thresh = 255 - cv2.threshold(image_copy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            #cv2.imshow('thresh', thresh)
            #cv2.waitKey(0)


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

            for (x,y,w,h) in boxes:
                x = int(x * rW)
                y = int(y * rH)
                w = int(w * rW) + 10
                h = int(h * rH) + 10
                #print("x,y,w,h")
                #print(x,y,w,h)
                print("==========")
                ROI  = thresh[y:h, x:w]
                if ROI.shape[0] > 0 and ROI.shape[1] > 0:
                    cv2.imshow('ROI', ROI)
                    cv2.waitKey(0)
                    data = pytesseract.image_to_string(ROI, lang = 'eng', config = '--psm 6')
                    print(data)
                    card_words.append(data)
                #cv2.imshow('ROI', ROI)
                #cv2.waitKey(0)
                


            (x,y,w,h) = boxes[0]
            x = int(x * rW)
            y = int(y * rH)
            w = int(w * rW) + 10
            h = int(h * rH) + 10
            #print("x,y,w,h")
            #print(x,y,w,h)
            #print("=====")
            #ROI  = thresh[y:h, x:w]
            #cv2.imshow('ROI', ROI)
            #cv2.waitKey(0)

            #data = pytesseract.image_to_string(ROI, lang = 'eng', config = '--psm 6')
            #print(data)
        print(f"Card is: {card_color}")
        verbose_display([card_img, red_distance, blue_distance, yellow_distance])
        return card_color
