#Importing libraries
import cv2
import pytesseract
import numpy as np

#Loading image using OpenCV
img = cv2.imread('Codenames Cards/IMG_1756.jpeg')

# Convert to RGB 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect texts from image
texts = pytesseract.image_to_string(img)
print(texts)



def funcBrightContrast(bright=0):
    img = cv2.imread('Codenames Cards/IMG_1756.jpeg')
    img = apply_brightness_contrast(img, brightness = 255, contrast = 175)
    cv2.imshow('Effect', img)
    # Detect texts from image
    texts = pytesseract.image_to_string(img)
    print(texts)

def apply_brightness_contrast(input_img, brightness = 255, contrast = 127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    cv2.putText(buf,'B:{},C:{}'.format(brightness,contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return buf

def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)


def draw_boxes_on_text(img):
    # Return raw information about the detected texts
    raw_data = pytesseract.image_to_data(img)

    print(raw_data)
    for count, data in enumerate(raw_data.splitlines()):
        if count > 0:
            data = data.split()
            if len(data) == 12:
                x, y, w, h, content = int(data[6]), int(data[7]), int(data[8]), int(data[9]), data[11]
                cv2.rectangle(img, (x, y), (w+x, h+y), (0, 255, 0), 1)
                cv2.putText(img, content, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255) , 1)
                
    return img

funcBrightContrast()

# show the output
#cv2.imshow("Output", img)
cv2.waitKey(0)

img = draw_boxes_on_text(img)    # Uncomment this if you want to detect texts




