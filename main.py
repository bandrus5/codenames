from util.data_loader import get_data
from util.image_functions import display_image
import cv2

if __name__=='__main__':
    # Background 1, difficulty 1, validation
    dataset1 = get_data('1', '1', 'validation')
    display_image(dataset1[0]['image'])
    # Backgrounds 1-3, all difficulties, test
    dataset2 = get_data('13', '*', 'test')
    for el in dataset2:
        display_image(el['image'])
