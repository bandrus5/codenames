from util.data_loader import *
from util.image_functions import display_image

if __name__ == '__main__':
    # Background 1, difficulty 1, validation
    dataset1: BundleSet = get_data('1', '1', 'validation')
    display_image(dataset1[0]['image'])
    # Backgrounds 1-3, all difficulties, test
    dataset2: BundleSet = get_data('13', '*', 'test')
    for el in dataset2:
        display_image(el['image'])
