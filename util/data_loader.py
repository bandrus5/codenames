from cv2 import cv2
from glob import glob
import json
import os
import numpy as np
from data_labelling.game_state import *


class Bundle(TypedDict):
    name: str
    image: np.ndarray
    label: GameState


BundleSet = List[Bundle]


def get_data(background: str = '*', difficulty: str = '*', dataset: str = 'validation') -> BundleSet:
    if len(background) == 2:
        background = f'[{background[0]}-{background[1]}]'
    if len(difficulty) == 2:
        difficulty = f'[{difficulty[0]}-{difficulty[1]}]'

    image_locations = sorted(glob(f'data_labelling/data/{dataset}/bg{background}_d{difficulty}*.jpg'))
    label_locations = [il.replace('.jpg', '.json') for il in image_locations]

    data_bundles: BundleSet = []
    for image_location, label_location in zip(image_locations, label_locations):
        if not os.path.exists(label_location):
            print(f'ERROR: No label file corresponding to {image_location}')
            continue

        input_image: np.ndarray = cv2.imread(image_location)
        image_name: str = image_location.split('/')[-1]
        with open(label_location) as label_file:
            game_state: GameState = json.load(label_file)

        new_bundle: Bundle = {'image': input_image, 'name': image_name, 'label': game_state}
        data_bundles.append(new_bundle)
    return data_bundles
