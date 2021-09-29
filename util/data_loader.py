import cv2
from glob import glob
import json
import os

def get_data(background='*', difficulty='*', dataset='validation'):
    if len(background) == 2:
        background = f'[{background[0]}-{background[1]}]'
    if len(difficulty) == 2:
        difficulty = f'[{difficulty[0]}-{difficulty[1]}]'

    image_locations = sorted(glob(f'data-labelling/data/{dataset}/bg{background}_d{difficulty}*.jpg'))
    label_locations = [il.replace('.jpg', '.json') for il in image_locations]

    data_bundles = []
    for image_location, label_location in zip(image_locations, label_locations):
        if not os.path.exists(label_location):
            print(f'ERROR: No label file corresponding to {image_location}')
            continue
        new_bundle = { 'image': cv2.imread(image_location), 'name': image_location.split('/')[-1] }
        with open(label_location) as label_file:
            new_bundle['label'] = json.load(label_file)
        data_bundles.append(new_bundle)
    return data_bundles
