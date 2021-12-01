from util.data_loader import *
import util.image_functions
from util.image_functions import display_image
from ryan.ryan_image_processor import RyanImageProcessor
from berkeley.berkeley_image_processor import BerkeleyImageProcessor
from codenames_image_processsor import CodenamesImageProcessor
import colorama
import argparse

colorama.init()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_set',
                        type=str,
                        choices=['test', 'validation', 'dark'],
                        default='validation',
                        help='The dataset to process')

    parser.add_argument('-b', '--background',
                        type=str,
                        default='1',
                        help='Filter dataset based on a specific background')

    parser.add_argument('-d', '--difficulty',
                        type=str,
                        default='1',
                        help='Filter dataset based on difficulty of images')

    parser.add_argument('-p', '--processor',
                        type=str,
                        choices=['berkeley', 'ryan', 'duncan', 'display', 'codenames'],
                        default='display',
                        help='Which analyzer to use for extracting game state from images')

    parser.add_argument('-tp', '--tesseract_path',
                        type=str,
                        help='Filter dataset based on difficulty of images')

    parser.add_argument('-v', '--verbose',
                        action='store_true')

    flags, _ = parser.parse_known_args()

    dataset: BundleSet = get_data(flags.background, flags.difficulty, flags.input_set)

    util.image_functions.is_verbose = flags.verbose

    if flags.processor.lower() == 'berkeley':
        image_processor = BerkeleyImageProcessor(flags=flags)
    elif flags.processor.lower() == 'ryan':
        image_processor = RyanImageProcessor(flags=flags)
    elif flags.processor.lower() == 'duncan':
        image_processor = None
    elif flags.processor.lower() == 'codenames':
        image_processor = CodenamesImageProcessor(flags=flags)
    else:
        image_processor = None

    for image_bundle in dataset:
        if image_processor is not None:
            game_state: Optional[GameState] = image_processor.extract_state_from_image(image_bundle['image'])
        else:
            game_state = None

        print()
        print()
        print()
        print('---Image---')
        print(image_bundle['name'])
        print('---Expected Game State---')
        print_game_state(image_bundle['label'])
        print('---Predicted Game State---')
        print_game_state(game_state)
        # display_image(image_bundle['image'], 1000, 1000)
