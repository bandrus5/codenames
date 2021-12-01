from image_processor_interface import ImageProcessorInterface
from data_labelling.game_state import *
from util.type_defs import *
from ryan.card_identifier import CardIdentifier
from ryan.ryan_image_processor import RyanImageProcessor
from berkeley.berkeley_image_processor import BerkeleyImageProcessor

class CodenamesImageProcessor(ImageProcessorInterface):
    def __init__(self, flags):
        self.card_grid_processor = RyanImageProcessor(flags)
        self.key_card_processor = BerkeleyImageProcessor(flags)

    def extract_state_from_image(self, input_image: Int2D_3C) -> Optional[GameState]:
        card_grid_state = self.card_grid_processor.extract_state_from_image(input_image)
        key_card_state = self.key_card_processor.extract_state_from_image(input_image)
        total_game_state = GameState(
            cards=card_grid_state.get(CARDS),
            key=key_card_state.get(KEY),
            first_turn=key_card_state.get(FIRST_TURN))
        return total_game_state
