from typing import *
from colorama import Fore, Back

CARDS = 'cards'
FIRST_TURN = 'first_turn'
KEY = 'key'

CardState = List[List[str]]
KeyState = List[List[str]]


class GameState(TypedDict):
    cards: Optional[CardState]
    first_turn: Optional[str]
    key: Optional[KeyState]


__color_map = {'r': Fore.RED,
             'b': Fore.BLUE,
             'y': Fore.YELLOW,
             'k': Fore.GREEN}


def __get_colored_item(item: str, background_color: int = None) -> str:
    color = __color_map.get(item, Fore.RESET)
    if background_color is not None:
        bg_color = background_color
    else:
        bg_color = ""
    return f'{color}{bg_color}{item}{Fore.RESET}{Back.RESET}'


def get_formatted_row(row: List[str], width: int, highlight_row: Optional[List[bool]], highlight_color: str) -> str:
    output = ''
    for i, item in enumerate(row):
        if highlight_row is not None and highlight_row[i]:
            background_color = highlight_color
        else:
            background_color = None
        colored_item = __get_colored_item(item, background_color)
        colored_adjustment = len(colored_item) - len(item)
        output += colored_item.ljust(width + colored_adjustment)
    return output


def print_game_state(game_state: Optional[GameState], highlight_game_state: Optional[GameState] = None):
    if game_state is None:
        print("No GameState")
    else:
        highlight_color = Back.MAGENTA
        first_turn = game_state.get(FIRST_TURN)
        highlight_first_turn_color = None
        if highlight_game_state is not None:
            if highlight_game_state.get(FIRST_TURN):
                highlight_first_turn_color = highlight_color
        print(f'Turn: {__get_colored_item(first_turn, highlight_first_turn_color)}')
        cards = game_state.get(CARDS)
        highlight_cards = None
        if highlight_game_state is not None:
            highlight_cards = highlight_game_state.get(CARDS)
        if cards is not None:
            for i, card_row in enumerate(cards):
                highlight_cards_row = None
                if highlight_cards is not None:
                    highlight_cards_row = highlight_cards[i]
                print(get_formatted_row(card_row, 14, highlight_cards_row, highlight_color))
        else:
            print('No cards')
        key = game_state.get(KEY)
        highlight_key = None
        if highlight_game_state is not None:
            highlight_key = highlight_game_state.get(KEY)
        if key is not None:
            for i, key_row in enumerate(key):
                highlight_key_row = None
                if highlight_key is not None:
                    highlight_key_row = highlight_key[i]
                print(get_formatted_row(key_row, 2, highlight_key_row, highlight_color))
        else:
            print('No key')

def __matrix_differences(matrix_1: List[List[str]], matrix_2: List[List[str]]) -> List[List[bool]]:
    differences = []
    for i in range(len(matrix_1)):
        row_1 = matrix_1[i]
        row_2 = matrix_2[i]
        row_differences = []
        for j in range(len(row_1)):
            item_1 = row_1[j]
            item_2 = row_2[j]
            row_differences.append(item_1 != item_2)
        differences.append(row_differences)
    return differences

def find_game_state_differences(game_state_1: GameState, game_state_2: GameState) -> Optional[GameState]:
    if game_state_1 is None or game_state_2 is None:
        return None
    cards_1 = game_state_1.get(CARDS)
    cards_2 = game_state_2.get(CARDS)
    cards_differences = __matrix_differences(cards_1, cards_2)
    key_1 = game_state_1.get(KEY)
    key_2 = game_state_2.get(KEY)
    key_differences = __matrix_differences(key_1, key_2)
    first_turn_1 = game_state_1.get(FIRST_TURN)
    first_turn_2 = game_state_2.get(FIRST_TURN)
    first_turn_difference = first_turn_1 != first_turn_2

    is_game_state_different = GameState(cards=cards_differences, key=key_differences, first_turn=first_turn_difference)
    return is_game_state_different
