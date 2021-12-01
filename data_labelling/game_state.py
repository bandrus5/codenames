from typing import *
from colorama import Fore

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


def __get_colored_item(item: str) -> str:
    color = __color_map.get(item, Fore.RESET)
    return f'{color}{item}{Fore.RESET}'


def get_formatted_row(row: List[str], width: int) -> str:
    output = ''
    for item in row:
        colored_item = __get_colored_item(item)
        colored_adjustment = len(colored_item) - len(item)
        output += colored_item.ljust(width + colored_adjustment)
    return output


def print_game_state(game_state: Optional[GameState]):
    if game_state is None:
        print("No GameState")
    else:
        first_turn = game_state.get(FIRST_TURN)
        print(f'Turn: {first_turn}')
        cards = game_state.get(CARDS)
        if cards is not None:
            for card_row in cards:
                print(get_formatted_row(card_row, 14))
        else:
            print('No cards')
        key = game_state.get(KEY)
        if key is not None:
            for key_row in key:
                print(get_formatted_row(key_row, 2))
        else:
            print('No key')
