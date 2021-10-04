from typing import *
from colorama import Fore

CardState = List[List[str]]
KeyState = List[List[str]]


class GameState(TypedDict):
    cards: CardState
    first_turn: str
    key: KeyState


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


def print_game_state(game_state: GameState):
    first_turn = game_state['first_turn']
    print(f'Turn: {first_turn}')
    for card_row in game_state['cards']:
        print(get_formatted_row(card_row, 14))
    for key_row in game_state['key']:
        print(get_formatted_row(key_row, 2))
