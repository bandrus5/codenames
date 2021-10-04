from typing import *

CardState = List[List[str]]
KeyState = List[List[str]]


class GameState(TypedDict):
    cards: CardState
    first_turn: str
    key: KeyState
