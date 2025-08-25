import random
from enum import Enum, auto

# List of note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class Colors:
    bgColor = (0x44, 0x63, 0xB2)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    yellow = (231, 199, 25)
    white = (255, 255, 255)
    orange = (255, 150, 50)
    SAND = (232, 210, 160)

    
class FishColors:
    yellow = (255, 255, 0)
    orange = (255, 165, 0)
    pink = (255, 192, 203)
    purple = (128, 0, 128)
    light_blue = (173, 216, 230)
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    def get_random_color():
        # Récupère tous les attributs de la classe sauf spéciaux (__xxx__)
        values = [
            v
            for k, v in FishColors.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("get")
            and not k.startswith("black")
            and not k.startswith("white")
        ]
        return values[int(random() * len(values))]
    

class Direction(Enum):
    LEFT: float = -1
    RIGHT: float = 1