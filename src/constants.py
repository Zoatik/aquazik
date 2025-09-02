import random
from enum import Enum, auto


class Colors:
    bgColor = (125, 125, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    yellow = (231, 199, 25)
    white = (255, 255, 255)
    orange = (255, 150, 50)
    SAND = (232, 210, 160)
    bobHouse = (240, 100, 0)
    bobHouseLines = (0xAB, 0x21, 0x00)
    LEAF_MAIN = (27, 142, 73)
    LEAF_DARK = (18, 102, 53)
    LEAF_LIGHT = (46, 181, 101)

    patrick = (0xF8, 0xAE, 0xAF)


class FishColors:
    yellow = (255, 255, 0)
    orange = (255, 165, 0)
    pink = (255, 192, 203)
    purple = (128, 0, 128)
    light_blue = (173, 216, 230)
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    pinkTruite = (0xA9, 0x83, 0x80)
    darkGrayTruite = (0x3B, 0x45, 0x44)
    lightGrayTruite = (0xC9, 0xC9, 0xBF)

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
