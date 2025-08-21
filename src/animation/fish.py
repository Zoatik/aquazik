import pygame
from random import randrange

class Colors:
    bgColor = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    yellow = (231, 199, 25)
    white = (255, 255, 255)

# -----Fish Class----------------------------------------------------------------------------------
# I can't seem to find a way to import this class from fish.py, so i copied it here...
# Need to learn how to do that...
class Fish:
    global listTriangles

    def __init__(self, window, name: str, color: str, center):
        self.window = window
        self.name = name
        self.center = center
        self.color = color
        self.listTriangles = [
            (
                (self.center - 75, self.center - 25),
                (self.center - 75, self.center + 25),
                (self.center, self.center),
            ),
            (
                (self.center - 50, center - 17),
                (self.center - 25, center - 50),
                (self.center, self.center),
            ),
            (
                (self.center - 50, self.center + 17),
                (self.center - 25, self.center + 50),
                (self.center, self.center),
            ),
            (
                (self.center - 25, self.center - 50),
                (self.center + 25, self.center - 50),
                (self.center, self.center),
            ),
            (
                (self.center - 25, self.center + 50),
                (self.center + 25, self.center + 50),
                (self.center, self.center),
            ),
            (
                (self.center + 25, self.center - 50),
                (self.center + 50, self.center - 17),
                (self.center, self.center),
            ),
            (
                (self.center + 25, self.center + 50),
                (self.center + 50, self.center + 17),
                (self.center, self.center),
            ),
        ]

    def __str__(self):
        return f"{self.name}, {self.color}"

    def draw(self):
        # body parts and contouring
        for i in range(0, len(self.listTriangles)):
            pygame.draw.polygon(self.window, self.color, self.listTriangles[i])
            pygame.draw.polygon(self.window, Colors.black, self.listTriangles[i], 2)

        # black eye
        pygame.draw.polygon(
            self.window,
            Colors.black,
            (
                (self.center - 5, self.center - 40),
                (self.center + 5, self.center - 40),
                (self.center, self.center - 35),
            ),
        )

    # change the color of the fish to a random color
    def changeColor(self):
        self.color = (
            randrange(255),
            randrange(255),
            randrange(255),
        )
