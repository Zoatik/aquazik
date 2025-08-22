import pygame

from random import randrange, random
from animation.drawings import Colors
import animation.drawings


# -----Fish Class----------------------------------------------------------------------------------
# I can't seem to find a way to import this class from fish.py, so i copied it here...
# Need to learn how to do that...
class Fish:
    global listTriangles
    global fistColor
    global secondColor

    def __init__(self, window: pygame.Surface, name: str, color, center):
        self.window = window
        self.name = name
        self.center = center
        self.color = color
        self.firstColor = color
        self.secondColor = (randrange(255), randrange(255), randrange(255))
        self.listTriangles = [
            (
                (self.center[0] - 75, self.center[1] - 25),
                (self.center[0] - 75, self.center[1] + 25),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 50, self.center[1] - 17),
                (self.center[0] - 25, self.center[1] - 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 50, self.center[1] + 17),
                (self.center[0] - 25, self.center[1] + 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 25, self.center[1] - 50),
                (self.center[0] + 25, self.center[1] - 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 25, self.center[1] + 50),
                (self.center[0] + 25, self.center[1] + 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] + 25, self.center[1] - 50),
                (self.center[0] + 50, self.center[1] - 17),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] + 25, self.center[1] + 50),
                (self.center[0] + 50, self.center[1] + 17),
                (self.center[0], self.center[1]),
            ),
        ]

    def __str__(self):
        return f"{self.name}, {self.color}"

    def draw(self):
        self.drawBorder(3)

        for i in range(0, len(self.listTriangles)):
            pygame.draw.polygon(self.window, self.color, self.listTriangles[i])

        # black eye
        pygame.draw.polygon(
            self.window,
            Colors.black,
            (
                (self.center[0] - 5, self.center[1] - 40),
                (self.center[0] + 5, self.center[1] - 40),
                (self.center[0], self.center[1] - 35),
            ),
        )

    # change the color of the fish to a random color
    def changeColor(self):
        if self.color == self.firstColor:
            self.color = self.secondColor
        else:
            self.color = self.firstColor
    
    def drawBorder(self, bordersize = 1):
        for i in range(len(self.listTriangles)):
            pygame.draw.polygon(
                self.window, Colors.black, self.listTriangles[i], width=(bordersize+3)
            )

class Bubble:
    out_of_bounds: bool = False

    def __init__(
        self, window: pygame.Surface, starting_pos: tuple[int, int], radius: int
    ):
        self.window = window
        self.pos = starting_pos
        self.radius = radius

    def move_and_draw(self):
        # bouge +- en x et toujours - en y
        self.pos = (
            self.pos[0] + (int(random() * 2 - 1) * 4),
            self.pos[1] - int(randrange(2)),
        )

        if (self.pos[1] - self.radius <= 0):
            self.out_of_bounds = True
            return

        points = animation.drawings.getPolygonPoints(
            15, self.pos[0], self.pos[1], self.radius
        )
        # draw bubble's border
        for t in points:
            pygame.draw.polygon(self.window, Colors.black, t, width = 3)
        # draw white part of bubble (inside)
        for t in points:
            pygame.draw.polygon(self.window, Colors.white, t)
