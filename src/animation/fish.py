import pygame
<<<<<<< HEAD


=======
from random import randrange, random

class Colors:
    bgColor = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    yellow = (231, 199, 25)
    white = (255, 255, 255)
    
    def get_random_color():
        # Récupère tous les attributs de la classe sauf spéciaux (__xxx__)
        values = [v for k, v in Colors.__dict__.items() if not k.startswith("__") and not k.startswith("bgColor") and not k.startswith("get")]
        return values[int(random()*len(values))]

# -----Fish Class----------------------------------------------------------------------------------
# I can't seem to find a way to import this class from fish.py, so i copied it here...
# Need to learn how to do that...
>>>>>>> af75cfafb174f725afd365f84fffdd7b9be3144e
class Fish:
    global listTriangles

    def __init__(self, window, name: str, color: str, center):
        self.window = window
        self.name = name
        self.center = center
        self.color = color
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
        # body parts and contouring
        for i in range(0, len(self.listTriangles)):
            pygame.draw.polygon(self.window, self.color, self.listTriangles[i])
<<<<<<< HEAD
            pygame.draw.polygon(self.window, black, self.listTriangles[i], 2)
=======
            pygame.draw.polygon(self.window, Colors.black, self.listTriangles[i], 2)
>>>>>>> af75cfafb174f725afd365f84fffdd7b9be3144e

        # black eye
        pygame.draw.polygon(
            self.window,
<<<<<<< HEAD
            black,
=======
            Colors.black,
>>>>>>> af75cfafb174f725afd365f84fffdd7b9be3144e
            (
                (self.center[0] - 5, self.center[1] - 40),
                (self.center[0] + 5, self.center[1] - 40),
                (self.center[0], self.center[1] - 35),
            ),
        )

    # change the color of the fish to a random color
    def changeColor(self):
        self.color = (
            randrange(255),
            randrange(255),
            randrange(255),
        )
