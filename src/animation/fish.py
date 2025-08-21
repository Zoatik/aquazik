import pygame


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
            pygame.draw.polygon(self.window, black, self.listTriangles[i], 2)

        # black eye
        pygame.draw.polygon(
            self.window,
            black,
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
