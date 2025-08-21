import pygame
from random import randrange


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


# --------------------------------------------------------------------------------------------------
# -------Create the window--------------------------------------------------------------------------

# Initialise pygame
pygame.init()

# Create a window
(width, height) = (600, 500)
window = pygame.display.set_mode((width, height))

# Set window's caption // and icon
pygame.display.set_caption("Aquazik")
# icon = pygame.image.load('....png')
# pygame.display.set_icon(icon)

# color variables
bgColor = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
yellow = (231, 199, 25)
white = (255, 255, 255)

# List of note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

noteUsed = ["C", "B"]


# -----Functions to create the fishes --------------------------------------------------------------
def drawFishes():
    for i in range(0, len(fishList)):
        fishList[i].draw()


# instance of fish --> change it so i don't do it manually
# x + 150, y + 125
centerList = [
    (100, 50),
    (250, 50),
    (400, 50),
    (500, 50),
    (100, 175),
    (250, 175),
    (400, 175),
    (500, 175),
    (100, 300),
    (250, 300),
    (400, 300),
    (500, 300),
    (100, 425),
    (250, 425),
    (400, 425),
    (500, 425),
]
f1 = Fish(window, "D", red, centerList[0])
f2 = Fish(window, "C", blue, centerList[12])
fishList: list[Fish] = [f1, f2]

# ---Loop, update display and quit------------------------------------------------------------------

run = True
init = True
# Loop that updates the display
while run:
    window.fill(bgColor)
    drawFishes()
    for event in pygame.event.get():
        # quit if click quit
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONUP:

            """
            Need to import the function from midi_reader
            if get_instrument == "PIANO":
                print("Piano note played")
            """
            for i in range(0, len(fishList)):
                # change to function that checks if note is played
                if noteUsed.__contains__(fishList[i].name):
                    fishList[i].changeColor()
    pygame.display.flip()

pygame.quit
exit()
