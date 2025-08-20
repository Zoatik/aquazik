import pygame


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
            pygame.draw.polygon(self.window, black, self.listTriangles[i], 2)

        # black eye
        pygame.draw.polygon(
            self.window,
            black,
            (
                (self.center - 5, self.center - 40),
                (self.center + 5, self.center - 40),
                (self.center, self.center - 35),
            ),
        )

    def changeColor(self, c: tuple):
        self.color = c


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


# -----Functions to create the fishes --------------------------------------------------------------
def drawFishes():
    for i in range(0, len(fishList)):
        fishList[i].draw()


# instance of fish --> change it so i don't do it manually
f1 = Fish(window, "D", red, 300)
f2 = Fish(window, "C", blue, 150)
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
            for i in range(0, len(fishList)):
                if fishList[i].color == red:
                    fishList[i].changeColor(green)
                elif fishList[i].color == green:
                    fishList[i].changeColor(red)
                elif fishList[i].color == blue:
                    fishList[i].changeColor(yellow)
                elif fishList[i].color == yellow:
                    fishList[i].changeColor(blue)
    pygame.display.flip()

pygame.quit
exit()
