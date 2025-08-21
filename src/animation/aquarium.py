import pygame
from time import time
from src.animation.fish import Fish, Colors

# faut lancer avec py -m src.animation.aquarium
from src.audio_processing.midi_reader import MidiFile, MidiNote

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

# List of note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
fishList: list[Fish] = [
    # Fish(window, "D", Colors.red, centerList[0]),
    # Fish(window, "C", Colors.blue, centerList[12])
]
for ni in range(len(NOTE_NAMES)):
    fishList.append(
        Fish(window, NOTE_NAMES[ni], Colors.get_random_color(), centerList[ni])
    )

# Create MidiFile instance
mdi = MidiFile("audio_in/PinkPanther.midi")

# ---Loop, update display and quit------------------------------------------------------------------

run = True
init = True
# Loop that updates the display

start = time()
last_notes = []
while run:
    # time
    currentTime = time() - start

    notes = mdi.find_note(currentTime)
    # [:-1] enlève le dernier char du string (l'octave de la note)
    result = [
        x.get_real_note()[:-1] for x in notes if x not in last_notes
    ]  # uniquement les nouvelles notes
    # result = [x.get_real_note()[:-1] for x in notes]                           # contient toutes les notes jouées à ce moment-là

    last_notes = notes

    # for each fish
    for i in range(len(fishList)):
        # if notes played contain fish name, change it's color
        if result.__contains__(fishList[i].name):
            print(fishList[i].name)
            fishList[i].changeColor()

    window.fill(Colors.bgColor)
    drawFishes()
    for event in pygame.event.get():
        # quit if click quit
        if event.type == pygame.QUIT:
            run = False

    pygame.display.flip()

pygame.quit
exit()
