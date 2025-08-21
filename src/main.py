import pygame
from time import time
from animation.aquarium import Aquarium
from animation.fish import Colors
from audio_processing.midi_reader import MidiFile


def main():
    print("Hello from Aquazik!")

    # -------Create the window--------------------------------------------------------------------------

    # Initialise pygame
    pygame.init()

    # Create a window
    (width, height) = (625, 500)
    window = pygame.display.set_mode((width, height))

    # Set window's caption // and icon
    pygame.display.set_caption("Aquazik")
    # icon = pygame.image.load('....png')
    # pygame.display.set_icon(icon)

    # Create MidiFile instance
    mdi = MidiFile("audio_in/PinkPanther.midi")

    # ---Loop, update display and quit------------------------------------------------------------------

    run = True
    init = True
    # Loop that updates the display

    start = time()
    last_notes = []
    fishList = Aquarium.createFishList(window)
    starFishList = Aquarium.createStarfishList(window)

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
        Aquarium.drawFishes(fishList)
        Aquarium.drawStarfish(starFishList)
        for event in pygame.event.get():
            # quit if click quit
            if event.type == pygame.QUIT:
                run = False

        pygame.display.flip()

    pygame.quit()
    exit()


if __name__ == "__main__":
    main()
