import pygame
from time import time
from random import random
from animation.aquarium import Aquarium
from animation.fish import Colors, Bubble
from audio_processing.midi_reader import MidiFile
from audio_processing.freq_analysis import AudioAnalyzer
import audio_processing.MidiV2

def main():
    # Setup analysis
    print("-- Analysing audio --")
    #audio_analyser = AudioAnalyzer("PinkPanther_Both.mp3")

    print("-- Creating MIDI file --")
    #file_name = audio_processing.MidiV2.midi_maker()
    
    # Create MidiFile instance
    print("-- Processing MIDI file --")
    mdi = MidiFile("audio_in/PinkPanther.midi")

    # -------Create the window--------------------------------------------------------------------------
    print("-- Creating pygame window --")
    pygame.init()

    # Create a window
    (width, height) = (1600, 900)
    window = pygame.display.set_mode((width, height))

    # Set window's caption // and icon
    pygame.display.set_caption("Aquazik")
    # icon = pygame.image.load('....png')
    # pygame.display.set_icon(icon)


    # ---Loop, update display and quit------------------------------------------------------------------

    run = True
    init = True
    # Loop that updates the display

    start = time()
    last_notes = []
    bubbleList: list[Bubble] = []
    fishList = Aquarium.createFishList(window)
    starFishList = Aquarium.createStarfishList(window)

    while run:
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
                fishList[i].changeColor()
                bubbleList.append(
                    Bubble(
                        window,
                        (
                            fishList[i].center[0] + 20 + random() * 5,
                            fishList[i].center[1],
                        ),
                        5 + random() * 20,
                    )
                )

        for i in range(len(starFishList)):
            # if notes played contain fish name, change it's color
            if result.__contains__(starFishList[i].name):
                starFishList[i].animStarfish()

        # draw aquarium background and details
        Aquarium.drawBackground(window)
        Aquarium.drawFishes(fishList)
        Aquarium.drawStarfish(starFishList)
        for b in bubbleList:
            b.move_and_draw()
        Aquarium.drawProgressBar(window, currentTime, mdi.totalTime)
        for event in pygame.event.get():
            # quit if click quit
            if event.type == pygame.QUIT:
                run = False

        pygame.display.flip()

    pygame.quit()
    exit()


if __name__ == "__main__":
    main()
