import pygame
from time import time
from random import random
from animation.aquarium import Aquarium
from animation.fish import Bubble
from audio_processing.midi_reader import MidiFile, Instrument
from audio_processing.freq_analysis import AudioAnalyzer
import audio_processing.MidiV2
from constants import Colors,FishColors
import ctypes
import platform

def main():
    FILE = "PinkPanther_Trumpet_Only.mp3"

    # Setup analysis
    print("-- Analysing audio --")
    audio_analyser = AudioAnalyzer(FILE)
    audio_data = audio_analyser.convert_to_notes()

    print("-- Creating MIDI file --")
    midi_path = audio_processing.MidiV2.midi_maker([(0,audio_data[1])], bpm=audio_data[0])
    print(f"bpm = {audio_data[0]}")

    # Create MidiFile instance
    print("-- Processing MIDI file --")
    mdi = MidiFile(midi_path)

    # -------Create the window--------------------------------------------------------------------------
    print("-- Creating pygame window --")
    pygame.init()

    # Create a window
    if "Windows" in platform.system():
        ctypes.windll.user32.SetProcessDPIAware()
    (width, height) = (1600, 900)
    window = pygame.display.set_mode((width, height))

    # Set window's caption // and icon
    pygame.display.set_caption("Aquazik")
    path = 'src/journalist.png'
    icon = pygame.image.load(path)
    pygame.display.set_icon(icon)

    # ---Loop, update display and quit------------------------------------------------------------------

    run = True
    init = True

    # Start music
    pygame.mixer.init()
    pygame.mixer.music.load("audio_in/" + FILE)
    pygame.mixer.music.play()
    pygame.event.wait()

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

        result_piano = [
            x.get_real_note()[:-1]
            for x in notes
            if x not in last_notes and x.get_instrument() == Instrument.PIANO
        ]
        result_trumpet = [
            x.get_real_note()[:-1]
            for x in notes
            if x not in last_notes and x.get_instrument() == Instrument.TRUMPET
        ]

        allnotes_piano = [x for x in notes if x.get_instrument() == Instrument.PIANO]

        last_notes = notes

        # fish note animation
        for i in range(len(fishList)):
            # create bubble if there's a note played
            if result_piano.__contains__(fishList[i].name):
                bubbleList.append(fishList[i].createBubble(window))

            fishList[i].animate = [x.get_real_note()[:-1] for x in allnotes_piano].__contains__(fishList[i].name)

        # for each starfish
        for i in range(len(starFishList)):
            # if notes played contain fish name, change it's color
            if result_trumpet.__contains__(starFishList[i].name):
                starFishList[i].animStarfish()

        # draw aquarium background and details
        Aquarium.drawBackground(window)

        Aquarium.drawFishes(fishList)
        Aquarium.drawStarfish(starFishList)
        for b in [x for x in bubbleList if not x.out_of_bounds]:
            b.move_and_draw()

        Aquarium.drawPatrickHouse(window)
        Aquarium.drawSquidwardHouse(window)
        Aquarium.drawBobHouse(window)
        Aquarium.drawBobTopHouse(window)

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
