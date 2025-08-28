import pygame
from time import time
import random
from animation.aquarium import Aquarium
from animation.fish import Fish, Bubble
from audio_processing.midi_reader import MidiFile, Instrument
from audio_processing.freq_analysis import AudioAnalyzer
import audio_processing.MidiV2
from constants import Colors,FishColors, Direction
import ctypes
import platform
import os

def main():
    FILE = "PinkPanther_Both.mp3"

    # Setup analysis
    print("-- Analysing audio --")
    #audio_analyser = AudioAnalyzer(FILE)
    #audio_data = audio_analyser.convert_to_notes()

    print("-- Creating MIDI file --")
    #midi_path = audio_processing.MidiV2.midi_maker([(0,audio_data[1])], bpm=audio_data[0])
    midi_path = "audio_in/PinkPanther.midi"
    #print(f"bpm = {audio_data[0]}")

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

    # start timestamp, used for bpm sync
    start = time()
    last_notes = []
    bubbleList: list[Bubble] = []
    #fishList = Aquarium.createFishList(window)
    fishList = []
    starFishList = Aquarium.createStarfishList(window)
    last_nextNotes = []

    deltaTime = 0

    while run:
        runStartTime = time()
        currentTime = runStartTime - start

        notes = mdi.find_note(currentTime)
        
        result_piano = [
            x
            for x in notes
            if x not in last_notes and x.get_instrument() == Instrument.PIANO
        ]
        result_trumpet = [
            x
            for x in notes
            if x not in last_notes and x.get_instrument() == Instrument.TRUMPET
        ]

        # [:-1] enlève le dernier char du string (l'octave de la note)
        allnotes_piano = [x.get_real_note()[:-1] for x in notes if x.get_instrument() == Instrument.PIANO]
        allnotes_trumpet = [x.get_real_note()[:-1] for x in notes if x.get_instrument() == Instrument.TRUMPET]

        last_notes = notes

        nextNotes = mdi.find_note(currentTime + 2)

        fishList = [f for f in fishList if f.enabled]
        for note in [x for x in nextNotes 
                     if not last_nextNotes.__contains__(x) 
                     and x.get_instrument() == Instrument.PIANO]:
            if len([x for x in fishList if x.name == note.get_real_note()[:-1]]) == 0:
                # create fish
                direction = random.choice([Direction.LEFT, Direction.RIGHT])
                distance = note.velocity / 6

                fishList.append(Fish(
                    window,
                    note.get_real_note()[:-1],
                    FishColors.yellow,
                    # TODO center.y du poisson à changer par rapport à la note
                    ((distance if direction == Direction.RIGHT else window.get_size()[0] - note.velocity / 6), random.randrange(int(window.get_size()[1] / 2))),
                    length = note.velocity / 3,
                    height = note.velocity / 4,
                    direction = direction
                ))

        last_nextNotes = nextNotes

        # fish note animation
        for fish in fishList:
            if fish.playing:
                fish.lastNoteTime = time()
            fish.playing = False
            result_piano_fish = [x for x in result_piano if x.get_real_note()[:-1] == fish.name]

            # new notes
            if len(result_piano_fish) != 0:
                fish.openMouth(result_piano_fish[0].velocity, result_piano_fish[0].get_time())
                bubbleList.append(fish.createBubble(window))

            # all notes
            if allnotes_piano.__contains__(fish.name):
                fish.playing = True
            
            # animer tous les poissons à chaque fois
            fish.animate(deltaTime)

        # for each starfish
        for starfish in starFishList:
            starfish.playing = False
            result_trumpet_starfish = [x for x in result_trumpet if x.get_real_note()[:-1] == starfish.name]

            # new notes
            if len(result_trumpet_starfish) != 0:
                #I don't know if needed
                #starfish.animStarfish()
                print("Is Mayonnaise an Instrument?")

            # all notes
            if allnotes_trumpet.__contains__(starfish.name):
                starfish.playing = True

        # draw aquarium background and details
        Aquarium.drawBackground(window)

        Aquarium.drawPatrickHouse(window)
        Aquarium.drawSquidwardHouse(window)
        Aquarium.drawBobHouse(window)
        Aquarium.drawBobTopHouse(window)

        Aquarium.drawStarfish(starFishList)
        Aquarium.drawFishes(fishList)
        for b in [x for x in bubbleList if not x.out_of_bounds]:
            b.move_and_draw()

        Aquarium.drawProgressBar(window, currentTime, mdi.totalTime)

        for event in pygame.event.get():
            # quit if click quit
            if event.type == pygame.QUIT:
                run = False

        pygame.display.flip()

        deltaTime = time() - runStartTime

    pygame.quit()
    exit()


if __name__ == "__main__":
    main()
