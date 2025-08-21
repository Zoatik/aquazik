from animation.fish import Fish
from pygame import draw, Surface
from constants import NOTE_NAMES
import animation.drawings
from animation.drawings import Colors

class Aquarium:
    # -----Functions to create the fishes --------------------------------------------------------------
    def drawFishes(fishList):
        for i in range(0, len(fishList)):
            fishList[i].draw()

    def createFishList(window):
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
        fishList: list[Fish] = []
        for ni in range(len(NOTE_NAMES)):
            fishList.append(Fish(window, NOTE_NAMES[ni], Colors.get_random_color() , centerList[ni]))
        return fishList
    
    def drawProgressBar(window: Surface, current_time, total_time):
        height = window.get_size()[1] - 25
        start = 15
        # end = window_width - 50px
        end = window.get_size()[0] - 50

        percentage = current_time / total_time

        current = start + (end - start) * percentage
        if (current > end):
            current = end
        
        # ---------------- DRAW GARY
        radius = 20
        
        # draw body
        body = [
            (current, height + radius),
            (current + 50, height),
            (current + 50, height + radius)
        ]

        bodyColor = (0x87, 0xcd, 0xea)
        draw.polygon(window, Colors.black, body, 2)
        draw.polygon(window, bodyColor, body)

        # draw coquille
        coquille = animation.drawings.getOctogonPoints(current, height, radius)

        for t in coquille:
            draw.polygon(window, (0xdc, 0x49, 0x60), t, 2)
            draw.polygon(window, (0xf4, 0x83, 0xac), t)
        
        # draw eyelids
        eyelids = [
            [ # eyelid 1
                (current + 43, height + 5),
                (current + 38, height - 15),
                (current + 40, height - 15)
            ],
            [ # eyelid 1 filler
                (current + 42, height + 5),
                (current + 44, height + 5),
                (current + 39, height - 15) 
            ],
            [ # eyelid 2
                (current + 47, height + 5),
                (current + 52, height - 15),
                (current + 50, height - 15)
            ],
            [ # eyelid 2 filler
                (current + 46, height + 5),
                (current + 48, height + 5),
                (current + 51, height - 15)
            ]
        ]

        for e in eyelids:
            draw.polygon(window, bodyColor, e)

        # eyes
        eyes = [
            animation.drawings.getOctogonPoints(current + 51, height - 15, 8),
            animation.drawings.getOctogonPoints(current + 39, height - 15, 8)
        ]
        for e in eyes:
            for t in e:
                draw.polygon(window,Colors.yellow, t)
        
        # iris
        iris = [
            animation.drawings.getOctogonPoints(current + 53, height - 14, 4),
            animation.drawings.getOctogonPoints(current + 41, height - 14, 4)
        ]
        for e in iris:
            for t in e:
                draw.polygon(window,(0xda, 0x6e, 0x2c), t)
        
        pupils = [
            animation.drawings.getOctogonPoints(current + 54, height - 13, 2),
            animation.drawings.getOctogonPoints(current + 42, height - 13, 2),
        ]
        for e in pupils:
            for t in e:
                draw.polygon(window,Colors.black, t)