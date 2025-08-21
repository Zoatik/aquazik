from animation.fish import Fish, Colors
from constants import NOTE_NAMES
from animation.starfish import Starfish


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
            (550, 50),
            (100, 175),
            (250, 175),
            (400, 175),
            (550, 175),
            (100, 300),
            (250, 300),
            (400, 300),
            (550, 300),
            (100, 425),
            (250, 425),
            (400, 425),
            (550, 425),
        ]
        fishList: list[Fish] = []
        for ni in range(len(NOTE_NAMES)):
            fishList.append(
                Fish(window, NOTE_NAMES[ni], Colors.get_random_color(), centerList[ni])
            )
        return fishList

    def createStarfishList(window):
        starFishList = []
        center = (200, 450)
        s1 = Starfish(window, center)
        starFishList.append(s1)
        return starFishList

    def drawStarfish(starFishList):
        for i in range(0, len(starFishList)):
            starFishList[i].draw()
