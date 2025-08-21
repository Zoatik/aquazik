from animation.fish import Fish, Colors
from constants import NOTE_NAMES

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