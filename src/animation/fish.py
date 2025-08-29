import pygame

from random import randrange, random, choice
from constants import FishColors,Colors
import animation.drawings 
import math
from constants import Direction
import time
from audio_processing.midi_reader import MidiNote
from enum import Enum, auto

class FishType(Enum):
    NORMAL = auto()
    LONG = auto()
    MOON = auto()

class Fish:
    global listTriangles
    global fistColor
    global secondColor

    def __init__(self, window: pygame.Surface, base_note: MidiNote, direction = choice([Direction.LEFT, Direction.RIGHT]), color = FishColors.darkGrayTruite):
        # saved parameters
        self.window = window
        self.direction = direction
        self.color = color
        self.firstColor = color
        self.secondColor = FishColors.lightGrayTruite

        # calculated from given parameters
        self.name = base_note.get_real_note()[:-1]
        distance = base_note.velocity / 6
        self.center = (
            (distance if direction == Direction.RIGHT else window.get_size()[0] - base_note.velocity / 6),
            randrange(50, int(window.get_size()[1] / 2 - 50))
            #int(base_note.get_real_note()[-1]) * (window.get_size()[1] / 9)
        )

        # random values
        self.speed = 70 + randrange(81) # pixels / second, max = 150
        self.length = randrange(30,70)
        self.height = randrange(15,40)

        # default values
        self.playing = True
        self.lastNoteTime = time.time()
        self.enabled = True
        self.spawnTime = time.time()
        self.flowOffset = random() * 4 - 2
        self.angleDeg = 0

        # fish body parts / animation
        self.fishType: FishType = FishType.LONG if self.length > 50 and self.height < 30 else FishType.MOON if self.height >= 30 else FishType.NORMAL
        self.fishMouth = FishMouth(
            window,
            maxAngleDeg = 65,
            parent = self
        )
        self.fishTail = FishTail(self)

    def __str__(self):
        return f"{self.name}, {self.color}"

    def draw(self):
        cx, cy = self.center
        air = (math.pi*self.length*self.height)/60
        irisRadius = air/10
        pupilRadius = air/15

        eyeCenter = (cx - (1 if self.direction == Direction.LEFT else -1) * self.length/2, cy - self.height/2)
        nageoireTopX = cx + (1 if self.direction == Direction.LEFT else -1) * self.length/2
        nageoireLeftX = nageoireTopX - (1 if self.direction == Direction.LEFT else -1) * self.length/4
        nageoireRightX = nageoireTopX + (1 if self.direction == Direction.LEFT else -1) * self.length/4
        nageoireInDownY = cy + self.height /2
        nageoireInUpY = cy - self.height /2
        
        self.color = self.secondColor if self.playing else self.firstColor

        # calculate triangles for long fish
        dorsalTopY = cy - 3* self.height/2
        dorsalDownY = cy - self.height
        dorsalLeftX = cx - self.length/4
        dorsalRightX = cx + self.length/4
        dorsalFinTriangle = [(cx, dorsalTopY),(dorsalRightX, dorsalDownY),(dorsalLeftX, dorsalDownY)]

        # calculate triangles for moon fish
        #down
        nageoireTopDownY = cy + self.height
        nageoireDownDownY = nageoireTopDownY + self.height/2
        #up
        nageoireTopUpY = cy - self.height
        nageoireDownUpY = nageoireTopUpY - self.height/2
        nageoireTopDown = [
            [(nageoireTopX, nageoireDownUpY),(nageoireRightX, nageoireInUpY),(nageoireLeftX, nageoireTopUpY)],
            [(nageoireTopX, nageoireDownDownY),(nageoireRightX, nageoireInDownY),(nageoireLeftX, nageoireTopDownY)]
        ]

        # get all needed triangles
        bodyTriangles = animation.drawings.pivotTriangles(
            self.center,
            animation.drawings.getEllipseTriangles(cx, cy, self.length, self.height, segments = 80),
            self.angleDeg
        )
        
        # draw borders
        for t in bodyTriangles:
            pygame.draw.polygon(self.window, Colors.black, t, width=5)
        if self.fishType == FishType.MOON:
            nageoireTopDown = [animation.drawings.pivotTriangle(self.center, t, self.angleDeg) for t in nageoireTopDown]
            for t in nageoireTopDown:
                pygame.draw.polygon(self.window, Colors.black, t, width=5)
        if self.fishType == FishType.LONG:
            dorsalFinTriangle = animation.drawings.pivotTriangle(self.center, dorsalFinTriangle, self.angleDeg)
            pygame.draw.polygon(self.window, Colors.black, dorsalFinTriangle, width=5)

        self.fishTail.drawBorder()

        # draw body parts (real color)
        for t in bodyTriangles:
            pygame.draw.polygon(self.window, self.color, t)
        
        # iris
        eyeIrisTriangles = animation.drawings.pivotTriangles(
            self.center,
            animation.drawings.getEllipseTriangles(eyeCenter[0], eyeCenter[1], irisRadius, irisRadius, segments=20),
            self.angleDeg
        )
        for t in eyeIrisTriangles:
            pygame.draw.polygon(self.window, Colors.orange, t)
        
        # pupils
        eyePupilsTriangles = animation.drawings.pivotTriangles(
            self.center,
            animation.drawings.getEllipseTriangles(eyeCenter[0], eyeCenter[1], pupilRadius, pupilRadius, segments=20),
            self.angleDeg
        )
        for t in eyePupilsTriangles:
            pygame.draw.polygon(self.window, Colors.black, t)

        # dorsal fin for long fish
        if self.fishType == FishType.LONG:
            pygame.draw.polygon(self.window, self.color, dorsalFinTriangle)

        #poisson lune
        if self.fishType == FishType.MOON:
            for t in nageoireTopDown:
                pygame.draw.polygon(self.window, self.color, t)
        
        # rest of body parts
        self.fishMouth.draw()
        self.fishTail.draw()

    def animate(self, deltaTime):
        if not self.enabled:
            return
        
        if self.center[0] < - self.length or self.center[0] > self.window.get_size()[0] + self.length:
            self.enabled = False
        
        self.fishMouth.animate(deltaTime)
        self.fishTail.animate(deltaTime)

        t = time.time() - self.spawnTime + self.flowOffset

        norm = (self.speed - 70) / (150 - 70)
        # 70  -> 0
        # 150 -> 1

        amplitude = (1/2) + (1/15 - 1/2) * norm
        vx = (-1 if self.direction == Direction.LEFT else 1) * self.speed
        vy = math.cos(t) * amplitude  # dérivée de sin pour l'angle

        # mise à jour de la position
        self.center = (
            self.center[0] + vx * deltaTime,
            self.center[1] + math.sin(t) * amplitude
        )

        self.angleDeg = (1 if self.direction == Direction.RIGHT else -1) * math.degrees(math.atan(vy))
        
        if (time.time() - self.lastNoteTime < 5):
            if self.center[0] <= self.length:
                self.direction = Direction.RIGHT
            elif self.center[0] >= self.window.get_size()[0] - self.length:
                self.direction = Direction.LEFT
    
    def openMouth(self, velocity, noteTime):
        # maximum de velocity est 127 ce qui correspondra à un angle de 70°
        # minimum de 20°
        self.fishMouth.maxAngle = (50 / 127) * velocity + 20
        self.fishMouth.angleDeg = self.fishMouth.maxAngle

        # add 650 ms to time to close so the animation is more visible
        self.fishMouth.timeToClose = noteTime + 0.65
    
    def createBubble(self, window):
        return Bubble(
            window,
            (
                self.center[0] + self.direction.value * (self.length + random() * 5),
                self.center[1] - 10, # 10 pixels plus haut pour mieux voir la bouche
            ),
            radius = 5 + random() * 20,
        )

class Bubble:
    out_of_bounds: bool = False

    def __init__(
        self, window: pygame.Surface, starting_pos: tuple[int, int], radius: int
    ):
        self.window = window
        self.pos = starting_pos
        self.radius = radius

    def move_and_draw(self):
        # bouge +- en x et toujours - en y
        self.pos = (
            self.pos[0] + (0 if randrange(0,11) <= 8 else -1 if randrange(2) % 2 == 0 else 1),
            self.pos[1] - int(randrange(2)),
        )

        if (self.pos[1] + self.radius <= 0):
            self.out_of_bounds = True
            return

        points = animation.drawings.getPolygonPoints(
            15, self.pos[0], self.pos[1], self.radius
        )
        # draw bubble's border
        for t in points:
            pygame.draw.polygon(self.window, Colors.black, t, width = 3)
        # draw white part of bubble (inside)
        for t in points:
            pygame.draw.polygon(self.window, Colors.white, t)

class FishMouth:
    def __init__(self, window, maxAngleDeg: int, parent: Fish):
        # saved parameters
        self.window = window
        self.maxAngle = maxAngleDeg
        self.parent = parent

        # calculated parameters
        self.length = (2.15/5) * self.parent.length

        # default values
        self.isOpening = False
        self.angleDeg = 0.01
        self.timeToClose = 0
    
    def animate(self, deltaTime):
        if self.angleDeg > 0.01:
            self.angleDeg -= (deltaTime / self.timeToClose) * self.angleDeg

        self.timeToClose -= deltaTime


    # Calculates new angle and draws
    def draw(self):
        if self.angleDeg == 0:
            return
        
        cx = self.parent.center[0] + (self.parent.direction.value * 3) * self.parent.length / 5
        cy = self.parent.center[1]

        triangle = [
            (cx, cy),
            (cx + (self.parent.direction.value * self.length), cy - math.tan(math.radians(self.angleDeg / 2))*self.length),
            (cx + (self.parent.direction.value * self.length), cy + math.tan(math.radians(self.angleDeg / 2))*self.length)
        ]

        # TODO BORDER
        pygame.draw.polygon(self.window, Colors.bgColor, animation.drawings.pivotTriangle(self.parent.center, triangle, self.parent.angleDeg))

class FishTail:
    def __init__(self, parent: Fish, angleMax = 6):
        self.parent: Fish = parent
        self.angleMax = angleMax
        self.angleDeg: float = angleMax
        self.animation = {
            "currentSeconds": 0,
            "fullTime": abs((self.parent.speed/150) * 1.5 - 1.6),
            "up": False
        }

    def animate(self, deltaTime):
        parent_angle = self.parent.angleDeg

        self.animation["currentSeconds"] += deltaTime
        if self.animation["currentSeconds"] >= self.animation["fullTime"]:
            self.animation["currentSeconds"] -= self.animation["fullTime"]
            self.animation["up"] = not self.animation["up"]

        oscillation = (
            (-1 if self.animation["up"] else 1)
            * self.angleMax * 2
            * (self.animation["currentSeconds"] / self.animation["fullTime"])
            + (1 if self.animation["up"] else -1) * self.angleMax
        )

        # angle total = angle du corps + oscillation de la queue
        self.angleDeg = parent_angle + oscillation
    
    def getTriangles(self):
        topTailY = self.parent.center[1] - self.parent.height
        downTailY = self.parent.center[1] + self.parent.height
        
        midTail = self.parent.center[0] - self.parent.direction.value * self.parent.length * (9/8)
        endTail = self.parent.center[0] - self.parent.direction.value * self.parent.length * (5/4)

        before_pivot = [
            [
                (endTail, topTailY),(midTail, self.parent.center[1]), self.parent.center
            ],
            [
                (endTail, downTailY),(midTail, self.parent.center[1]), self.parent.center
             ]
        ]
        return animation.drawings.pivotTriangles(self.parent.center, before_pivot, self.angleDeg)

    
    def draw(self):
        for t in self.getTriangles():
            pygame.draw.polygon(self.parent.window, self.parent.color, t)
    
    def drawBorder(self, bordersize = 5):
        for t in self.getTriangles():
            pygame.draw.polygon(self.parent.window, Colors.black, t, width=bordersize)
