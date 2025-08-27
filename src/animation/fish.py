import pygame

from random import randrange, random
from constants import FishColors,Colors
import animation.drawings 
import math
from constants import Direction
import time

class Fish:
    global listTriangles
    global fistColor
    global secondColor

    def __init__(self, window: pygame.Surface, name: str, color, center, length, height, direction):
        self.window = window
        self.name = name
        self.center = center
        self.length = length
        self.height = height
        self.direction = direction
        self.color = color
        self.firstColor = color
        self.secondColor = (randrange(255), randrange(255), randrange(255))
        self.listTriangles = [
            (
                (self.center[0] - 75, self.center[1] - 25),
                (self.center[0] - 75, self.center[1] + 25),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 50, self.center[1] - 17),
                (self.center[0] - 25, self.center[1] - 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 50, self.center[1] + 17),
                (self.center[0] - 25, self.center[1] + 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 25, self.center[1] - 50),
                (self.center[0] + 25, self.center[1] - 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] - 25, self.center[1] + 50),
                (self.center[0] + 25, self.center[1] + 50),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] + 25, self.center[1] - 50),
                (self.center[0] + 50, self.center[1] - 17),
                (self.center[0], self.center[1]),
            ),
            (
                (self.center[0] + 25, self.center[1] + 50),
                (self.center[0] + 50, self.center[1] + 17),
                (self.center[0], self.center[1]),
            ),
        ]
        self.fishMouth = FishMouth(
            window,
            length = 2 * self.length / 5,
            maxAngleDeg = 65,
            parent = self
        )
        self.playing = True
        self.speed = 70 + randrange(81) # pixels / second
        self.lastNoteTime = time.time()
        self.enabled = True

    def __str__(self):
        return f"{self.name}, {self.color}"


    def draw(self):
        center =self.center
        rx = self.length
        ry = self.height
        segments = 40
        cx, cy = center
        air = (math.pi*self.length*self.height)/60
        irisRadius = air/10
        pupilRadius = air/15
        

        if self.direction == Direction.LEFT:
            eyeCenter = (cx - rx/2, cy - ry/2)
            endTail = cx + rx + self.length/4
            midTail = cx + rx + self.length/8

            nageoireTopX = cx + rx/2
            nageoireLeftX = nageoireTopX - rx/4
            nageoireRightX = nageoireTopX + rx/4
            nageoireInDownY = cy + ry- ry/2
            nageoireInUpY = cy - ry + ry/2

            
        else : # right
            eyeCenter = (cx + rx/2, cy - ry/2)
            endTail = cx - rx - self.length/4
            midTail = cx - rx - self.length/8

            nageoireTopX = cx - rx/2
            nageoireLeftX = nageoireTopX + rx/4
            nageoireRightX = nageoireTopX - rx/4
            nageoireInDownY = cy + ry- ry/2
            nageoireInUpY = cy - ry + ry/2
            
        
        topTailY = cy - ry
        downTailY = cy + ry
        dorsalTopY = cy - ry - ry/2
        dorsalDownY = cy - ry
        dorsalLeftX = cx - rx/4
        dorsalRightX = cx + rx/4

        #down
        nageoireTopDownY = cy + ry
        nageoireDownDownY = nageoireTopDownY + ry/2
        #up
        nageoireTopUpY = cy - ry
        nageoireDownUpY = nageoireTopUpY - ry/2
        
        self.color = self.secondColor if self.playing else self.firstColor

        # body
        Triangles = animation.drawings.getEllipseTriangles(cx, cy, rx, ry, segments)
        for triangle in Triangles:
            pygame.draw.polygon(self.window, self.color, triangle)

        # tail
        pygame.draw.polygon(self.window, self.color,((endTail, topTailY),(midTail, cy),(cx, cy)))
        pygame.draw.polygon(self.window, self.color,((endTail, downTailY),(midTail, cy),(cx, cy)))

        # iris
        eyeTriangles = animation.drawings.getEllipseTriangles(eyeCenter[0], eyeCenter[1], irisRadius, irisRadius, segments=20)
        for triangle in eyeTriangles:
            pygame.draw.polygon(self.window, Colors.orange, triangle)
        
        # pupil
        eyeTriangles = animation.drawings.getEllipseTriangles(eyeCenter[0], eyeCenter[1], pupilRadius, pupilRadius, segments=20)
        for triangle in eyeTriangles:
            pygame.draw.polygon(self.window, Colors.black, triangle)

        # dorsal fin for long fish
        if self.length > 50 and self.height < 30:
            pygame.draw.polygon(self.window, self.color, ((cx, dorsalTopY),(dorsalRightX, dorsalDownY),(dorsalLeftX, dorsalDownY)))
        
        if self.height>= 30:
            # top fin
            pygame.draw.polygon(self.window, self.color, ((nageoireTopX, nageoireDownUpY),(nageoireRightX, nageoireInUpY),(nageoireLeftX, nageoireTopUpY)))
            # bottom fin
            pygame.draw.polygon(self.window, self.color, ((nageoireTopX, nageoireDownDownY),(nageoireRightX, nageoireInDownY),(nageoireLeftX, nageoireTopDownY)))
        
        self.fishMouth.draw()

    def animate(self, deltaTime):
        if not self.enabled:
            return
        
        if self.center[0] < - self.length / 2 or self.center[0] > self.window.get_size()[0] + self.length / 2:
            self.enabled = False
        
        self.fishMouth.animate(deltaTime)
        self.center = (self.center[0] + (-1 if self.direction == Direction.LEFT else 1) * deltaTime * self.speed, self.center[1])
        
        if (time.time() - self.lastNoteTime < 5):
            if self.center[0] <= - self.length:
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
    
    def drawBorder(self, bordersize = 1):
        for i in range(len(self.listTriangles)):
            pygame.draw.polygon(
                self.window, Colors.black, self.listTriangles[i], width=(bordersize+3)
            )
    
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
    def __init__(self, window, length, maxAngleDeg: int, parent = Fish):
        self.window = window
        self.length = length
        self.maxAngle = maxAngleDeg
        self.isOpening = False
        self.angleDeg = 0.05
        self.timeToClose = 0
        self.parent = parent
    
    def animate(self, deltaTime):
        """
        if self.angleDeg < self.maxAngle:
            self.angleDeg += 20
        else:
            self.isOpening = False
        """
        if self.angleDeg > 0.05:
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

        pygame.draw.polygon(self.window, Colors.bgColor, triangle)