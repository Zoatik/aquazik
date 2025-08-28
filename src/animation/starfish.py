import pygame
import math
from constants import Colors,FishColors
import animation.drawings
import random


class Starfish:
    def __init__(self, window, name: str, center, length):
        self.window = window
        self.name = name
        self.center = center
        self.color = FishColors.orange
        self.arm_count = 5
        self.arm_length = length
        self.arm_width = random.randrange(int(length/4),length)

        self.playing = True

    # Function to create a single triangle for an arm
    def triangle_arm(self, center, length, width, angle):
        x, y = center
        # tip of the arm
        tip_x = x + length * math.cos(math.radians(angle))
        tip_y = y + length * math.sin(math.radians(angle))
        # base corners
        left_x = x + width / 2 * math.cos(math.radians(angle + 90))
        left_y = y + width / 2 * math.sin(math.radians(angle + 90))
        right_x = x + width / 2 * math.cos(math.radians(angle - 90))
        right_y = y + width / 2 * math.sin(math.radians(angle - 90))
        return [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)]

    def draw(self, borders: bool = False):
        self.color = Colors.patrick if self.playing else FishColors.orange
        if not borders:
            self.draw(borders=True)
        for i in range(self.arm_count):
            angle = i * (360 / self.arm_count) - 90  # rotate to start pointing upwards
            points = self.triangle_arm(
                self.center, self.arm_length, self.arm_width, angle
            )
            if borders:
                pygame.draw.polygon(self.window, Colors.black, points, width=3)
            else:
                pygame.draw.polygon(self.window, self.color, points)

        if self.playing and self.arm_count == 5:
            self.drawPatrick()
        else:
            pass


    def drawPatrick(self):
        cx,cy = self.center
        length = self.arm_length
        width = self.arm_width
        
        left_eye = animation.drawings.getEllipseTriangles(cx-length/15, cy-length/4, length/10/2, length/10)
        right_eye = animation.drawings.getEllipseTriangles(cx+length/15, cy-length/4, length/10/2, length/10)
        left_pupil = animation.drawings.getEllipseTriangles(cx-length/18, cy-length/4, length/20/2, width/20)
        right_pupil = animation.drawings.getEllipseTriangles(cx+length/18, cy-length/4, length/20/2, width/20)

        for triangle in left_eye:
            pygame.draw.polygon(self.window, Colors.white, triangle)
        for triangle in right_eye:
            pygame.draw.polygon(self.window, Colors.white, triangle)
        for triangle in left_pupil:
            pygame.draw.polygon(self.window, Colors.black, triangle)
        for triangle in right_pupil:
            pygame.draw.polygon(self.window, Colors.black, triangle)
        

        
