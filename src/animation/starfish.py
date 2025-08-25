import pygame
import math
from constants import Colors,FishColors


class Starfish:
    def __init__(self, window, name: str, center, length):
        self.window = window
        self.name = name
        self.center = center
        self.color = FishColors.orange
        self.arm_count = 5
        #self.arm_length = 50
        self.arm_length = length
        #self.arm_width = 25
        self.arm_width = length / 2

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

    # change the color of the fish to a random color
    def animStarfish(self):
        """temp = pygame.time.get_ticks()
        if pygame.time.get_ticks() - temp < 1000:
            self.color = FishColors.yellow
        else.color = FishColors.orange"""
        if self.color == FishColors.orange:
            self.color = FishColors.green
        else:
            self.color = FishColors.orange
