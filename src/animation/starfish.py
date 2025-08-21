import pygame
import math
from animation.drawings import Colors


class Starfish:
    def __init__(self, window, name: str, center):
        self.window = window
        self.name = name
        self.center = center
        self.color = Colors.orange
        self.arm_count = 5
        self.arm_length = 50
        self.arm_width = 25

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

    def draw(self):
        for i in range(self.arm_count):
            angle = i * (360 / self.arm_count) - 90  # rotate to start pointing upwards
            points = self.triangle_arm(
                self.center, self.arm_length, self.arm_width, angle
            )
            pygame.draw.polygon(self.window, self.color, points)
            pygame.draw.polygon(self.window, Colors.black, points, width=2)

    # change the color of the fish to a random color
    def animStarfish(self):
        """temp = pygame.time.get_ticks()
        while pygame.time.get_ticks() - temp < 1000:
            self.color = Colors.yellow
        self.color = Colors.orange"""
        if self.color == Colors.orange:
            self.color = Colors.yellow
        else:
            self.color = Colors.orange
