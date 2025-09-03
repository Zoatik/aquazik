import pygame
from constants import Direction, Colors
import animation.drawings
from animation.starfish import Starfish
import random
import math

# https://www.thecraftballoon.com/wp-content/uploads/2021/03/CRAB-CRAFT-FEATURED-IMAGE.jpg
class Crab:

    def __init__(self, window: pygame.Surface, center: tuple[int, int], length:float, musicStartTime, bpm):
        self.window = window
        self.center = center
        self.length = length
        self.height = (50/90) * length
        self.color = Colors.red

        # body parts
        self.arms = CrabArms(self)
        self.mouth = CrabMouth(self)
        self.eyes = CrabEyes(self)
        self.legs = CrabLegs(self)

        # animation
        self.arrived = True
        self.nextMove = (0,0)
        self.speed = 150

        # beat
        self.bpm = bpm
        self.music_start_time = musicStartTime
        self.seconds_per_beat = 60.0 / bpm
        self.beat_timer = 0.0


    def chooseNewDestination(self, starFishList: list[Starfish]):
        self.arrived = False
        self.nextMove = starFishList[random.randrange(len(starFishList))].center

    def move(self, deltaTime, starFishList):
        if self.arrived:
            self.chooseNewDestination(starFishList)

        speedChange = deltaTime * self.speed

        if abs(self.center[0] - self.nextMove[0]) >= 10:
            if self.center[0] > self.nextMove[0]:
                speedChange *= -1
            self.center = (self.center[0] + speedChange, self.center[1])
        elif abs(self.center[1] - self.nextMove[1]) >= 10:
            if self.center[1] > self.nextMove[1]:
                speedChange *= -1
            self.center = (self.center[0], self.center[1] + speedChange)
            # move
        else:
            # plus besoin de bouger donc 
            self.chooseNewDestination(starFishList)

        # BEAT TIMER
        self.beat_timer += deltaTime
        
        # Check if it's time for a beat
        if self.beat_timer >= self.seconds_per_beat:
            self.on_beat()
            # Reset timer, preserving overflow for accuracy
            self.beat_timer -= self.seconds_per_beat

        # other body parts
        self.legs.update(deltaTime)
        self.eyes.update(deltaTime)
        
    def on_beat(self):
        self.arms.hand_opened = not self.arms.hand_opened


    def getBodyTriangles(self):
        return animation.drawings.getEllipseTriangles(self.center[0], self.center[1], self.length, self.height, 20)

    def drawBorders(self):
        for t in self.getBodyTriangles():
            pygame.draw.polygon(self.window, Colors.black, t, 5)
        
        self.eyes.drawBorders()
        self.arms.drawBorders()

    def draw(self):
        self.legs.draw()

        self.drawBorders()

        self.eyes.draw()
        self.arms.draw()

        for t in self.getBodyTriangles():
            pygame.draw.polygon(self.window, self.color, t)
        
        self.mouth.draw()

class CrabLegs:
    def __init__(self, parent: Crab):
        self.parent = parent
        self.triangles = []

        # Animation properties
        self.animation_timer = 0.0
        self.animation_speed = 3.0  # Speed multiplier for leg movement
        self.max_angle = 8  # Maximum rotation angle in degrees
        self.leg_offset = 0.5  # Phase offset between left and right legs
        

    def update(self, delta_time):
        """Update animation timer"""
        self.animation_timer += delta_time * self.animation_speed

    def getAnimationAngle(self, side_multiplier, leg_index):
        """Calculate the current animation angle for a specific leg"""
        # Create phase offset for each leg to create walking effect
        phase = self.animation_timer + (leg_index * 0.3) + (side_multiplier * self.leg_offset)
        
        # Use sine wave for smooth back-and-forth motion
        angle = math.sin(phase) * self.max_angle
        return angle

    def prepTriangles(self):
        self.triangles = []

        for i in range(2):
            op = -1 if i % 2 == 0 else 1
            triangles = [
                [
                    (self.parent.center[0] + op * self.parent.length* 0.9, self.parent.center[1]- self.parent.height / 6),
                    (self.parent.center[0] + op * self.parent.length* 0.9, self.parent.center[1] + self.parent.height / 6),
                    (self.parent.center[0] + op * self.parent.length * 1.2, self.parent.center[1] + self.parent.height / 8),
                ],
                [
                    (self.parent.center[0] + op * self.parent.length * 1.2, self.parent.center[1]),
                    (self.parent.center[0] + op * self.parent.length * 1.1, self.parent.center[1] + self.parent.height / 4),
                    (self.parent.center[0] + op * self.parent.length * 1.3, self.parent.center[1] + self.parent.height / 3)
                ]
            ]

            # Create 3 legs per side with animation
            for leg_index in range(3):
                # Calculate animated angle for this specific leg
                animated_angle = self.getAnimationAngle(op, leg_index)
                base_angle = op * leg_index * 10  # Original static angle
                final_angle = base_angle + animated_angle
                
                for triangle in triangles:
                    animated_triangle = animation.drawings.pivotTriangle(
                        (self.parent.center[0], self.parent.center[1] - self.parent.height), 
                        triangle, 
                        final_angle
                    )
                    self.triangles.append((self.parent.color, animated_triangle))

    def drawBorders(self):
        self.prepTriangles()

    def draw(self):
        self.prepTriangles()
        for (color, triangle) in self.triangles:
            pygame.draw.polygon(self.parent.window, Colors.black, triangle, 5)
        for (color, triangle) in self.triangles:
            pygame.draw.polygon(self.parent.window, color, triangle)
            

class CrabArms:
    def __init__(self, parent: Crab):
        self.parent = parent
        self.length = parent.height / 1.2
        self.radius = parent.height / 10
        self.hand_opened = True

        self.triangles = []

    def prepTriangles(self):
        self.triangles = []
        for i in range(2):
            op = -1 if i % 2 == 0 else 1
            arm1 = (self.parent.center[0] - op * self.parent.length / 1.2, self.parent.center[1] -  self.parent.height * 1.3)
            arm2 = (self.parent.center[0] - op * 0.95 * self.parent.length, self.parent.center[1] - 1.1 * self.parent.height)
            self.triangles.append((self.parent.color, [
                (self.parent.center[0] - op * self.parent.length / 2, self.parent.center[1]),
                arm1,
                arm2,
            ]))
            joint_arm_hand = animation.drawings.getMiddleOfTwoPoints(arm1, arm2)
            end_hand = (self.parent.center[0] - op * self.parent.length / 1.2, self.parent.center[1] - self.parent.height * 2)
            (cx, cy) = animation.drawings.getMiddleOfTwoPoints(joint_arm_hand, end_hand)
            (ox, oy) = animation.drawings.getMiddleOfTwoPoints((cx,cy), end_hand)

            base_angle = 0 if op == 1 else 180
            variation = 30 if self.hand_opened else 10

            ellipse = animation.drawings.getEllipseArcTriangles(cx, cy, self.parent.length / 3, self.parent.height / 2.3, base_angle + variation, base_angle - variation, segments=40)

            hand = animation.drawings.pivotTriangles((cx, cy), ellipse, op * -80)
            for t in hand:
                self.triangles.append((self.parent.color, t))

    def drawBorders(self):
        self.prepTriangles()
        for (color, triangle) in self.triangles:
            pygame.draw.polygon(self.parent.window, Colors.black, triangle, 5)

    def draw(self):
        self.prepTriangles()
        for (color, triangle) in self.triangles:
            pygame.draw.polygon(self.parent.window, color, triangle)
    

class CrabMouth:
    def __init__(self, parent: Crab):
        self.parent = parent
        self.MOUTH_SIZE = self.parent.length / 9

    def draw(self):
        triangles = []

        # Black
        triangles.append((Colors.black, animation.drawings.getEllipseArcTriangles(self.parent.center[0], self.parent.center[1] - self.parent.length / 3, self.parent.length / 3 + self.MOUTH_SIZE / 2, self.parent.height / 2 + self.MOUTH_SIZE / 2, 0, 180)))
        
        # Red
        triangles.append((self.parent.color,animation.drawings.getEllipseArcTriangles(self.parent.center[0], self.parent.center[1] - self.parent.length / 3, self.parent.length / 3, self.parent.height / 2, 0, 180)))

        for clr, triangles in triangles:
            for t in triangles:
                pygame.draw.polygon(self.parent.window, clr, t)

class CrabEyes:
    def __init__(self, parent):
        self.parent = parent
        self.triangles = []
        self.eye_position = (0,0) # relative to eye center -> -1 to 1
        self.eye_radius = parent.length / 5

        # Animation properties
        self.look_timer = 0.0
        self.look_duration = 3.0  # How long to look in one direction
        self.transition_speed = 2.0  # How fast eyes move between positions
        
        # Current and target positions
        self.current_position = (0.0, 0.0)
        self.target_position = (0.0, 0.0)
        
        # Look positions
        self.look_positions = [
            (0.0, 0.0),    # Center
            (-0.7, 0.0),   # Left
            (0.7, 0.0),    # Right
            (-0.5, -0.3),  # Upper left
            (0.5, -0.3),   # Upper right
            (0.0, 0.4),    # Down
        ]
        
        # Start with random position
        self.target_position = random.choice(self.look_positions)
    
    def update(self, delta_time):
        """Update eye animation"""
        self.look_timer += delta_time
        
        # Time to choose a new target?
        if self.look_timer >= self.look_duration:
            self.target_position = random.choice(self.look_positions)
            self.look_duration = random.uniform(2.0, 4.0)
            self.look_timer = 0.0
        
        # Move current position toward target
        dx = self.target_position[0] - self.current_position[0]
        dy = self.target_position[1] - self.current_position[1]
        
        move_speed = self.transition_speed * delta_time
        self.current_position = (
            self.current_position[0] + dx * move_speed,
            self.current_position[1] + dy * move_speed
        )
        
        # Clamp to valid range and update eye_position
        self.eye_position = (
            max(-1.0, min(1.0, self.current_position[0])),
            max(-1.0, min(1.0, self.current_position[1]))
        )

    def prepTriangles(self):
        self.triangles = []
        EYE_HEIGHT = 1.9

        for i in range(2):
            eyeball_position = (self.parent.center[0] + (-1 if i % 2 == 0 else 1) * self.parent.length / 3, self.parent.center[1] - EYE_HEIGHT * self.parent.height)

            self.triangles.append((self.parent.color,[
                (self.parent.center[0] + (-1 if i % 2 == 0 else 1) * self.parent.length / 4, self.parent.center[1]),
                (eyeball_position[0], self.parent.center[1]),
                (self.parent.center[0] + (-1 if i % 2 == 0 else 1) * self.parent.length / 4, eyeball_position[1])
            ]))
            self.triangles.append((self.parent.color,[
                eyeball_position,
                (eyeball_position[0], self.parent.center[1]),
                (self.parent.center[0] + (-1 if i % 2 == 0 else 1) * self.parent.length / 4, self.parent.center[1] - EYE_HEIGHT * self.parent.height)
            ]))
            
            eyeball = animation.drawings.getPolygonPoints(20, eyeball_position[0], eyeball_position[1], self.eye_radius)
            eyeball_border = animation.drawings.getPolygonPoints(20, eyeball_position[0], eyeball_position[1], self.eye_radius + 3)
            
            for t in eyeball_border:
                self.triangles.append((Colors.black, t))

            for t in eyeball:
                self.triangles.append((Colors.white, t))

            pupilRadius = self.eye_radius / 2
            pupil = animation.drawings.getPolygonPoints(20, eyeball_position[0] + self.eye_position[0] * (self.eye_radius - pupilRadius), eyeball_position[1] + self.eye_position[1] * (self.eye_radius - pupilRadius), pupilRadius)

            for t in pupil:
                self.triangles.append((Colors.black,t))

    def drawBorders(self):
        self.prepTriangles()
        for (color, triangle) in self.triangles:
            if color != Colors.black:
                pygame.draw.polygon(self.parent.window, Colors.black, triangle, 5)

    def draw(self):
        self.prepTriangles()
        for (color, triangle) in self.triangles:
            pygame.draw.polygon(self.parent.window, color, triangle)
