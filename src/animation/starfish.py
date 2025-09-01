import pygame
import math
from constants import Colors,FishColors
import animation.drawings
import random
import time

class Starfish:
    def __init__(self, window, name: str, center, length, arm_count = 5):
        self.window = window
        self.name = name
        self.center = center
        self.color = FishColors.orange
        self.arm_count = arm_count
        self.arm_length = length
        self.arm_width = random.randrange(int(length/4),length)
        self.angle = random.randrange(int(-(360/arm_count)),int(360/arm_count))

        self.playing = True

        self.armList : list[Arm] =[]
        
        

    def body(self):
        pentagone = animation.drawings.getPolygonPoints(self.arm_count,self.center[0],self.center[1],self.arm_length/3)
        pivotedPentagone = animation.drawings.pivotTriangles(self.center,pentagone,self.angle)
        return pivotedPentagone
    
    def arms(self):
        pentagone = self.body()
        borderPoints = []
        for i in range (len(pentagone)):
            borderPoints.append((pentagone[i][1],pentagone[i][2]))
        '''triangles = []
        for i, border in enumerate(borderPoints):
            anim = self.arm_animations[i]
            animated_length = (self.arm_length/3*2) * anim['current_length_multiplier']

            mid_x,mid_y = animation.drawings.getMiddleOfTwoPoints(border[0],border[1])
            
            # perpendicular direction
            center_to_mid_x = mid_x - self.center[0]
            center_to_mid_y = mid_y - self.center[1]
            
            # Normalize direction
            length = math.sqrt(center_to_mid_x**2 + center_to_mid_y**2)
            if length > 0:
                norm_x = center_to_mid_x / length
                norm_y = center_to_mid_y / length
            else:
                norm_x, norm_y = 0, 1
            
            cos_offset = math.cos(math.radians(anim['current_angle_offset']))
            sin_offset = math.sin(math.radians(anim['current_angle_offset']))
            
            rotated_x = norm_x * cos_offset - norm_y * sin_offset
            rotated_y = norm_x * sin_offset + norm_y * cos_offset
            
            # apex with animation
            apex_x = mid_x + rotated_x * animated_length
            apex_y = mid_y + rotated_y * animated_length
            
            triangles.append((border[0], border[1], (apex_x, apex_y)))'''
        
        typeArm = ["Head","Arm","Leg","Leg","Arm"]
        

        for type in typeArm:
            self.armList.append(Arm(self.window,type,self.arm_length,self.arm_count,self.center,self))
        
        for arm in self.armList:
            triangles = arm.create(borderPoints)
        
        return triangles

    def draw(self, borders: bool = True):
        self.color = Colors.patrick if self.playing else FishColors.orange

        arms = self.arms()
        pentagone = self.body()

        # Draw borders
        if borders:
            for t in arms:
                pygame.draw.polygon(self.window,Colors.black,t,5)

        # Draw arms and body (always)
        for triangle in pentagone:
            pygame.draw.polygon(self.window,self.color,triangle)

        for triangle in arms:
            pygame.draw.polygon(self.window,self.color,triangle)
        
        if self.playing and self.arm_count == 5:
            self.drawPatrick()


    def drawPatrick(self):
        cx,cy = self.center
        length = self.arm_length
        width = self.arm_width

        arms = self.arms()
        
        middles = []
        for arm in arms:
            middles.append(animation.drawings.centerOfTriangle(arm))

        print(middles)
        
        left_eye = animation.drawings.getEllipseTriangles(middles[0][0]-math.cos(self.angle), middles[0][1]-math.sin(self.angle), length/10/2, length/10)
        right_eye = animation.drawings.getEllipseTriangles(middles[0][0]+math.cos(self.angle), middles[0][1]+math.sin(self.angle), length/10/2, length/10)
        left_pupil = animation.drawings.getEllipseTriangles(cx-length/18, cy-length/4, length/20/2, width/20)
        right_pupil = animation.drawings.getEllipseTriangles(cx+length/18, cy-length/4, length/20/2, width/20)

        for triangle in left_eye:
            pygame.draw.polygon(self.window, Colors.white, triangle)
        for triangle in right_eye:
            pygame.draw.polygon(self.window, Colors.green, triangle)
        for triangle in left_pupil:
            pygame.draw.polygon(self.window, Colors.black, triangle)
        for triangle in right_pupil:
            pygame.draw.polygon(self.window, Colors.black, triangle)
        
        mouth = animation.drawings.getEllipseTriangles(cx,cy,width/15,length/18)
        for triangle in mouth:
            pygame.draw.polygon(self.window,Colors.black,triangle)
        
    def update(self,):
        if self.playing:
            for arm in self.armList:
                arm.update(True)


class Arm:
    def __init__(self, window, name : str, length, arm_count,starCenter,star):
        self.window = window
        self.name = name
        self.color = FishColors.orange
        self.triangles = []
        self.arm_count = arm_count
        self.arm_length = length
        self.arm_width = random.randrange(int(length/4),length)
        self.starCenter = starCenter
        self.star = star
        

        # Animation properties for moving arms
        self.arm_animations = []
        self.base_time = time.time()
        
        for i in range(self.arm_count):
            arm_animation = {
                'base_angle': (360 / self.arm_count) * i,
                'wave_amplitude': random.uniform(10, 25),
                'wave_frequency': random.uniform(0.5, 2.0), 
                'phase_offset': random.uniform(0, 2 * math.pi),
                'length_variation': random.uniform(0.8, 1.2), 
                'current_angle_offset': 0,  
                'current_length_multiplier': 1.0
            }
            self.arm_animations.append(arm_animation)
    
    def create (self, borderPoints):
        for i, border in enumerate(borderPoints):
            anim = self.arm_animations[i]
            animated_length = (self.arm_length/3*2) * anim['current_length_multiplier']

            mid_x,mid_y = animation.drawings.getMiddleOfTwoPoints(border[0],border[1])
            
            # perpendicular direction
            center_to_mid_x = mid_x - self.starCenter[0]
            center_to_mid_y = mid_y - self.starCenter[1]
            
            # Normalize direction
            length = math.sqrt(center_to_mid_x**2 + center_to_mid_y**2)
            if length > 0:
                norm_x = center_to_mid_x / length
                norm_y = center_to_mid_y / length
            else:
                norm_x, norm_y = 0, 1
            
            cos_offset = math.cos(math.radians(anim['current_angle_offset']))
            sin_offset = math.sin(math.radians(anim['current_angle_offset']))
            
            rotated_x = norm_x * cos_offset - norm_y * sin_offset
            rotated_y = norm_x * sin_offset + norm_y * cos_offset
            
            # apex with animation
            apex_x = mid_x + rotated_x * animated_length
            apex_y = mid_y + rotated_y * animated_length
            
            self.triangles.append((border[0], border[1], (apex_x, apex_y)))

        return self.triangles
    
    def moveArms(self, move: bool):
        if not move:
            # Reset arms to base position
            for anim in self.arm_animations:
                anim['current_angle_offset'] = 0
                anim['current_length_multiplier'] = 1.0
            return
        
        current_time = time.time() - self.base_time
        
        # Update each arm's animation
        for i, anim in enumerate(self.arm_animations):
            wave_time = current_time * anim['wave_frequency'] + anim['phase_offset']
            angle_wave = math.sin(wave_time) * anim['wave_amplitude']
            anim['current_angle_offset'] = angle_wave
            
            # Calculate subtle length variation (breathing effect)
            length_wave_time = current_time * (anim['wave_frequency'] * 0.5) + anim['phase_offset']
            length_variation = 1.0 + math.sin(length_wave_time) * 0.1  # ±10% length variation
            anim['current_length_multiplier'] = length_variation * anim['length_variation']

    def update(self, move_arms=False):
        move_arms = self.star.playing
        if move_arms:
            self.moveArms(move_arms)
        else :
            return