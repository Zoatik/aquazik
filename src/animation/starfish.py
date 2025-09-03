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

        # Create arms 
        self.armList : list[Arm] = []
        if arm_count == 5 :
            typeArm = ["Head","Arm","Leg","Leg","Arm"]
        else :
            typeArm = ["Arm" for i in range(arm_count)]
        for armNb, arm_type in enumerate(typeArm):
            self.armList.append(Arm(self.window, arm_type, self.arm_length, self.arm_count, self.center, self, armNb))
        
        

    def body(self):
        pentagone = animation.drawings.getPolygonPoints(self.arm_count,self.center[0],self.center[1],self.arm_length/3)
        pivotedPentagone = animation.drawings.pivotTriangles(self.center,pentagone,self.angle)
        return pivotedPentagone
    
    def arms(self):
        pentagone = self.body()
        borderPoints = []
        for i in range (len(pentagone)):
            borderPoints.append((pentagone[i][1],pentagone[i][2]))
        
        triangles = []
        for armNb, arm in enumerate(self.armList):
            arm_triangles = arm.create(borderPoints, armNb)
            triangles.extend(arm_triangles)
        
        return triangles

    def draw(self, borders: bool = True):
        self.color = Colors.patrick if self.playing else FishColors.orange

        arms = self.arms()
        pentagone = self.body()

        # Draw borders
        if borders:
            for t in arms:
                pygame.draw.polygon(self.window,Colors.black,t,5)

        # Draw arms and body
        for triangle in pentagone:
            pygame.draw.polygon(self.window,self.color,triangle)

        for triangle in arms:
            pygame.draw.polygon(self.window,self.color,triangle)
        
        
        if self.playing and self.arm_count == 5:
            #self.drawPatrick()
            for arm in self.armList :
                #if arm.name == "Head":
                    #arm.drawHead()
                arm.drawPatrick()

    def update(self,):
        if self.playing:
            for arm in self.armList:
                arm.update(True)


class Arm:
    def __init__(self, window, name : str, length, arm_count, starCenter, star :Starfish, arm_index):
        self.window = window
        self.name = name
        self.color = FishColors.orange
        self.triangles = []
        self.arm_count = arm_count
        self.arm_length = length
        self.arm_width = random.randrange(int(length/4),length)
        self.starCenter = starCenter
        self.star = star
        self.angle = star.angle
        self.arm_index = arm_index  # Which arm this object represents

        # Animation properties for this specific arm
        self.arm_animation = {
            'base_angle': (360 / self.arm_count) * arm_index,
            'wave_amplitude': random.uniform(10, 25),
            'wave_frequency': random.uniform(0.5, 2.0), 
            'phase_offset': random.uniform(0, 2 * math.pi),
            'length_variation': random.uniform(0.8, 1.2), 
            'current_angle_offset': 0,  
            'current_length_multiplier': 1.0
        }
        self.base_time = time.time()
    
    def create(self, borderPoints, arm_index):
        # Clear previous triangles
        self.triangles.clear()
        
        # Only create triangle for THIS specific arm
        if arm_index < len(borderPoints):
            border = borderPoints[arm_index]
            anim = self.arm_animation
            animated_length = (self.arm_length/3*2) * anim['current_length_multiplier']

            mid_x, mid_y = animation.drawings.getMiddleOfTwoPoints(border[0], border[1])
            
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
            # Reset arm to base position
            self.arm_animation['current_angle_offset'] = 0
            self.arm_animation['current_length_multiplier'] = 1.0
            return
        
        current_time = time.time() - self.base_time
        
        # Update this arm anim
        anim = self.arm_animation
        wave_time = current_time * anim['wave_frequency'] + anim['phase_offset']
        angle_wave = math.sin(wave_time) * anim['wave_amplitude']
        anim['current_angle_offset'] = angle_wave
        
        # Calculate length variation
        length_wave_time = current_time * (anim['wave_frequency'] * 0.5) + anim['phase_offset']
        length_variation = 1.0 + math.sin(length_wave_time) * 0.1  # ±10% length variation
        anim['current_length_multiplier'] = length_variation * anim['length_variation']

    def drawHead(self):
        head_triangle = self.triangles[0]
        cx, cy = animation.drawings.centerOfTriangle(head_triangle)
            
        eye_size = self.arm_length / 15
        pupil_size = eye_size * 0.4
        mouth_size = self.arm_length / 20

        eye_separation = self.arm_length / 20  
        eyes_toward_tip = self.arm_length / 15
        pupils_separation = 0.2
        mouth_toward_center = self.arm_length / 12 
           
        # direction center-apex
        direction_x = cx - self.starCenter[0]
        direction_y = cy - self.starCenter[1]
            
        # ||direction||
        length = math.sqrt(direction_x**2 + direction_y**2)
        if length > 0:
            norm_x = direction_x / length
            norm_y = direction_y / length
        else:
            norm_x, norm_y = 0, 1
            
        perp_x = -norm_y
        perp_y = norm_x
            
        #eyes pos
        eye_center_x = cx + norm_x * eyes_toward_tip
        eye_center_y = cy + norm_y * eyes_toward_tip
            
        left_eye_x = eye_center_x + perp_x * eye_separation
        left_eye_y = eye_center_y + perp_y * eye_separation
        right_eye_x = eye_center_x - perp_x * eye_separation
        right_eye_y = eye_center_y - perp_y * eye_separation

            

        pupil_offset = eye_separation * pupils_separation
        left_pupil_x = left_eye_x - perp_x * pupil_offset    # Move left (toward center)
        left_pupil_y = left_eye_y - perp_y * pupil_offset
        right_pupil_x = right_eye_x + perp_x * pupil_offset  # Move right (toward center)  
        right_pupil_y = right_eye_y + perp_y * pupil_offset
            
        # mouth pos
        mouth_x = cx - norm_x * mouth_toward_center
        mouth_y = cy - norm_y * mouth_toward_center
            
        # Create
        left_eye = animation.drawings.getEllipseTriangles(left_eye_x, left_eye_y, eye_size, eye_size)
        right_eye = animation.drawings.getEllipseTriangles(right_eye_x, right_eye_y, eye_size, eye_size)
        left_pupil = animation.drawings.getEllipseTriangles(left_pupil_x, left_pupil_y, pupil_size, pupil_size)
        right_pupil= animation.drawings.getEllipseTriangles(right_pupil_x, right_pupil_y, pupil_size, pupil_size)
        mouth = animation.drawings.getEllipseTriangles(mouth_x, mouth_y, mouth_size, mouth_size)
        
        return left_eye,right_eye,left_pupil,right_pupil,mouth

    def drawShorts(self, horizontal_offset=0):
        leg_triangle = self.triangles[0]
        cx, cy = animation.drawings.centerOfTriangle(leg_triangle)
        
        # direction to star center
        direction_x = cx - self.starCenter[0]
        direction_y = cy - self.starCenter[1]
        
        # ||direction||
        length = math.sqrt(direction_x**2 + direction_y**2)
        if length > 0:
            norm_x = direction_x / length
            norm_y = direction_y / length
        else:
            norm_x, norm_y = 0, 1
        
        # Perp
        perp_x = -norm_y
        perp_y = norm_x
        
        # Shorts dimensions
        shorts_width = self.arm_length/2
        shorts_height = self.arm_length/4
        shorts_toStarCenter = self.arm_length / 22
        
        # + star center
        shorts_center_x = cx - norm_x * shorts_toStarCenter
        shorts_center_y = cy - norm_y * shorts_toStarCenter
        
        # horizontal offset
        if self.arm_index == 2:  # Left leg
            shorts_center_x += perp_x * horizontal_offset * self.arm_length
            shorts_center_y += perp_y * horizontal_offset * self.arm_length
        elif self.arm_index == 3:  # Right leg
            shorts_center_x -= perp_x * horizontal_offset * self.arm_length
            shorts_center_y -= perp_y * horizontal_offset * self.arm_length
        
       
        half_width = shorts_width / 2
        half_height = shorts_height / 2
        
        # Top corners
        top_left_x = shorts_center_x - perp_x * half_width - norm_x * half_height
        top_left_y = shorts_center_y - perp_y * half_width - norm_y * half_height
        top_right_x = shorts_center_x + perp_x * half_width - norm_x * half_height
        top_right_y = shorts_center_y + perp_y * half_width - norm_y * half_height
        
        # Bottom corners
        bottom_left_x = shorts_center_x - perp_x * half_width + norm_x * half_height
        bottom_left_y = shorts_center_y - perp_y * half_width + norm_y * half_height
        bottom_right_x = shorts_center_x + perp_x * half_width + norm_x * half_height
        bottom_right_y = shorts_center_y + perp_y * half_width + norm_y * half_height
        
        # Create two triangles to form a rectangle
        triangle1 = [(top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_left_x, bottom_left_y)]
        triangle2 = [(top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (bottom_left_x, bottom_left_y)]
        
        return triangle1, triangle2

    def drawPatrick(self):
        if self.name == "Head":
            
            left_eye,right_eye,left_pupil,right_pupil,mouth = self.drawHead()

            # Draw
            for triangle in left_eye:
                pygame.draw.polygon(self.window, Colors.white, triangle)
            for triangle in right_eye:
                pygame.draw.polygon(self.window, Colors.white, triangle)
            for triangle in left_pupil:
                pygame.draw.polygon(self.window, Colors.black, triangle)
            for triangle in right_pupil:
                pygame.draw.polygon(self.window, Colors.black, triangle)
            for triangle in mouth:
                pygame.draw.polygon(self.window, Colors.black, triangle)
        
        elif self.name == "Leg":
            #(0.05-0.2) toward center / (-0.05 to -0.2)away from center
            shorts_horizontal_offset = 0.02
        
            triangle1, triangle2 = self.drawShorts(shorts_horizontal_offset)

            # Draw
            pygame.draw.polygon(self.window, Colors.green, triangle1)
            pygame.draw.polygon(self.window, Colors.green, triangle2)

            
            for _ in range(2):
                fx = random.randint(int(min([p[0] for p in triangle1])),
                                    int(max([p[0] for p in triangle2])))
                fy = random.randint(int(min([p[1] for p in triangle1])),
                                    int(max([p[1] for p in triangle2])))
                boules = animation.drawings.getEllipseTriangles(fx,fy,int(self.arm_length/10),int(self.arm_length/10))
            for tri in boules :
                pygame.draw.polygon(self.window, Colors.purple, tri)

    def update(self, move_arms=False):
        move_arms = self.star.playing
        if move_arms:
            self.moveArms(move_arms)
        else :
            return