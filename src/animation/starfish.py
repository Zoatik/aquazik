import pygame
import math
from constants import Colors,FishColors
import animation.drawings
import random


class Starfish:
    def __init__(self, window, name: str, center, length, arm_count = 5):
        self.window = window
        self.name = name
        self.center = center
        self.color = FishColors.orange
        self.arm_count = arm_count
        self.arm_length = length
        self.arm_width = random.randrange(int(length/4),length)
        self.angle = random.randrange(45)

        self.playing = True

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
        for border in borderPoints:
            triangles.append((border[0],border[1],animation.drawings.getApexPointTriangle(border[0],border[1],self.arm_length/3*2))) 

        return triangles

    def draw(self, borders: bool = False):
        self.color = Colors.patrick if self.playing else FishColors.orange

        pentagone = self.body()
        for triangle in pentagone:
            pygame.draw.polygon(self.window,self.color,triangle)
            
        arms = self.arms()
        for triangle in arms:
            pygame.draw.polygon(self.window,self.color,triangle)

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
        
        mouth = animation.drawings.getEllipseTriangles(cx,cy,width/15,length/18)
        for triangle in mouth:
            pygame.draw.polygon(self.window,Colors.black,triangle)
        

        
