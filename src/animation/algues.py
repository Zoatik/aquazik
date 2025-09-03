import math, random
import pygame
from constants import Colors

def lerp(a, b, t): return a + (b - a) * t
def normalize(vx, vy):
    l = math.hypot(vx, vy)
    if l == 0: return 0.0, 0.0
    return vx / l, vy / l

class Seaweed:
    def __init__(self, x, y, height=20, base_width=5, sway_speed=1.0, sway_amp_deg=20):
        self.base = pygame.Vector2(x, y)
        self.height = height
        self.base_width = base_width
        self.segments = 20
        self.seg_len = height / self.segments
        self.sway_amp = math.radians(sway_amp_deg)
        self.sway_speed = sway_speed
        self.color = Colors.algues_green
        self.phase = random.random() * math.tau  # random phase for variety
        self.tip_width = base_width * 0.2

    def _centerline(self, t):
        points = [self.base.xy]
        x, y = self.base.xy
        base_angle = -math.pi/2
        for i in range(1, self.segments+1):
            u = i / self.segments
            bend = math.sin(t*self.sway_speed + self.phase) * self.sway_amp * (u**1.3)
            dx = math.cos(base_angle + bend) * self.seg_len
            dy = math.sin(base_angle + bend) * self.seg_len
            x += dx
            y += dy
            points.append((x, y))
        return points

    def draw(self, surf, t):
        points = self._centerline(t)
        n = len(points)
        for i in range(n-1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            # width tapering
            w0 = lerp(self.base_width, self.tip_width, i/(n-1)) / 2
            w1 = lerp(self.base_width, self.tip_width, (i+1)/(n-1)) / 2
            # perpendicular vector
            dx, dy = x1-x0, y1-y0
            nx, ny = normalize(-dy, dx)
            left = [ (x0+nx*w0, y0+ny*w0), (x1+nx*w1, y1+ny*w1) ]
            right = [ (x0-nx*w0, y0-ny*w0), (x1-nx*w1, y1-ny*w1) ]
            pygame.draw.polygon(surf, self.color, [left[0], right[0], left[1]])
            pygame.draw.polygon(surf, self.color, [left[1], right[0], right[1]])
            pygame.draw.polygon(surf, Colors.black, [left[0], right[0], left[1]],1)
            pygame.draw.polygon(surf, Colors.black, [left[1], right[0], right[1]],1)