import math
from random import random

class Colors:
    bgColor = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    black = (0, 0, 0)
    yellow = (231, 199, 25)
    white = (255, 255, 255)
    orange = (255, 150, 50)
    
    def get_random_color():
        # Récupère tous les attributs de la classe sauf spéciaux (__xxx__)
        values = [v for k, v in Colors.__dict__.items() if not k.startswith("__") and not k.startswith("bgColor") and not k.startswith("get")]
        return values[int(random()*len(values))]

def getOctogonPoints(cx, cy, radius):
    return getPolygonPoints(8, cx, cy, radius)

def getPolygonPoints(sides: int, cx, cy, radius):
    points = []
    ret = []
    # recherche des sommets du triangle à l'aide des angles : 45° : octogone
    for i in range(sides):  # 8 sommets
        angle = math.radians((360 / sides) * i)  # 45° * i
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))

    # ajout des points à la liste triangles
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1) % len(points)]
        ret.append(((cx, cy), p1, p2))

    return ret