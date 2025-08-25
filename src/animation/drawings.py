import math
from random import random


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
        p2 = points[(i + 1) % len(points)]
        ret.append(((cx, cy), p1, p2))

    return ret

def getEllipseTriangles(cx, cy, rx, ry, segments=40):
    points = []
    triangle = []
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        points.append((x, y))
    for i in range(segments):
        triangle.append([ (cx, cy), points[i], points[i+1] ])
    
    return triangle
    