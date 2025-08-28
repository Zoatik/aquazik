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

def pivotTriangles(centerPoint: tuple[float, float], triangles: list[list[tuple[float, float]]], angleDeg: float):
    return [pivotTriangle(centerPoint, t, angleDeg) for t in triangles]

def pivotTriangle(centerPoint: tuple[float, float], triangle: list[tuple[float, float]], angleDeg: float):
        pivoted_triangle = []
        for point in triangle:
            angle_rad = math.radians(angleDeg)
            dx, dy = point[0] - centerPoint[0], point[1] - centerPoint[1]
            x2 = centerPoint[0] + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y2 = centerPoint[1] + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            pivoted_triangle.append((x2, y2))
        return pivoted_triangle

def twoPointDistance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt(math.pow(p2[0] - p1[0],2) + math.pow(p2[1] - p1[1],2))