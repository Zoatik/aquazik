from animation.fish import Fish
from pygame import draw, Surface
from constants import NOTE_NAMES
from animation.starfish import Starfish
import animation.drawings
from animation.drawings import Colors
from random import randrange
import math
import pygame
import random


class Aquarium:
    # -----Functions to create the fishes --------------------------------------------------------------
    def drawFishes(fishList):
        for i in range(0, len(fishList)):
            fishList[i].draw()

    def createFishList(window):
        # instance of fish --> change it so i don't do it manually
        # x + 150, y + 125
        fishCenterList = [
            (randrange(50, 350), randrange(50, 100)),
            (randrange(50, 350), randrange(200, 250)),
            (randrange(50, 350), randrange(350, 400)),
            (randrange(450, 750), randrange(50, 100)),
            (randrange(450, 750), randrange(200, 250)),
            (randrange(450, 750), randrange(350, 400)),
            (randrange(850, 1150), randrange(50, 100)),
            (randrange(850, 1150), randrange(200, 250)),
            (randrange(850, 1150), randrange(350, 400)),
            (randrange(1250, 1550), randrange(50, 100)),
            (randrange(1250, 1550), randrange(200, 250)),
            (randrange(1250, 1550), randrange(350, 400)),
        ]
        fishList: list[Fish] = []
        for ni in range(len(NOTE_NAMES)):
            fishList.append(
                Fish(
                    window,
                    NOTE_NAMES[ni],
                    (randrange(255), randrange(255), randrange(255)),
                    fishCenterList[ni],
                )
            )
        return fishList

    def createStarfishList(window):
        starFishList = []
        starfishCenterList = [
            (randrange(50, 350), randrange(500, 550)),
            (randrange(50, 350), randrange(650, 700)),
            (randrange(50, 350), randrange(800, 850)),
            (randrange(450, 750), randrange(500, 550)),
            (randrange(450, 750), randrange(650, 700)),
            (randrange(450, 750), randrange(800, 850)),
            (randrange(850, 1150), randrange(500, 550)),
            (randrange(850, 1150), randrange(650, 700)),
            (randrange(850, 1150), randrange(800, 850)),
            (randrange(1250, 1550), randrange(500, 550)),
            (randrange(1250, 1550), randrange(650, 700)),
            (randrange(1250, 1550), randrange(800, 850)),
        ]
        for ni in range(len(NOTE_NAMES)):
            starFishList.append(
                Starfish(window, NOTE_NAMES[ni], starfishCenterList[ni])
            )
        return starFishList

    def drawStarfish(starFishList):
        for i in range(0, len(starFishList)):
            starFishList[i].draw()

    def drawProgressBar(window: Surface, current_time, total_time):
        height = window.get_size()[1] - 25
        start = 15
        # end = window_width - 50px
        end = window.get_size()[0] - 50

        percentage = current_time / total_time

        current = start + (end - start) * percentage
        if current > end:
            current = end

        # ---------------- DRAW GARY
        radius = 20

        # draw body
        body = [
            (current, height + radius),
            (current + 50, height),
            (current + 50, height + radius),
        ]

        bodyColor = (0x87, 0xCD, 0xEA)
        draw.polygon(window, Colors.black, body, 2)
        draw.polygon(window, bodyColor, body)

        # draw coquille
        coquille = animation.drawings.getPolygonPoints(15, current, height, radius)

        for t in coquille:
            draw.polygon(window, (0xDC, 0x49, 0x60), t, 2)
            draw.polygon(window, (0xF4, 0x83, 0xAC), t)

        # draw eyelids
        eyelids = [
            [  # eyelid 1
                (current + 43, height + 5),
                (current + 38, height - 15),
                (current + 40, height - 15),
            ],
            [  # eyelid 1 filler
                (current + 42, height + 5),
                (current + 44, height + 5),
                (current + 39, height - 15),
            ],
            [  # eyelid 2
                (current + 47, height + 5),
                (current + 52, height - 15),
                (current + 50, height - 15),
            ],
            [  # eyelid 2 filler
                (current + 46, height + 5),
                (current + 48, height + 5),
                (current + 51, height - 15),
            ],
        ]

        for e in eyelids:
            draw.polygon(window, bodyColor, e)

        # eyes
        eyes = [
            animation.drawings.getOctogonPoints(current + 51, height - 15, 8),
            animation.drawings.getOctogonPoints(current + 39, height - 15, 8),
        ]
        for e in eyes:
            for t in e:
                draw.polygon(window, Colors.yellow, t)

        # iris
        iris = [
            animation.drawings.getOctogonPoints(current + 53, height - 14, 4),
            animation.drawings.getOctogonPoints(current + 41, height - 14, 4),
        ]
        for e in iris:
            for t in e:
                draw.polygon(window, (0xDA, 0x6E, 0x2C), t)

        pupils = [
            animation.drawings.getOctogonPoints(current + 54, height - 13, 2),
            animation.drawings.getOctogonPoints(current + 42, height - 13, 2),
        ]
        for e in pupils:
            for t in e:
                draw.polygon(window, Colors.black, t)

    def drawBackground(window: Surface):
        window.fill((125, 125, 255))

    # x, y = point central de la base de la plante
    def drawPlant(
        window: Surface, x: int, y: int, length: float = 15, angle: float = 0
    ):
        # base
        base_wall_ratio = 10
        # triangle qui a le haut
        draw.polygon(
            window,
            Colors.green,
            [
                (x, y),
                (x - length / base_wall_ratio, y - length),
                (x + length / base_wall_ratio, y - length),
            ],
        )
        draw.polygon(
            window,
            Colors.green,
            [
                (x - length / base_wall_ratio, y),
                (x - length / base_wall_ratio, y - length),
                (x + length / base_wall_ratio, y),
            ],
        )
        draw.polygon(
            window,
            Colors.green,
            [
                (x + length / base_wall_ratio, y),
                (x + length / base_wall_ratio, y - length),
                (x - length / base_wall_ratio, y),
            ],
        )

    def drawPatrickHouse(window:Surface):
        # Semi-circle parameters
        center_x, center_y = 80,850
        radius = 50
        num_triangles = 30  # more = smoother curve
        TRIANGLE_COLOR = (139, 69, 19)  # brown

        # Draw semicircle with triangles
        for i in range(num_triangles):
            angle1 = math.pi * i / num_triangles  # start angle
            angle2 = math.pi * (i + 1) / num_triangles  # end angle
            x1 = center_x + radius * math.cos(angle1)
            y1 = center_y - radius * math.sin(angle1)
            x2 = center_x + radius * math.cos(angle2)
            y2 = center_y - radius * math.sin(angle2)
            pygame.draw.polygon(window, TRIANGLE_COLOR, [(center_x, center_y), (x1, y1), (x2, y2)])
        #Draw the door
        pygame.draw.polygon(window,Colors.black,[(center_x - radius/10, center_y),(center_x - radius/10 + radius/5, center_y),(center_x + radius/10, center_y - radius/5)],)
        pygame.draw.polygon(window,Colors.black,[(center_x - radius/10, center_y),(center_x - radius/10, center_y - radius/5),(center_x + radius/10, center_y - radius/5)],)
        #Draw the pseudo antenna
        pygame.draw.polygon(window,Colors.white,[(center_x, center_y-radius),(center_x - radius/10, center_y-radius - radius/5),(center_x + radius/10, center_y -radius - radius/5)],)

    def drawSquidwardHouse(window:Surface, base_x: int = -1, base_y: int = -1, base_size: int = 50):
        # default parameters
        if base_x == -1:
            base_x = window.get_size()[0] / 2
        if base_y == -1:
            base_y = window.get_size()[1] - 5

        height_ratio = 2.2
        triangles = [
            # house center
            [
                (base_x - base_size / 2, base_y),
                (base_x - (2 * base_size) / 5, base_y - base_size * height_ratio),
                (base_x + base_size / 2, base_y)
            ],
            [
                (base_x + base_size / 2, base_y),
                (base_x + (2*base_size) / 5, base_y - base_size * height_ratio),
                (base_x - base_size / 2, base_y)
            ],
            [
                (base_x - (2 * base_size) / 5, base_y - base_size * height_ratio),
                (base_x + (2*base_size) / 5, base_y - base_size * height_ratio),
                (base_x, base_y)
            ],
            # left ear
            [
                (base_x - base_size/3, base_y - base_size * 1.5),
                (base_x - base_size / 1.5, base_y - base_size * 1.5),
                (base_x - base_size/3, base_y - base_size),
            ],
            [
                (base_x - base_size / 1.5, base_y - base_size * 1.5),
                (base_x - base_size / 1.5, base_y - base_size),
                (base_x - base_size/3, base_y - base_size),
            ],
            # right ear
            [
                (base_x + base_size/3, base_y - base_size * 1.5),
                (base_x + base_size / 1.5, base_y - base_size * 1.5),
                (base_x + base_size/3, base_y - base_size),
            ],
            [
                (base_x + base_size / 1.5, base_y - base_size * 1.5),
                (base_x + base_size / 1.5, base_y - base_size),
                (base_x + base_size/3, base_y - base_size),
            ],
        ]

        nose_eyebrow = [
            # eyebrow
            [
                (base_x - (2*base_size) / 5 + 2, base_y - (base_size * height_ratio)*(2/3)),
                (base_x + (2*base_size) / 5 - 2, base_y - (base_size * height_ratio)*(2/3) - 10),
                (base_x + (2*base_size) / 5 - 2, base_y - (base_size * height_ratio)*(2/3))
            ],
            [
                (base_x - (2*base_size) / 5 + 2, base_y - (base_size * height_ratio)*(2/3) - 10),
                (base_x + (2*base_size) / 5 - 2, base_y - (base_size * height_ratio)*(2/3) - 10),
                (base_x - (2*base_size) / 5 + 2, base_y - (base_size * height_ratio)*(2/3))
            ],
            # nose
            [
                (base_x - 3, base_y - (base_size * height_ratio)*(2/3)),
                (base_x - (base_size / 6), base_y - (base_size * height_ratio)/3),
                (base_x + (base_size / 6), base_y - (base_size * height_ratio)/3)
            ],
            [
                (base_x + 3, base_y - (base_size * height_ratio)*(2/3)),
                (base_x - (base_size / 6), base_y - (base_size * height_ratio)/3),
                (base_x + (base_size / 6), base_y - (base_size * height_ratio)/3)
            ],
            [
                (base_x - 3, base_y - (base_size * height_ratio)*(2/3)),
                (base_x + 3, base_y - (base_size * height_ratio)*(2/3)),
                (base_x, base_y - (base_size * height_ratio)/3)
            ]
        ]

        # eyes
        eye_pos_left = ((((base_x - 10) + base_x - (base_size / 6)) / 2) + (base_x - (2*base_size) / 5 + 2))/2
        eye_pos_right = base_x - (eye_pos_left - base_x)
        outer = [
            animation.drawings.getPolygonPoints(15, eye_pos_left, base_y - (base_size * height_ratio)*(2/3) + 10, 7),
            animation.drawings.getPolygonPoints(15, eye_pos_right, base_y - (base_size * height_ratio)*(2/3) + 10, 7)
        ]
        inner = [
            animation.drawings.getPolygonPoints(15, eye_pos_left, base_y - (base_size * height_ratio)*(2/3) + 10, 3.5),
            animation.drawings.getPolygonPoints(15, eye_pos_right, base_y - (base_size * height_ratio)*(2/3) + 10, 3.5)
        ]

        # door
        door = [
            [
                (base_x - (base_size / 6), base_y),
                (base_x - (base_size / 6), base_y - (base_size * height_ratio / 6)),
                (base_x + (base_size / 6), base_y - (base_size * height_ratio / 6)),
            ],
            [
                (base_x + (base_size / 6), base_y),
                (base_x - (base_size / 6), base_y),
                (base_x + (base_size / 6), base_y - (base_size * height_ratio / 6)),
            ]
        ]

        for t in triangles:
            pygame.draw.polygon(window, (0,0,150), t)

        for e in nose_eyebrow:
            pygame.draw.polygon(window, (0,0, 255), e)

        for polygon in outer:
            for e in polygon:
                pygame.draw.polygon(window, (125, 125, 255), e)
        
        for polygon in inner:
            for e in polygon:
                pygame.draw.polygon(window, (150, 150, 255), e)

        brown_color = (139, 69, 19)

        for t in door:
            pygame.draw.polygon(window, brown_color, t) # brown
        # Draw semicircle with triangles

        triangle_number = 12
        door_radius = (base_size / 6)

        for i in range(triangle_number):
            angle1 = math.pi * i / triangle_number  # start angle
            angle2 = math.pi * (i + 1) / triangle_number  # end angle
            x1 = base_x + door_radius * math.cos(angle1)
            y1 = base_y - (base_size * height_ratio / 6) - door_radius * math.sin(angle1)
            x2 = base_x + door_radius * math.cos(angle2)
            y2 = base_y - (base_size * height_ratio / 6) - door_radius * math.sin(angle2)
            pygame.draw.polygon(window, brown_color, [(base_x, base_y - (base_size * height_ratio / 6)), (x1, y1), (x2, y2)])
                    


    def lerp(a, b, t):
        return a + (b - a) * t

    def color_lerp(c1, c2, t):
        return (
            int(Aquarium.lerp(c1[0], c2[0], t)),
            int(Aquarium.lerp(c1[1], c2[1], t)),
            int(Aquarium.lerp(c1[2], c2[2], t)),
            )


    # --- CROWN (LEAVES) ----------------------------------------------------------
    def drawBobHouse(surface):
        center = 1520, 820
        rx = 30
        ry = 50
        segments = 40
        #color = (255, 140, 0)
        color=(240, 100, 0)
        cx, cy = center
        points = []
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = cx + rx * math.cos(angle)
            y = cy + ry * math.sin(angle)
            points.append((x, y))

        # Draw triangle fan
        for i in range(segments):
            triangle = [ (cx, cy), points[i], points[i+1] ]
            pygame.draw.polygon(surface, color, triangle)
            pygame.draw.polygon(surface, (0xAB, 0x21, 0x00), triangle, 2)

    def drawBobTopHouse(surface):
        #surface, base_center, base_width, height,layers=3, spikes=9, tilt=0.15,jitter=0.12, seed=None

        base_center = (1520, 780)
        base_width = 50
        height = 70
        layers = 10 
        spikes = 9
        tilt = 0.18
        jitter = 0.10
        seed = 20

        LEAF_MAIN = (27, 142, 73)
        LEAF_DARK = (18, 102, 53)
        LEAF_LIGHT = (46, 181, 101)
        SAND = (232, 210, 160)
        """
Draw a pineapple crown using ONLY triangles.


Parameters
-----------
base_center : (x, y) tuple for where the crown sits on the pineapple top
base_width : width of the lowest leaf layer
height : total crown height from base to top tips
layers : number of stacked leaf rows
spikes : number of leaf spikes on the widest, lowest layer
tilt : horizontal lean of leaf tips (positive leans right)
jitter : randomness factor (0..~0.3) to vary tips & bases
seed : optional random seed for reproducible shapes
"""

    

        if seed is not None:
            rnd = random.Random(seed)
        else:
            rnd = random


        cx, cy = base_center


        # For each layer, shrink width and raise base line
        for layer in range(layers):
            t = layer / max(1, layers - 1) # 0..1
            # Narrower and higher for upper layers
            w = Aquarium.lerp(base_width, base_width * 0.45, t)
            base_y = cy - Aquarium.lerp(0, height * 0.65, t)
            # Taller spikes on lower layers, a bit shorter above
            h_layer = Aquarium.lerp(height * 0.55, height * 0.30, t)


            # Increase spike count slightly on mid layers for fullness
            spk = int(round(Aquarium.lerp(spikes, spikes + 2, t)))


            # Base points along the width (spk segments => spk triangles)
            left_x = cx - w / 2
            right_x = cx + w / 2
            # Generate base divisions
            xs = [Aquarium.lerp(left_x, right_x, i / spk) for i in range(spk + 1)]


            for i in range(spk):
                # Each leaf spike uses a triangle from base segment (xi, xi+1) to a tip
                x0 = xs[i]
                x1 = xs[i + 1]
                mid = (x0 + x1) / 2

                seg = x1 - x0
                # Random jitter on base and height for organic feel (triangles only!)
                jitter_x0 = seg * (rnd.uniform(-jitter, jitter))
                jitter_x1 = seg * (rnd.uniform(-jitter, jitter))
                jitter_mid = seg * (rnd.uniform(-jitter, jitter))
                jitter_h = h_layer * (rnd.uniform(-jitter, jitter))

                a = (x0 + jitter_x0, base_y) # left base
                b = (x1 + jitter_x1, base_y) # right base


                # Tip position: lean to the side a bit via tilt, alternate direction
                lean_dir = -1 if (i + layer) % 2 == 0 else 1
                tip_x = mid + jitter_mid + lean_dir * tilt * h_layer
                tip_y = base_y - (h_layer + jitter_h)
                c_pt = (tip_x, tip_y)

                # Choose a subtle color variation per spike
                shade_t = 0.35 * rnd.random()
                col = Aquarium.color_lerp(LEAF_MAIN, LEAF_LIGHT, shade_t)


                # Main leaf triangle
                pygame.draw.polygon(surface, col, (a, b, c_pt))


                # Add a darker inner ridge (thin triangle) for depth, still triangles only
                ridge_mid = Aquarium.lerp(mid, tip_x, 0.55)
                ridge_tip = (ridge_mid, Aquarium.lerp(base_y, tip_y, 0.75))
                ridge_left = (Aquarium.lerp(a[0], b[0], 0.48), Aquarium.lerp(a[1], b[1], 0.48))
                pygame.draw.polygon(surface, Aquarium.color_lerp(LEAF_MAIN, LEAF_DARK, 0.35), (ridge_left, c_pt, ridge_tip))

            # Between layers, stitch small back-facing fillers to avoid gaps (triangles)
            if layer < layers - 1:
                next_w = Aquarium.lerp(base_width, base_width * 0.45, (layer + 1) / max(1, layers - 1))
                next_base_y = cy - Aquarium.lerp(0, height * 0.65, (layer + 1) / max(1, layers - 1))
                # Simple fan triangles from current base to the start of next layer
                segs = max(5, int(spikes * 0.6))
                for j in range(segs):
                    t0 = j / segs
                    t1 = (j + 1) / segs
                    x0 = Aquarium.lerp(cx - w / 2, cx + w / 2, t0)
                    x1 = Aquarium.lerp(cx - w / 2, cx + w / 2, t1)
                    nx = Aquarium.lerp(cx - next_w / 2, cx + next_w / 2, (t0 + t1) / 2)
                    # Back filler triangle
                    pygame.draw.polygon(surface, Aquarium.color_lerp(LEAF_MAIN, LEAF_DARK, 0.15),( (x0, base_y), (x1, base_y), (nx, next_base_y)))

        pygame.draw.polygon(surface, (125, 125, 255), [(cx - base_width / 2, cy + 95), (cx + base_width / 2, cy + 95), (cx + base_width / 2, cy + 80)])
        pygame.draw.polygon(surface, (125, 125, 255), [(cx - base_width / 2, cy + 95), (cx - base_width / 2, cy + 80), (cx + base_width / 2, cy + 80)])

