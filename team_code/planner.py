import os
from collections import deque

import numpy as np


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=480):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

        self.debug = Plotter(debug_size)
        self.centre = None
        self.store_wps = []
        self.store_wps_real = []

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

        self.centre = self.route[int(len(self.route)/2)][0]

    def run_step(self, gps):
        self.debug.clear()

        self.cur_veh_gps = gps
        if self.centre is None:
            self.centre = gps

        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[i][1].value == 4)
            b = 255
            self.debug.dot(self.centre, self.route[i][0], (r, g, b))

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        self.debug.dot(self.centre, self.route[0][0], (0, 255, 0))
        self.debug.dot(self.centre, self.route[1][0], (255, 0, 0))

        self.store_wps_real.append(gps)
        for pos in self.store_wps_real:
            self.debug.dot(self.centre, pos, (0, 0, 255))
        return self.route[1]

    def run_step2(self, _poses, is_gps=True, color=(255, 255, 0), store=True):
        if is_gps:
            poses = _poses
        else:
            poses = _poses + self.cur_veh_gps if len(_poses) else []

        for pos in self.store_wps:
            self.debug.dot(self.centre, pos, color)

        if store:
            self.store_wps.extend(list(poses))

        for pos in poses:
            self.debug.dot(self.centre, pos, color)

    def show_route(self):
        self.debug.show()
