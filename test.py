import glob
import os
import sys
import numpy as np
import time
from time import sleep

try:
    sys.path.append(glob.glob('../../../carla/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append("../../../carla/carla/PythonAPI")
except IndexError:
    pass

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
map = world.get_map()
roads = map.get_topology()

sampling_resolution = 2

grp = GlobalRoutePlanner(map, sampling_resolution)

route_a_point_a = carla.Location(x=149, y=-151, z=0.6)
route_a_point_b = carla.Location(x=220, y=-210, z=0.6)

route_b_point_a = carla.Location(x=210, y=-158, z=0.6)
route_b_point_b = carla.Location(x=160, y=-190, z=0.6)

# has waypoints in the route!
route_a = grp.trace_route(route_a_point_a, route_a_point_b)
route_b = grp.trace_route(route_b_point_a, route_b_point_b)

for waypoint in route_a:
    world.debug.draw_string(waypoint[0].transform.location, "^", draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120, persistent_lines=True)

for waypoint in route_b:
    world.debug.draw_string(waypoint[0].transform.location, "^", draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=120, persistent_lines=True)
