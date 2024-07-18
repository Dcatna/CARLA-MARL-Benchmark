import glob
import os
import sys
import numpy as np
import time
from time import sleep
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import pickle
import cv2
import pygame

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_P = False
SEC_PER_EP = 10

class InitializeEnv():
    front_cam = None
    SHOW_CAM = SHOW_P

    def __init__(self):

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.bp_library = self.world.get_blueprint_library()
        self.model3 = self.bp_library.filter("model3")[0]
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        
    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.bp_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=2.5,  z=.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data : self.process_image(data))

        #vehicle control
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))


        col_sensor = self.bp_library.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collision_data(event))


        while self.front_cam is None:
            sleep(.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, brake = 0.0))

        return self.front_cam
    
    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        #action = [throttle, steer, brake]
        self.vehicle.apply_control(carla.VehicleControl(throttle = action[0], steer=action[1], brake=action[2]))

        
        pass

    def process_image(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((image.height, image.width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_cam = i3
        if self.SHOW_CAM:
            surface = pygame.surfarray.make_surface(i3.swapaxes(0, 1))
            return surface
        return None


def game_loop():
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

    env = InitializeEnv()
    front_cam = env.reset()

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        if env.front_cam is not None:
            frame = pygame.surfarray.make_surface(env.front_cam.swapaxes(0, 1))
            display.blit(frame, (0, 0))
            pygame.display.flip()

        clock.tick_busy_loop(60)

    pygame.quit()

if __name__ == "__main__":
    game_loop()