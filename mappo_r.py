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
import torch.nn as nn
import torch.functional as F
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), 'dom_mappo'))

from agent import MAPPO

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_P = False
SEC_PER_EP = 10

###############################################
#
#     WORLD CLASS
#
###############################################


class InitializeEnv:
    def __init__(self, num_agents):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.bp_library = self.world.get_blueprint_library()
        self.model3 = self.bp_library.filter("model3")[0]
        self.im_width = IM_WIDTH
        self.im_height = IM_HEIGHT
        self.num_agents = num_agents
        self.vehicles = []
        self.sensors = []
        self.images = [None] * num_agents
        self.collision_hist = [[] for _ in range(num_agents)]

    def reset(self):
        self.collision_hist = [[] for _ in range(self.num_agents)]
        self.actor_list = []
        self.vehicles = []
        self.sensors = []
        self.images = [None] * self.num_agents

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for i in range(self.num_agents):
            if i < len(spawn_points):
                transform = spawn_points[i]
            else:
                print(f"Not enough spawn points available for agent {i}, using default location")
                transform = carla.Transform(carla.Location(x=230, y=195, z=40))

            vehicle = self.world.spawn_actor(self.model3, transform)
            self.vehicles.append(vehicle)
            self.actor_list.append(vehicle)

            rgb_cam = self.bp_library.find("sensor.camera.rgb")
            rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
            rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
            rgb_cam.set_attribute("fov", "110")

            transform = carla.Transform(carla.Location(x=2.5, z=.7))
            sensor = self.world.spawn_actor(rgb_cam, transform, attach_to=vehicle)
            self.sensors.append(sensor)
            self.actor_list.append(sensor)
            sensor.listen(lambda data, i=i: self.process_image(data, i))

            col_sensor = self.bp_library.find("sensor.other.collision")
            col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=vehicle)
            self.actor_list.append(col_sensor)
            col_sensor.listen(lambda event, i=i: self.collision_data(event, i))

        while any(img is None for img in self.images):
            sleep(.01)

        self.episode_start = time.time()
        for vehicle in self.vehicles:
            vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))

        return [np.transpose(img, (2, 0, 1)) for img in self.images]  # Transpose images to [channels, height, width]

    def process_image(self, image, index):
        i = np.array(image.raw_data)
        i2 = i.reshape((image.height, image.width, 4))
        i3 = i2[:, :, :3]
        self.images[index] = i3

    def collision_data(self, event, index):
        self.collision_hist[index].append(event)

    def step(self, actions):
        for i, action in enumerate(actions):
            if len(action) != 3:
                action = [0.0, 0.0, 1.0]  # Default action (full brake) if the action is invalid
            self.vehicles[i].apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2]))

        rewards = [self.compute_reward(i) for i in range(self.num_agents)]
        done = self.check_done()
        next_states = [np.transpose(img, (2, 0, 1)) for img in self.images]  # Transpose images to [channels, height, width]
        return next_states, rewards, done

    def compute_reward(self, agent_index):
        if self.collision_hist[agent_index]:
            return -1  # Penalize collisions
        return 1  # Reward for staying on track

    def check_done(self):
        if any(self.collision_hist) or (time.time() - self.episode_start) > SEC_PER_EP:
            return True
        return False




    
##################################################
#
#       Networks Actor and Critic
#
##################################################

##################################################
#
#     Game LOOOP
#
##################################################

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

# if __name__ == "__main__":
#     game_loop()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    num_agents = 2  # Example for 2 agents
    env = InitializeEnv(num_agents)
    state_dim = (3, 480, 640)
    action_dim = 3
    agent_params = {
        'actor_hidden_size': 128,
        'critic_hidden_size': 128,
        'actor_output_act': nn.functional.log_softmax,
        'memory_capacity': 10000,
        'actor_lr': 0.0001,
        'critic_lr': 0.0001,
        'reward_gamma': 0.99,
        'clip_param': 0.2,
        'use_cuda': True,
        'num_agents': num_agents  # Pass number of agents here
    }
    mappo = MAPPO(env, state_dim=state_dim, action_dim=action_dim, agent_params=agent_params)
    mappo.run(num_episodes=1000, batch_size=16)

