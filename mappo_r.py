import glob
import os
import sys
import numpy as np
import time
from time import sleep

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
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

        pygame.init()
        self.display = pygame.display.set_mode((self.im_width, self.im_height))
        pygame.display.set_caption("CARLA Camera")

    def reset(self):
        for vehicle in self.vehicles:
            vehicle.destroy()
        for sensor in self.sensors:
            sensor.destroy()

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
        return np.array([np.transpose(img, (2, 0, 1)) for img in self.images])

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
                action = [0.0, 0.0, 1.0]
            print(f"Applying control to vehicle {i}: throttle={action[0]}, steer={action[1]}, brake={action[2]}")
            self.vehicles[i].apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2]))

        rewards = [self.compute_reward(i) for i in range(self.num_agents)]
        self.render_camera(0)
        done = self.check_done()
        next_states = np.array([np.transpose(img, (2, 0, 1)) for img in self.images])

        # Ensure the shape of next_states is as expected
        next_states = next_states.reshape(self.num_agents, 3, 480, 640)

        # Ensure done is a list with the correct length
        done = np.array(done).reshape(self.num_agents)

        return next_states, rewards, done



    def compute_reward(self, agent_index):
        if self.collision_hist[agent_index]:
            return -1
        return 1

    def check_done(self):
        if any(self.collision_hist) or (time.time() - self.episode_start) > SEC_PER_EP:
            return [True] * self.num_agents
        return [False] * self.num_agents

    def render_camera(self, index):
        if self.images[index] is not None:
            image = self.images[index]
            image = np.fliplr(image)
            image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(image, (0, 0))
            pygame.display.flip()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    num_agents = 2
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
        'clip_epsilon': 0.2,
        'num_agents': num_agents,
        
    }
    mappo = MAPPO(env, state_dim=state_dim, action_dim=action_dim, agent_params=agent_params)
    mappo.run(num_episodes=1000, batch_size=16)
