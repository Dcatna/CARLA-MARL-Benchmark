import glob
import os
import sys
import numpy as np
import time
from time import sleep
import yaml

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
    def __init__(self, config):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.bp_library = self.world.get_blueprint_library()
        self.model3 = self.bp_library.filter("model3")[0]
        self.im_width = config['environment']['im_width']
        self.im_height = config['environment']['im_height']
        self.num_agents = config['agent']['num_agents']
        self.vehicles = []
        self.sensors = []
        self.images = [None] * self.num_agents
        self.collision_hist = [[] for _ in range(self.num_agents)]
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        pygame.init()
        self.display = pygame.display.set_mode((self.im_width, self.im_height))
        pygame.display.set_caption("CARLA Camera")

        self.spawn_points = self.map.get_spawn_points()
        self.spawn_point = random.choice(self.spawn_points) if self.spawn_points else None

        # Define custom spawn points
        self.custom_spawn_points = [
            carla.Transform(carla.Location(x=148, y=-151, z=10), carla.Rotation(pitch=0, yaw=180, roll=0)),
            carla.Transform(carla.Location(x=210, y=-160, z=10), carla.Rotation(pitch=0, yaw=180, roll=0))
        ]

    def reset(self):
        self._destroy_actors()
        self._spawn_vehicles()
        while any(img is None for img in self.images):
            sleep(0.01)
        self.episode_start = time.time()
        return np.array([np.transpose(img, (2, 0, 1)) for img in self.images])

    def _destroy_actors(self):
        for actor in self.vehicles + self.sensors:
            actor.destroy()
        self.vehicles = []
        self.sensors = []
        self.images = [None] * self.num_agents
        self.collision_hist = [[] for _ in range(self.num_agents)]
    
    def _spawn_vehicles(self):
        if self.spawn_point == None:
            for i in range(self.num_agents):
                if i < len(self.custom_spawn_points):
                    transform = self.custom_spawn_points[i]
                else:
                    print(f"Not enough custom spawn points available for agent {i}, using default location")
                    transform = carla.Transform(carla.Location(x=160, y=-175, z=10))

                vehicle = self.world.spawn_actor(self.model3, transform)
                self.vehicles.append(vehicle)
                self._attach_sensors(vehicle, i)
        else:
            for i in range(self.num_agents):
                transform = random.choice(self.spawn_points)
                vehicle = self.world.spawn_actor(self.model3, transform)
                self.vehicles.append(vehicle)
                self._attach_sensors(vehicle, i)
    
    def _attach_sensors(self, vehicle, index):
        rgb_cam = self.bp_library.find("sensor.camera.rgb")
        rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        rgb_cam.set_attribute("fov", "110")
        transform = carla.Transform(carla.Location(x=2.5, z=.7))
        sensor = self.world.spawn_actor(rgb_cam, transform, attach_to=vehicle)
        self.sensors.append(sensor)
        sensor.listen(lambda data: self.process_image(data, index))

        col_sensor = self.bp_library.find("sensor.other.collision")
        col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=vehicle)
        self.sensors.append(col_sensor)
        col_sensor.listen(lambda event: self.collision_data(event, index))

    def process_image(self, image, index):
        i = np.array(image.raw_data).reshape((self.im_height, self.im_width, 4))[:, :, :3]
        self.images[index] = i

    def collision_data(self, event, index):
        self.collision_hist[index].append(event)

    def step(self, actions):
        for i, action in enumerate(actions):
            self.vehicles[i].apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2]))
        rewards = [self.compute_reward(i) for i in range(self.num_agents)]
        self.render_camera(0)
        done = self.check_done()
        next_states = np.array([np.transpose(img, (2, 0, 1)) for img in self.images])
        next_states = next_states.reshape(self.num_agents, 3, self.im_height, self.im_width)
        done = np.array(done).reshape(self.num_agents)
        return next_states, rewards, done

    def compute_reward(self, agent_index):
        return -1 if self.collision_hist[agent_index] else 1

    def check_done(self):
        return [True] * self.num_agents if any(self.collision_hist) or (time.time() - self.episode_start) > 10 else [False] * self.num_agents

    def render_camera(self, index):
        if self.images[index] is not None:
            image = np.fliplr(self.images[index])
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            pygame.display.flip()

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    torch.cuda.empty_cache()
    env = InitializeEnv(config)
    state_dim = (3, config['environment']['im_height'], config['environment']['im_width'])
    action_dim = 3
    agent_params = config['agent']

    mappo = MAPPO(env, state_dim=state_dim, action_dim=action_dim, agent_params=agent_params)
    mappo.run(num_episodes=agent_params['num_episodes'], batch_size=agent_params['batch_size'])
