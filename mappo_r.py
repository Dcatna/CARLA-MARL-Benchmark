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
import weakref
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
        self.line_crossings = [[] for _ in range(num_agents)]  # Track line crossings

        pygame.init()
        self.display = pygame.display.set_mode((self.im_width, self.im_height))
        pygame.display.set_caption("CARLA Camera")

    def reset(self):
        for vehicle in self.vehicles:
            vehicle.destroy()
        for sensor in self.sensors:
            sensor.destroy()

        self.collision_hist = [[] for _ in range(self.num_agents)]
        self.line_crossings = [[] for _ in range(self.num_agents)]
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
            self.sensors.append(col_sensor)
            col_sensor.listen(lambda event, i=i: self.collision_data(event, i))

            lane_invasion_sensor_bp = self.bp_library.find('sensor.other.lane_invasion')
            lane_invasion_sensor = self.world.spawn_actor(lane_invasion_sensor_bp, carla.Transform(), attach_to=vehicle)
            self.sensors.append(lane_invasion_sensor)
            lane_invasion_sensor.listen(lambda event, i=i: self.on_invasion(event, i))

        while any(img is None for img in self.images):
            sleep(.01)

        self.episode_start = time.time()
        return self.get_state()
    
    def on_invasion(self, event, car_id):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        print(f"Vehicle {car_id} crossed line: {', '.join(text)}\n\n\n\n")
        self.line_crossings[car_id].append(text)

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
        next_states = self.get_state()

        return next_states, rewards, done

    def get_state(self):
        state = []
        for i in range(self.num_agents):
            vehicle = self.vehicles[i]
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            acceleration = vehicle.get_acceleration()
            image = np.transpose(self.images[i], (2, 0, 1))  # Convert image to correct shape

            state.append({
                'image': image,
                'state': np.concatenate((
                    [transform.location.x, transform.location.y, transform.location.z],
                    [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
                    [velocity.x, velocity.y, velocity.z],
                    [acceleration.x, acceleration.y, acceleration.z]
                ))
            })
        return state

    def compute_reward(self, agent_index):
        # Get the velocity of the vehicle
        velocity = self.vehicles[agent_index].get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        
        reward = 0
        
        # Reward for moving
        reward += speed * 0.1
        
        # Penalize for collisions
        if self.collision_hist[agent_index]:
            reward -= 1
        
        # Penalize for line crossings
        if self.line_crossings[agent_index]:
            reward -= 0.5  # Adjust the reward for line crossings
        
        return reward

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
    
    # Define the state dimensions including image and other state information
    state_dim = {
        'image': (3, 480, 640),
        'state': 12  # 3 for location, 3 for rotation, 3 for velocity, 3 for acceleration
    }
    
    action_dim = 3  # Assuming three actions: throttle, steer, brake
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


