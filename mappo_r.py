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
import weakref
import random
import pickle
import cv2
import pygame
import torch.nn as nn
import torch.functional as F
import torch
from agents.navigation.global_route_planner import GlobalRoutePlanner
sys.path.append(os.path.join(os.path.dirname(__file__), 'dom_mappo'))

from agent import MAPPO

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_P = False
SEC_PER_EP = 20

class InitializeEnv:
    def __init__(self, num_agents):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.roads = self.map.get_topology()

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
        self.initial_positions = [carla.Location(x=149, y=-151, z=0.6), carla.Location(x=210, y=-158, z=0.6)]
        pygame.init()
        self.display = pygame.display.set_mode((self.im_width, self.im_height))
        pygame.display.set_caption("CARLA Camera")


        # optimal route
        sampling_resolution = 2

        grp = GlobalRoutePlanner(self.map, sampling_resolution)

        route_a_point_a = carla.Location(x=149, y=-151, z=0.6)
        route_a_point_b = carla.Location(x=220, y=-210, z=0.6)

        route_b_point_a = carla.Location(x=210, y=-158, z=0.6)
        route_b_point_b = carla.Location(x=160, y=-190, z=0.6)

        # has waypoints in the route!
        self.route_a = grp.trace_route(route_a_point_a, route_a_point_b)
        self.route_b = grp.trace_route(route_b_point_a, route_b_point_b)

        self.reached_goal = [False, False]  # Track if the goal is reached
        self.prev_velocity = [carla.Vector3D(0, 0, 0), carla.Vector3D(0, 0, 0)]

        for waypoint in self.route_a:
            self.world.debug.draw_string(waypoint[0].transform.location, "^", draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0, persistent_lines=True)
        
        for waypoint in self.route_b:
            self.world.debug.draw_string(waypoint[0].transform.location, "^", draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=0, persistent_lines=True)


    def reset(self):
        for vehicle in self.vehicles:
            vehicle.destroy()
        for sensor in self.sensors:
            sensor.destroy()
        self.destroy_vehicles()
        self.collision_hist = [[] for _ in range(self.num_agents)]
        self.line_crossings = [[] for _ in range(self.num_agents)]
        self.actor_list = []
        self.vehicles = []
        self.sensors = []
        self.images = [None] * self.num_agents

        

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        self.custom_spawn_points = [
            carla.Transform(carla.Location(x=149, y=-151, z=0), carla.Rotation(pitch=0, yaw=326, roll=0)),
            carla.Transform(carla.Location(x=210, y=-158, z=0), carla.Rotation(pitch=0, yaw=218, roll=0))
        ]


        for i in range(self.num_agents):
            #if i < len(spawn_points):
                #transform = spawn_points[i]
            #else:
                #print(f"Not enough spawn points available for agent {i}, using default location")
                #transform = carla.Transform(carla.Location(x=230, y=195, z=40))
            if i < len(self.custom_spawn_points):
                transform = self.custom_spawn_points[i]
            else:
                print(f"Not enough custom spawn points available for agent {i}, using default location")
                transform = carla.Transform(carla.Location(x=160, y=-175, z=10)) # fix this point later

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
            print(f"Applying control to vehicle {i}: throttle={action[0]}, steer={action[1]}, brake={action[2]}\n\n")
            self.vehicles[i].apply_control(carla.VehicleControl(throttle=action[0]/2, steer=action[1]/2, brake=action[2]))

        rewards = [self.compute_reward(i) for i in range(self.num_agents)]
        print(f"Rewards: {rewards}")  # Add debug statement for rewards
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
          #  image = np.transpose(self.images[i], (2, 0, 1))  # Convert image to correct shape

            state.append({
            
                'state': np.concatenate((
                    [transform.location.x, transform.location.y, transform.location.z],
                    [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
                    [velocity.x, velocity.y, velocity.z],
                    [acceleration.x, acceleration.y, acceleration.z]
                ))
            })
        return state
    """
    def compute_reward(self, agent_index):
        # Get the velocity of the vehicle
        velocity = self.vehicles[agent_index].get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        reward = 0
        print("VELOCITY", speed, reward)
        # Reward for moving
        reward += speed * 100  # Adjust the scaling factor if needed
        print("VEL REWARD", reward, "\n\n")
        # Penalize for collisions
        if self.collision_hist[agent_index]:
            reward -= 1
        
        # Penalize for line crossings
        if self.line_crossings[agent_index]:
            reward -= 0.5  # Adjust the reward for line crossings
        
        # Penalize for staying still
        if speed < 5.0:  # Consider a small threshold for staying still
            reward -= 10
        
        return reward
        """
    def calculate_distance(self, point1, point2):
        dx = point1.x - point2.x
        dy = point1.y - point2.y
        dz = point1.z - point2.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def find_closest_waypoint(self, vehicle_location, waypoints):
        closest_waypoint = min(waypoints, key=lambda wp: self.calculate_distance(vehicle_location, wp[0].transform.location))
        return closest_waypoint

    def normalize_rewards(rewards):
        return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

    def compute_reward(self, agent_index):
        # Get the velocity of the vehicle
        velocity = self.vehicles[agent_index].get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        reward = 0
        print("VELOCITY", speed, reward)
        
        # Reward for moving
        reward += speed * 0.5  # Adjust the scaling factor if needed
        print("VEL REWARD", reward, "\n\n")
        
        # Penalize for collisions
        if self.collision_hist[agent_index]:
            reward -= 50
        
        # Penalize for line crossings
        if self.line_crossings[agent_index]:
            reward -= 10 # Adjust the reward for line crossings
        
        # Penalize for staying still
        if speed < 5.0:  # Consider a small threshold for staying still
            reward -= 15
        
        # Distance to the closest waypoint on the optimal route
        vehicle_location = self.vehicles[agent_index].get_transform().location
        # Determine which route the agent is following
        route = self.route_a if agent_index == 0 else self.route_b
        closest_waypoint = self.find_closest_waypoint(vehicle_location, route)
        distance_to_route = self.calculate_distance(vehicle_location, closest_waypoint[0].transform.location)
        
        # Reward for staying close to the optimal route
        max_distance = 10.0  # Maximum distance for a meaningful reward
        reward += max(0.0, (max_distance - distance_to_route) / max_distance * 10)  # Adjust the scaling factor if needed
        
        # Reward for making progress through the intersection
        initial_position = self.initial_positions[agent_index]
        distance_travelled = self.calculate_distance(initial_position, vehicle_location)
        reward += distance_travelled * 0.1  # Adjust the scaling factor if needed
        
        # Significant reward for successfully navigating the intersection without collisions
        if self.reached_goal[agent_index]:
            reward += 100  # Adjust the reward for reaching the goal
        
        # Penalize for abrupt changes in speed or direction
        acceleration = np.linalg.norm([velocity.x - self.prev_velocity[agent_index].x, velocity.y - self.prev_velocity[agent_index].y, velocity.z - self.prev_velocity[agent_index].z])
        reward -= acceleration * 0.1  # Adjust the penalty for abrupt changes
        self.prev_velocity[agent_index] = velocity
        
        # Print debugging information for reward components
        print("Reward components:")
        print(f"  Speed reward: {speed * 0.5}")
        print(f"  Collision penalty: {-25 if self.collision_hist[agent_index] else 0}")
        print(f"  Line crossing penalty: {-5 if self.line_crossings[agent_index] else 0}")
        print(f"  Still penalty: {-10 if speed < 5.0 else 0}")
        print(f"  Route proximity reward: {max(0.0, (max_distance - distance_to_route) / max_distance * 10)}")
        print(f"  Distance travelled reward: {distance_travelled * 0.1}")
        print(f"  Goal reward: {100 if self.reached_goal[agent_index] else 0}")
        print(f"  Abrupt change penalty: {-acceleration * 0.1}")
        print(f"  Total reward: {reward}")

        return reward



    def check_done(self):
        if any(self.collision_hist) or (time.time() - self.episode_start) > SEC_PER_EP:
            print("WE ARE DONE\n\n")
            return [True] * self.num_agents
        return [False] * self.num_agents

    def render_camera(self, index):
        if self.images[index] is not None:
            image = self.images[index]
            image = np.fliplr(image)
            image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(image, (0, 0))
            pygame.display.flip()
            
    def destroy_vehicles(self):
        world = self.client.get_world()
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')

        for vehicle in vehicles:
            vehicle.destroy()

    def benchmark(self):
        pass



if __name__ == "__main__":
    torch.cuda.empty_cache()
    num_agents = 2
    env = InitializeEnv(num_agents)
    
    # Define the state dimensions including image and other state information
    state_dim = {
        'state': 12  # 3 for location, 3 for rotation, 3 for velocity, 3 for acceleration
    }
    
    action_dim = 6  # Assuming three actions: throttle, steer, brake
    agent_params = {
        'actor_hidden_size': 128,
        'critic_hidden_size': 128,
        'actor_output_act': nn.functional.log_softmax,
        'memory_capacity': 10000,
        'actor_lr': 0.0005,  # Reduced learning rate
        'critic_lr': 0.0005,  # Reduced learning rate
        'reward_gamma': 0.99,
        'clip_epsilon': 0.2,
        'num_agents': num_agents,
        'buffer_size': 10000,
        'max_grad_norm': 0.5  # Common default value
    }
    
    mappo = MAPPO(env, state_dim=state_dim, action_dim=action_dim, agent_params=agent_params)
    # Load existing model if available10000
    # if os.path.exists('model'):
    #     mappo.load('model')

    mappo.run(num_episodes=1000, batch_size=64)

    # Save the model after training
    mappo.save('model')
