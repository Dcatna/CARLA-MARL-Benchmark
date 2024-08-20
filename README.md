# **WINLAB MARL Benchmark for Autonomous Driving Testing with MAPPO Implementation**

**HELLO to whomever** 👋 This project was worked on by Dominic Catena and Haejin Song throughout the summer of 2024 during the WINLAB research internship. The purpose of this project is to create a multi-agent reinforcement learning environment in CARLA (Driving simulator) for testing MARL driving algorithms in real world scenarios.

## **Set Up Environment**
We are using the latest version of CARLA 0.9.15 (non-stable version to get the most out of the OSM renderer). To run the code, make sure you have the following dependencies installed: **PyTorch, CARLA API, matplotlib, and numpy**. (Conda is recommended).

1. Run the `make launch` command at the root of CARLA to start the server.
2. Navigate to the custom maps folder and double-click the `nb_intersection` map to load it.
3. You can download the nb_intersection from the tes branch

## **Install and Run**
1. Pull the code from the `simple` branch (the main branch of this repo, other branches are for testing).
2. Activate the conda environment with `conda activate <YOUR_ENV_NAME>`.
3. Run the `mappo_r` file, which will create the environment (cars and sensors) and then run the MARL code using MAPPO.

## **VR Functionalities (Performance Issues)**
To get CARLA in VR mode on Windows (using the Meta Quest 2, though it may differ for other VR headsets), you need to:
1. Install the Oculus software and SteamVR.
2. Run `make launch` in the root folder of CARLA.
3. Navigate to the plugins option at the top right, install the SteamVR plugin, and run in VR mode.

For Linux VR is not directly supported. To get the Meta Quest 2 to recognize Linux:
1. Use ALVR (nightly) and SideQuest.
2. Download and unzip the `.tar` package from the ALVR nightly page, then run the launch command (`./alvr_dashboard`).
3. Download the `.apk` file for the version of ALVR you downloaded, then use SideQuest to connect your headset (via USB-C) and import the `.apk` file.
4. You should now see ALVR in your apps on the Meta Quest 2. Start ALVR on both systems, sync them up, and then follow the same steps as the Windows instructions.

**More Information here [VR documentation from Piti98](https://github.com/carla-simulator/carla/files/9333229/Adding.VR.to.CARLA.pdf)**

## **MAP GENERATION**
1. Go to https://www.openstreetmap.org/#map=15/40.15630/-75.01130, select and cutout of area that you want to generate in CARLA (CANT BE MORE THAN 1gb data so make relatively small maps), then copy the link from the Overpass API
2. Navigate to `CarlaToolsContent/OnroadMapGenerator`. Right click on `UW_OnRoadMainWidget` and select Launch Editor Utility Widget
3. Paste the copied link into the OSM URL and click generate
4. Go to the custom maps folder and double-click the desired map to load it, then just click play! 
5. For more detailed instructions go to https://carla.readthedocs.io/en/latest/adv_digital_twin/

## **Additional Information**
Note: The agents do not learn too well (our doing :/), and there is room for optimization (e.g., add frame buffering so the agents don’t update their policies and make so many decisions, change hyperparameters, etc.).
We implemented the MAPPO (Multi-Agent Proximal Policy Optimization) algorithm due to its high performance in other RL and MARL tasks. The code for MARL is found in the `agent` file, which contains the actor and critic networks, the agent class, and the actual training loop for the agents.

The mappo_r file contains the code to set up the enviroment meaning the cars, sensors, routes, reward functions and actually loading the model/s

The agnet.py files contains the code for the MAPPO implementation **(we werent able to run the code for more than 1000 episodes so we dont know the full capabilites of the model)**. Originally we were using convolutional layers in 
the Actor and Critic networks, but felt that it was too complicated for the network, so we got rid of it and tried to make it more simple (hence the branch name). The agents currently learn through epsilon decay exploration method and also clip gradients to reduce significant policy changes. We arent too sure where the problem in the model lies but im sure with a bit more testing it can be identified.

The replaybuffer.py file is just the replay buffer... lol

The benchmark function wasn’t finished. To complete it, loop through and load models, then run them through the test cases one per model.

*Graphs are displayed at the end of training; the critic loss graph might be a bit off.*

