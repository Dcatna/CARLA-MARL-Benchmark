HELLO to whomever👋 This was the project the Dominic Catena and Haejin Song worked on throughout the summer of 2024 for WINLAB research internship. The purpose of this project is to create a multiagent reinforcement learning enviroment 
in CARLA (Driving simulator) for testing MARL driving algorithms.

**--SET UP ENVIROMENT**
We are using the latest version of CARLA 0.9.15 (non-stable version so we can get the most out of the OSM renderer) 
Inorder to run the code make sure you have the dependencies installed (Pytorch, CARLA api, matplotlib, and numpy) **CONDA RECOMMENDED
Run the make launch command at the root of CARLA to start the server, then go through the custom maps folder and double-click the nb_intersection map to load it

**--INSTALL AND RUN **
Pull the code from the simple branch (the main branch of this repo, other branches are for testing) 
activate conda eviroment (conda activate <YOUR_ENV_NAME>) and run the mappo_r file (this will create the enviroment (cars and sensors) and the run the MARL code in this case MAPPO)

**--ADDITIONAL INFORMATION**
We decided to try to implement the MAPPO (multi agent proximal policy optimization) algorithm due to its high preformance in other RL and MARL tasks
The code for the MARL is found in the agent file which contains the actor and critic networks, agent class and the actual training loop for the agents
The agents are shown to not learn too well (This is our doing :/) and can definetly be optimized (add frame buffering so the agents dont update their policies and make so many decisions, change hyperparameters, etc...)

The benchmark function wasnt finished, all that really has to be done is just looping through and loading models and running them through the test cases one per model
**graphs are displayed at the end of training, the critic loss one might be a bit messed up


**--VR Functionalites (PREFORMANCE ISSUES)**
Inorder to get CARLA in VR mode on Windows (We were using the Meta Quest 2, so might not be the same for all VR headsets) you need to install the Oculus software and SteamVR to connect the VR to the computer, then run make launch in the
root folder of CARLA and navigate to the plugins option at the top right and install the SteamVR plugin and run in VR mode

Now to get it on Linux its almsot the same exepect the fact the VR doesnt support Linux. So to get Meta Quest 2 to recognize the Linux we use ALVR (nightly) and SideQuest. Go to alvr nightly page and download and unzip the .tar package 
and run the launch command (I think its ./alvr_dashboard). Now download the .apk file for the version of ALVR you downloaded and go to SideQuest connect your headset (usb-c) and import the .apk file. Now you should see ALVR in your apps
on the Meta Quest 2. Start ALVR on both systems and they should sync up, then the rest of the steps should be the same as the Windows instructions.

