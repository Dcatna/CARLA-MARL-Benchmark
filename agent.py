import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam
from replaybuffer import ReplayBuffer
import numpy as np
import os

class CriticNetwork(nn.Module):
    def __init__(self, image_shape, state_shape, hidden_size):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the conv layers
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(image_shape[1], 8, 4), 4, 2), 3, 1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(image_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64  # 64 is the number of output channels of the last conv layer

        self.fc_image = nn.Linear(linear_input_size, hidden_size)
        self.fc_state = nn.Linear(state_shape, hidden_size)
        self.fc_combined = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_output = nn.Linear(hidden_size, 1)

    def conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, image, state):
        print(f"Input shape to CriticNetwork: {image.shape}, {state.shape}")  # Debugging statement
        if image.dim() == 4:
            image = image.unsqueeze(1)
        if state.dim() == 2:
            state = state.unsqueeze(1)

        batch_size, num_agents, channels, height, width = image.shape
        image = image.view(batch_size * num_agents, channels, height, width)  # Combine batch_size and num_agents for conv layers
        image_out = F.relu(self.conv1(image))
        image_out = F.relu(self.conv2(image_out))
        image_out = F.relu(self.conv3(image_out))
        image_out = image_out.reshape(image_out.size(0), -1)  # Flatten the tensor
        image_out = F.relu(self.fc_image(image_out))
        
        state_out = F.relu(self.fc_state(state.view(batch_size * num_agents, -1)))

        combined = torch.cat((image_out, state_out), dim=1)
        combined = F.relu(self.fc_combined(combined))
        output = self.fc_output(combined)
        return output.view(batch_size, num_agents)

class ActorNetwork(nn.Module):
    def __init__(self, image_shape, state_shape, hidden_size, num_actions):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(image_shape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the conv layers
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(image_shape[1], 8, 4), 4, 2), 3, 1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(image_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 32

        self.fc_image = nn.Linear(linear_input_size, hidden_size)
        self.fc_state = nn.Linear(state_shape, hidden_size)
        self.fc_combined = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_output = nn.Linear(hidden_size, num_actions)

    def conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, image, state):
        print(f"Input shape to ActorNetwork: {image.shape}, {state.shape}")  # Debugging statement
        if image.dim() == 4:
            image = image.unsqueeze(1)
        if state.dim() == 2:
            state = state.unsqueeze(1)

        batch_size, num_agents, channels, height, width = image.shape
        image = image.view(batch_size * num_agents, channels, height, width)  # Combine batch_size and num_agents for conv layers
        image_out = F.relu(self.conv1(image))
        image_out = F.relu(self.conv2(image_out))
        image_out = F.relu(self.conv3(image_out))
        image_out = image_out.reshape(image_out.size(0), -1)  # Flatten the tensor
        image_out = F.relu(self.fc_image(image_out))
        
        state_out = F.relu(self.fc_state(state.view(batch_size * num_agents, -1)))

        combined = torch.cat((image_out, state_out), dim=1)
        combined = F.relu(self.fc_combined(combined))
        output = self.fc_output(combined)
        probabilities = F.softmax(output, dim=1)

        # Print the probabilities for debugging
        print(f"Action probabilities: {probabilities}")

        return probabilities




class Agent:
    def __init__(self, actor, critic, replay_buf, optimizer, gamma=0.99, clip_epsilon=0.2, use_cuda=True, num_agents=2):
        self.actor = actor
        self.critic = critic
        self.replay_buf = replay_buf
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.num_agents = num_agents
        self.memory = replay_buf
        self.composite_actions = [
            (0, 0, 0),  # No action
            (1, 0, 0),  # Accelerate
            (0, -1, 0),  # Steer full left
            (0, 1, 0),  # Steer full right
            (0, 0, 1),  # Brake
            (1, -1, 0),  # Accelerate and steer full left
            (1, 1, 0),  # Accelerate and steer full right
            (1, 0, 1),  # Accelerate and brake
            (0, -1, 1),  # Steer full left and brake
            (0, 1, 1),  # Steer full right and brake
        ]
        self.num_composite_actions = len(self.composite_actions)


        if self.use_cuda:
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()

    def select_action(self, state):
        try:
            image = torch.FloatTensor(state['image']).unsqueeze(0)  # Add batch dimension
            other_state = torch.FloatTensor(state['state']).unsqueeze(0)  # Add batch dimension
        except KeyError as e:
            print(f"Missing key in state: {e}")
            return [0, 0, 0]  # Return a default action

        if self.use_cuda:
            image, other_state = image.cuda(), other_state.cuda()

        print(f"Input shape to ActorNetwork: {image.shape}, {other_state.shape}")
        if image.dim() == 4:
            image = image.unsqueeze(1)  # Add num_agents dimension if missing

        probs = self.actor(image, other_state)
        if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
            print(f"Invalid probabilities detected: {probs}")
            return [0, 0, 0]  # Return a default action
        action_idx = probs.multinomial(1).detach().cpu().numpy().flatten()
        composite_action = [self.composite_actions[idx] for idx in action_idx]
        return composite_action[0]  # Ensure correct format


    def normalize_image(self, image):
        return image / 255.0  # Assuming image pixel values are in [0, 255]

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def check_tensor(self, tensor, name):
        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
            print(f"Invalid values detected in {name}: {tensor}")
            raise ValueError(f"{name} contains NaN or Inf values")


    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Flatten the nested structure of states and next_states
        states = [item for sublist in states for item in sublist]
        next_states = [item for sublist in next_states for item in sublist]

        # Convert state dictionaries to numpy arrays
        images = np.stack([s['image'] for s in states])
        other_states = np.stack([s['state'] for s in states])
        next_images = np.stack([s['image'] for s in next_states])
        next_other_states = np.stack([s['state'] for s in next_states])

        images = torch.tensor(images, dtype=torch.float32)
        other_states = torch.tensor(other_states, dtype=torch.float32)
        next_images = torch.tensor(next_images, dtype=torch.float32)
        next_other_states = torch.tensor(next_other_states, dtype=torch.float32)

        states = (images, other_states)
        next_states = (next_images, next_other_states)

        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        if self.use_cuda:
            states = (states[0].cuda(), states[1].cuda())
            next_states = (next_states[0].cuda(), next_states[1].cuda())
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()

        images, other_states = states
        next_images, next_other_states = next_states

        with torch.no_grad():
            next_values = self.critic(next_images, next_other_states).view(-1)
            rewards = rewards.view(-1)
            dones = dones.view(-1)
            target_values = rewards + self.gamma * next_values * (1 - dones)

        values = self.critic(images, other_states).view(-1)
        critic_loss = F.mse_loss(values, target_values)

        log_probs = self.actor(images, other_states)

        # Flatten actions and log_probs for gather operation
        actions_flat = actions.view(batch_size * self.num_agents, -1)  # Flatten to match log_probs dimensions
        log_probs_flat = log_probs.view(batch_size * self.num_agents, -1)

        print(f"Actions shape: {actions.shape}")
        print(f"Actions flat shape: {actions_flat.shape}")
        print(f"Log probs shape: {log_probs.shape}")
        print(f"Log probs flat shape: {log_probs_flat.shape}")

        # Ensure that actions_flat is within the range of log_probs_flat
        actions_flat = torch.clamp(actions_flat, 0, log_probs_flat.size(1) - 1)

        # Gather log probabilities for the actions taken
        gathered_log_probs = log_probs_flat.gather(1, actions_flat)

        # Ensure correct shape for log_probs
        log_probs = gathered_log_probs.view(batch_size * self.num_agents, -1).sum(dim=1).log()
        old_log_probs = log_probs.detach()

        advantages = target_values.view(-1, 1) - values.view(-1, 1)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()




    def save_model(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        actor_path = os.path.join(directory, f"{filename}_actor.pth")
        critic_path = os.path.join(directory, f"{filename}_critic.pth")
        optimizer_path = os.path.join(directory, f"{filename}_optimizer.pth")

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def load_model(self, directory, filename):
        actor_path = os.path.join(directory, f"{filename}_actor.pth")
        critic_path = os.path.join(directory, f"{filename}_critic.pth")
        optimizer_path = os.path.join(directory, f"{filename}_optimizer.pth")

        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))

class MAPPO:
    def __init__(self, env, state_dim, action_dim, agent_params, memory_capacity=10000, use_cuda=True):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(memory_capacity)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.num_agents = agent_params['num_agents']
        self.max_steps = 1000  # Define the maximum number of steps per episode
        self.agents = []

        for agent_id in range(self.num_agents):
            actor = ActorNetwork(state_dim['image'], state_dim['state'], agent_params['actor_hidden_size'], action_dim)
            critic = CriticNetwork(state_dim['image'], state_dim['state'], agent_params['critic_hidden_size'])
            optimizer = Adam(
                list(actor.parameters()) + list(critic.parameters()),
                lr=agent_params['actor_lr']
            )
            self.agents.append(Agent(actor, critic, self.memory, optimizer, agent_params['reward_gamma'], agent_params['clip_epsilon'], self.use_cuda, self.num_agents))


    def run(self, num_episodes, batch_size):
        for episode in range(num_episodes):
            states = self.env.reset()
            for step in range(self.max_steps):
                actions = [agent.select_action(state) for agent, state in zip(self.agents, states)]
                next_states, rewards, done = self.env.step(actions)
                self.memory.push(states, actions, rewards, next_states, done)
                states = next_states
                if all(done):
                    break
            self.train(batch_size)

    def train(self, batch_size):
        for agent in self.agents:
            agent.update(batch_size)


    def interact(self):
        states = self.env.reset()

        for step in range(1000):  # Example number of steps
            actions = [agent.select_action(state) for agent, state in zip(self.agents, states)]
            print(f"Actions selected: {actions}")  # Debugging statement
            try:
                next_states, rewards, done = self.env.step(actions)
                # Compute rewards using the reward function
                rewards = [self.env.compute_reward(agent_id) for agent_id in range(self.num_agents)]
            except Exception as e:
                print(f"Error in env.step: {e}")
                print(f"Actions: {actions}")
                print(f"States: {states}")
                raise e
            self.memory.push(states, actions, rewards, next_states, done)
            states = next_states

            if done:
                print("step: " + str(step))
                break




