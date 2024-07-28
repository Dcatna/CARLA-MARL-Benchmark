import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam
from replaybuffer import ReplayBuffer
import numpy as np
import os

class CriticNetwork(nn.Module):
    def __init__(self, in_shape, hidden_size):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the conv layers
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[1], 8, 4), 4, 2), 3, 1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64  # 64 is the number of output channels of the last conv layer

        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        print(f"Input shape to CriticNetwork: {x.shape}")  # Debugging statement
        while len(x.shape) > 5:  # Squeeze unnecessary dimensions
            x = x.squeeze(1)
        batch_size, num_agents, channels, height, width = x.shape
        x = x.view(batch_size * num_agents, channels, height, width)  # Combine batch_size and num_agents for conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(batch_size, num_agents)  # Reshape back to [batch_size, num_agents]

class ActorNetwork(nn.Module):
    def __init__(self, in_shape, hidden_size, num_actions):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_shape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the conv layers
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[1], 8, 4), 4, 2), 3, 1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        print(f"Input shape to ActorNetwork: {x.shape}")  # Debugging statement
        while len(x.shape) > 5:  # Squeeze unnecessary dimensions
            x = x.squeeze(1)
        batch_size, num_agents, channels, height, width = x.shape
        x = x.view(batch_size * num_agents, channels, height, width)  # Combine batch_size and num_agents for conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)



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

        if self.use_cuda:
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        if self.use_cuda:
            state = state.cuda()
        print(f"Input shape to ActorNetwork: {state.shape}")  # Debugging statement
        if state.dim() == 4:  # Check if the state tensor has 4 dimensions
            state = state.unsqueeze(1)  # Add num_agents dimension if missing
        probs = self.actor(state)
        if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
            raise ValueError("Probs tensor contains NaN or Inf values")
        action = probs.multinomial(1).detach().cpu().numpy()[0]
        action = [float(a) for a in action] + [0.0] * (3 - len(action))
        return action


        
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Convert NumPy arrays to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Move tensors to GPU if needed
        if self.use_cuda:
            states, actions, rewards, next_states, dones = states.cuda(), actions.cuda(), rewards.cuda(), next_states.cuda(), dones.cuda()

        # Ensure the tensors have the correct shape
        if states.dim() == 4:
            states = states.unsqueeze(2)  # Add the num_frames dimension
            next_states = next_states.unsqueeze(2)

        # Reshape states and next_states
        states = states.reshape(batch_size, self.num_agents, 3, 480, 640)
        next_states = next_states.reshape(batch_size, self.num_agents, 3, 480, 640)

        # Ensure shapes match
        print(f"Reshaped states shape: {states.shape}")
        print(f"Reshaped next_states shape: {next_states.shape}")

        # Compute values and targets
        values = self.critic(states).view(batch_size * self.num_agents, -1).mean(dim=1)
        next_values = self.critic(next_states).view(batch_size * self.num_agents, -1).mean(dim=1)

        # Reshape next_values for compatibility
        next_values = next_values.view(batch_size, self.num_agents, -1)

        # Reshape rewards and dones
        rewards = rewards.view(batch_size, self.num_agents, -1)
        dones = dones.view(batch_size, self.num_agents, -1)

        # Compute target values
        target_values = rewards + self.gamma * next_values * (1 - dones)
        target_values = target_values.view(batch_size * self.num_agents, -1)  # Ensure shape matches values

        # Check shapes for debugging
        print(f"Values shape: {values.shape}")
        print(f"Next values shape after critic: {next_values.shape}")
        print(f"Target values shape: {target_values.shape}")

        # Ensure the dimensions for MSE loss match
        if values.dim() == 1 and target_values.dim() == 2:
            values = values.unsqueeze(1)  # Make values shape [batch_size * num_agents, 1]

        assert values.shape == target_values.shape, f"Shape mismatch: values shape {values.shape}, target_values shape {target_values.shape}"

        # Critic loss
        critic_loss = F.mse_loss(values, target_values)

        # Actor loss
        log_probs = self.actor(states)
        print(f"log_probs shape: {log_probs.shape}")
        print(f"actions shape: {actions.shape}")
        actions = actions.view(batch_size * self.num_agents, -1)  # Ensure actions shape matches log_probs
        log_probs = log_probs.gather(1, actions).squeeze().log()
        old_log_probs = log_probs.detach()
        advantages = target_values.view(-1, 1) - values.view(-1, 1)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Backpropagation
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
        #self.replay_buf = ReplayBuffer(100000)

        for agent_id in range(self.num_agents):
            actor = ActorNetwork(state_dim, agent_params['actor_hidden_size'], action_dim)
            critic = CriticNetwork(state_dim, agent_params['critic_hidden_size'])
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


    def interact(self):
        states = self.env.reset()

        for step in range(1000):  # Example number of steps
            actions = [agent.select_action(state) for agent, state in zip(self.agents, states)]
            print(f"Actions selected: {actions}")  # Debugging statement
            try:
                next_states, rewards, done = self.env.step(actions)
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

    def train(self, batch_size):
        for agent in self.agents:
            agent.update(batch_size)





