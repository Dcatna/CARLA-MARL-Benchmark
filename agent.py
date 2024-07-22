import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam
from replaybuffer import ReplayBuffer

class ActorNetwork(nn.Module):
    def __init__(self, in_shape, num_actions, hidden_size=512):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the conv layers
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[1], 8, 4), 4, 2), 3, 1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        print(f"Input to conv1: {x}")
        x = F.relu(self.conv1(x))
        print(f"Output of conv1: {x}")
        x = F.relu(self.conv2(x))
        print(f"Output of conv2: {x}")
        x = F.relu(self.conv3(x))
        print(f"Output of conv3: {x}")
        x = x.reshape(x.size(0), -1)
        print(f"Flattened output: {x}")
        x = F.relu(self.fc1(x))
        print(f"Output of fc1: {x}")
        probs = F.softmax(self.fc2(x), dim=1)
        print(f"Output of softmax: {probs}")
        probs = torch.clamp(probs, min=1e-6)
        return probs

class CriticNetwork(nn.Module):
    def __init__(self, in_shape, hidden_size=512):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the conv layers
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[1], 8, 4), 4, 2), 3, 1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(in_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def conv2d_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Agent:
    def __init__(self, actor, critic, replay_buf, optimizer, gamma=0.99, clip_epsilon=0.2, use_cuda=True):
        self.actor = actor
        self.critic = critic
        self.replay_buf = replay_buf
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.use_cuda = use_cuda and torch.cuda.is_available()

        if self.use_cuda:
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()

    def select_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0)
            if self.use_cuda:
                state = state.cuda()
            
            # Clamping state to avoid extreme values
            state = torch.clamp(state, -1.0, 1.0)
            
            assert not torch.isnan(state).any(), "NaN value detected in state input"
            assert not torch.isinf(state).any(), "Infinite value detected in state input"
            
            probs = self.actor(state)
            
            # Additional debug information
            print(f"State input: {state}")
            print(f"Probabilities output: {probs}")

            assert not torch.isnan(probs).any(), "NaN value detected in actor output"
            assert not torch.isinf(probs).any(), "Infinite value detected in actor output"
            
            action = probs.multinomial(1).detach().cpu().numpy()[0]
            return action


    def update(self, batch_size):
        if len(self.replay_buf) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buf.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).squeeze()  # Ensure actions is (batch_size, 1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        if self.use_cuda:
            states, actions, rewards, next_states, dones = states.cuda(), actions.cuda(), rewards.cuda(), next_states.cuda(), dones.cuda()

        # Ensure actions is (batch_size, 1)
        actions = actions.view(-1, 1)

        # Compute targets for the critic
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            target_values = rewards + self.gamma * next_values * (1 - dones)
            target_values = target_values.unsqueeze(1)  # Ensure target_values is (batch_size, 1)

        values = self.critic(states).squeeze()

        # Critic loss
        critic_loss = F.mse_loss(values, target_values.squeeze())

        # Actor loss
        log_probs = self.actor(states)
        print(f"log_probs shape: {log_probs.shape}") 
        print(f"actions shape: {actions.shape}") 
        log_probs = log_probs.gather(1, actions).log()
        old_log_probs = log_probs.detach()
        advantages = target_values - values.unsqueeze(1)
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



class MAPPO:
    def __init__(self, env, state_dim, action_dim, agent_params, memory_capacity=10000, use_cuda=True):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(memory_capacity)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.num_agents = agent_params['num_agents']
        self.agents = []

        for agent_id in range(self.num_agents):
            actor = ActorNetwork(state_dim, action_dim, agent_params['actor_hidden_size'])
            critic = CriticNetwork(state_dim, agent_params['critic_hidden_size'])
            optimizer = Adam(
                list(actor.parameters()) + list(critic.parameters()), 
                lr=agent_params['actor_lr']
            )
            self.agents.append(Agent(
                actor=actor,
                critic=critic,
                replay_buf=self.memory,
                optimizer=optimizer,
                gamma=agent_params['reward_gamma'],
                clip_epsilon=agent_params['clip_param'],
                use_cuda=self.use_cuda
            ))

    def interact(self):
        states = self.env.reset()
        done = False
        while not done:
            actions = [agent.select_action(state) for agent, state in zip(self.agents, states)]
            next_states, rewards, done = self.env.step(actions)
            for i in range(self.num_agents):
                self.memory.push(states[i], actions[i], rewards[i], next_states[i], done)
            states = next_states

    def train(self, batch_size):
        for agent in self.agents:
            agent.update(batch_size)

    def run(self, num_episodes, batch_size):
        for episode in range(num_episodes):
            self.interact()
            self.train(batch_size)

