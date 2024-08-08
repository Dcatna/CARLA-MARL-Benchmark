import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from replaybuffer import ReplayBuffer
import os
import matplotlib.pyplot as plt
import copy

torch.autograd.set_detect_anomaly(True)


class ActorNetwork(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out), dim=1)  # Specify the dimension for log softmax
        return out

    def get_log_probs(self, state, action):
        log_probs = self(state)
        selected_log_probs = log_probs.gather(1, action.argmax(dim=1, keepdim=True)).squeeze(-1)
        return selected_log_probs

    def get_old_log_probs(self, state, action):
        log_probs = self(state)
        selected_log_probs = log_probs.gather(1, action.argmax(dim=1, keepdim=True)).squeeze(-1)
        return selected_log_probs



class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, num_agents=2, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size * num_agents)  # Output size multiplied by number of agents
        self.num_agents = num_agents

    def forward(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        print(f"Shape of state features after fc1: {out.shape}")
        print(f"Shape of action before concat: {action.shape}")
        out = torch.cat([out, action], 1)
        print(f"Shape after concatenation: {out.shape}")
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.view(-1, self.num_agents)  # Reshape to [batch_size, num_agents]
        return out




class Agent:
    def __init__(self, actor, critic, memory, optimizer, gamma, clip_epsilon, use_cuda, num_agents, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, action_dim=6, max_grad_norm=0.5, target_update_steps=5):
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.use_cuda = use_cuda
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.criterion = nn.MSELoss()
        self.action_dim = action_dim
        self.max_grad_norm = max_grad_norm
        self.target_update_steps = target_update_steps
        self.n_episodes = 0

        # Initialize composite actions
        self.composite_actions = [
            (1, 0, 0),  # Accelerate
            (0, -1, 0),  # Steer full left
            (0, 1, 0),  # Steer full right
            (0, 0, 1),  # Brake
            (1, -1, 0),  # Accelerate and steer full left
            (1, 1, 0),  # Accelerate and steer full right
        ]
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
        
        # Initialize weights
        self.actor.apply(self.initialize_weights)
        self.critic.apply(self.initialize_weights)
        self.image_shape = (3, 480, 640)
        self.state_shape = (12,)

        self.action_counter = {i: 0 for i in range(len(self.composite_actions))}
    def select_action(self, state):
        other_state = torch.tensor(state['state'], dtype=torch.float32).unsqueeze(0).to(self.device)

        if np.random.rand() < self.epsilon:
            print("GGREEDY ACCTION\n\n\n")
            action_idx = np.random.choice(len(self.composite_actions))
        else:
            with torch.no_grad():
                probs = F.softmax(self.actor(other_state), dim=1)
            print("NON GREEDDY")
            action_idx = np.argmax(probs.cpu().numpy().flatten())
        self.action_counter[action_idx] += 1

        return action_idx

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return None  # No update if not enough samples

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack([s['state'] for s in states]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack([s['state'] for s in next_states]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # One-hot encode the actions
        actions = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float()

        rewards = self.normalize_rewards(rewards.view(batch_size, self.num_agents))
        dones = dones.view(batch_size, self.num_agents)

        state_features = states
        next_state_features = next_states

        critic_values = self.critic(state_features, actions)
        next_critic_values = self.critic(next_state_features, actions).detach()

        # Debugging to check dimensions
        print(f"Shape of critic_values: {critic_values.shape}")
        print(f"Shape of next_critic_values: {next_critic_values.shape}")

        # Reshape only if the total size matches
        if next_critic_values.numel() == batch_size * self.num_agents:
            next_values = next_critic_values.view(batch_size, self.num_agents)
        else:
            next_values = next_critic_values.view(batch_size, -1)

        target_values = rewards + (self.gamma * next_values * (1 - dones))
        advantages = target_values - critic_values.view(batch_size, self.num_agents)
        advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-5)
        advantages = advantages.mean(dim=1).unsqueeze(-1)

        log_probs = self.actor.get_log_probs(states, actions)
        old_log_probs = self.actor.get_old_log_probs(states, actions).detach()

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        target_values = target_values.view(batch_size * self.num_agents, -1)
        critic_values = critic_values.view(batch_size * self.num_agents, -1)
        value_loss = self.criterion(critic_values, target_values)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.n_episodes % self.target_update_steps == 0:
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)

        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient: {param.grad.norm()}")

        self.n_episodes += 1

        print(f"Critic loss: {value_loss.item()}")
        print(f"Policy loss: {policy_loss.item()}")

        return value_loss.item()





    def check_for_nans(self, tensor, name):
        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
            raise ValueError(f"{name} contains NaN or Inf values")

    def normalize_image(self, image):
        return image / 255.0  # Assuming image pixel values are in [0, 255]

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def eps_decay(self,):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            
    def visualize_policy(self):
        with torch.no_grad():
            # Create a sample input with the same dimensions as the training data
            sample_image = torch.randn(1, *self.image_shape).to(self.device)
            sample_state = torch.randn(1, *self.state_shape).to(self.device)

            # Forward pass through the actor network
            combined_features = torch.cat((sample_image, sample_state), dim=1)
            probs = F.softmax(self.actor(combined_features), dim=1).cpu().numpy().flatten()

            # Plot the policy probabilities
            actions = range(len(self.composite_actions))
            plt.bar(actions, probs)
            plt.xlabel('Actions')
            plt.ylabel('Probability')
            plt.title('Policy Visualization')
            plt.show()

    def store_transition(self, state, action, reward, next_state, done):
            self.memory.push(state, action, reward, next_state, done)
            # Debug statement
            print(f"Transition stored: State: {state}, Action: {action}, Reward: {reward}, Next state: {next_state}, Done: {done}")
            
    def monitor_memory(self,):
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Current allocated memory: {allocated:.2f} GB")
            print(f"Max allocated memory: {max_allocated:.2f} GB")



    def normalize_rewards(self, rewards):
        mean = rewards.mean()
        std = rewards.std() + 1e-8  # Adding a small value to avoid division by zero
        normalized_rewards = (rewards - mean) / std
        return normalized_rewards

    def soft_update(self, target_net, source_net, tau=0.005):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



    def save(self, filepath):
        state = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.actor.load_state_dict(state['actor_state_dict'])
        self.critic.load_state_dict(state['critic_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epsilon = state['epsilon']


class MAPPO:
    def __init__(self, env, state_dim, action_dim, agent_params):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(agent_params['memory_capacity'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agents = []
        for _ in range(agent_params['num_agents']):
            actor = ActorNetwork(state_dim['state'], agent_params['actor_hidden_size'], action_dim, agent_params['actor_output_act']).to(self.device)
            critic = CriticNetwork(state_dim['state'], action_dim, agent_params['critic_hidden_size']).to(self.device)
            agent = Agent(actor, critic, self.memory, None, agent_params['reward_gamma'], agent_params['clip_epsilon'], torch.cuda.is_available(), agent_params['num_agents'], max_grad_norm=agent_params['max_grad_norm'])
            self.agents.append(agent)

        # Create a list of all the agent parameters for the optimizer
        actor_parameters = []
        critic_parameters = []
        all_parameters = []
        for agent in self.agents:
            actor_parameters += list(agent.actor.parameters())
            critic_parameters += list(agent.critic.parameters())
            all_parameters += list(agent.actor.parameters())
            all_parameters += list(agent.critic.parameters())
        # Define separate optimizers for actor and critic networks
        self.actor_optimizer = Adam(actor_parameters, lr=agent_params['actor_lr'])
        self.critic_optimizer = Adam(critic_parameters, lr=agent_params['critic_lr'])
        self.optimizer = Adam(all_parameters, lr=agent_params['actor_lr'])  # Shared optimizer
        for agent in self.agents:
            agent.actor_optimizer = self.actor_optimizer
            agent.critic_optimizer = self.critic_optimizer
            agent.optimizer = self.optimizer
            

    def run(self, num_episodes, batch_size, update_interval=10):
        episode_rewards = [None] * num_episodes
        critic_losses = []
        update_steps = 0
        
        def flatten_done(done):
            return [item for sublist in done for item in sublist]

        def normalize_and_clip_rewards(rewards, clip_value=10):
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-5)
            clipped_rewards = np.clip(normalized_rewards, -clip_value, clip_value)
            return clipped_rewards.tolist()

        for episode in range(num_episodes):
            states = self.env.reset()
            done = [(False, False)] * len(self.agents)  # Initialize done as a list of tuples

            print(f"Episode {episode} starting.")

            episode_rewards_list = []  # To store all rewards for this episode

            while not all(flatten_done(done)):
                action_indices = [agent.select_action(state) for agent, state in zip(self.agents, states)]
                actions = [self.agents[0].composite_actions[idx] for idx in action_indices]  # Convert indices to actual actions

                next_states, rewards, done = self.env.step(actions)

                if isinstance(rewards, (float, int)):
                    rewards = [rewards] * len(self.agents)
                elif isinstance(rewards, list):
                    rewards = [(reward, reward) for reward in rewards]

                if isinstance(done, bool):
                    done = [(done, done)] * len(self.agents)
                elif isinstance(done, list) and isinstance(done[0], bool):
                    done = [(d, d) for d in done]

                episode_rewards_list.extend(rewards)

                for i, agent in enumerate(self.agents):
                    agent.memory.push(states[i], action_indices[i], rewards[i], next_states[i], done[i])

                states = next_states

                update_steps += 1
                if episode % 5 == 0 and len(self.memory) >= batch_size:
                    for agent in self.agents:
                        critic_loss = agent.update(batch_size)
                        if critic_loss is not None:
                            critic_losses.append(critic_loss)
            
            for agent in self.agents:
                agent.eps_decay()
            episode_rewards_list = normalize_and_clip_rewards(episode_rewards_list)
            print("REWARDS LIST:  ", episode_rewards_list, "\n\n")
            for i, agent in enumerate(self.agents):
                episode_rewards[episode] = sum(episode_rewards_list[i])
            print(f"EPiSODE: {episode}")

        # Plotting rewards and critic loss
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(range(num_episodes), episode_rewards)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Total Reward per Episode')
        ax2.plot(range(len(critic_losses)), critic_losses)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Critic Loss')
        ax2.set_title('Critic Loss per Episode')
        plt.tight_layout()
        plt.show()


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

    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(directory, f'agent_{i}.pth'))

    def load(self, directory):
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(directory, f'agent_{i}.pth'))
