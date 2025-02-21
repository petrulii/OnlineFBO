import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import numpy as np
import math
import wandb
from gymnasium import spaces
from collections import deque
from typing import Tuple, List, Optional

def state_dict_to_tensor(model, device) -> torch.Tensor:
  """
  A function to wrap a tensor as parameter dictionary of a neural network.
    param model: neural network
  """
  start = 0
  current_tensor_list = []
  current_dict = model.state_dict()
  for name, param in model.named_parameters():
    t = param.clone().detach().flatten()
    current_tensor_list.append(t)
  params = (torch.cat(current_tensor_list, 0)).to(device)#put to cuda
  return params

def tensor_to_state_dict(model, params, device) -> dict:
  """
  A function to wrap a tensor as parameter dictionary of a neural network.
    param model: neural network
    param params: a tensor to be wrapped
  """
  start = 0
  current_dict = model.state_dict()
  for name, param in model.named_parameters():
      dim = torch.tensor(param.size())
      length = torch.prod(dim).item()
      end = start + length
      new_weights = params[start:end].view_as(param).to(device)
      current_dict[name] = new_weights
      start = end
  return current_dict

def tensor_to_grad_list(model, params, device) -> List[torch.Tensor]:
    """
    A function to wrap a tensor as list of gradients of parameters of a neural network.
        param model: neural network
        param params: a tensor to be wrapped
    """
    start = 0
    current_dict = model.state_dict()
    grad_list = []
    for name, param in model.named_parameters():
        dim = torch.tensor(param.size())
        length = torch.prod(dim).item()
        end = start + length
        new_weights = params[start:end].view_as(param).to(device)
        grad_list.append(new_weights)
        start = end
    return grad_list

class DynamicCartPole(gym.Wrapper):
    """CartPole environment with smoothly varying cart mass"""
    def __init__(self, change_frequency=10000, amplitude=0.15):
        super().__init__(gym.make("CartPole-v1"))
        self.timesteps = 0
        self.total_timesteps = 0
        self.change_frequency = change_frequency
        self.amplitude = amplitude  # Controls how much the mass varies (e.g. 0.15 = ±15%)
        self.base_masscart = self.env.unwrapped.masscart  # Store the initial mass
        
    def step(self, action):
        # Compute phase
        phase = 2 * math.pi * (self.total_timesteps / self.change_frequency)

        # Reset to base mass and apply variation
        self.env.unwrapped.masscart = self.base_masscart * (1 + self.amplitude * math.sin(phase))
        
        # Update total mass
        self.env.unwrapped.total_mass = self.env.unwrapped.masscart + self.env.unwrapped.masspole
        
        self.timesteps += 1
        self.total_timesteps += 1
        return super().step(action)

    def reset(self, **kwargs):
        """Standard reset - starts new episode with initial conditions"""
        obs, info = super().reset(**kwargs)
        self.timesteps = 0  # Reset timesteps since it's a new episode
        self.total_timesteps += 1
        return obs, info

class ReplayBuffer:
    """Stores transitions from CartPole environment"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        """
        State: [position, velocity, angle, angular_velocity]
        Action: 0 (left) or 1 (right)
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            action,
            np.float32(reward),
            np.array(next_state, dtype=np.float32)
        ))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = [], [], [], []
        
        for idx in indices:
            s, a, r, ns = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
        
        return (torch.FloatTensor(np.array(states)),  # Shape: [batch_size, 4]
                torch.LongTensor(actions),            # Shape: [batch_size]
                torch.FloatTensor(rewards),           # Shape: [batch_size]
                torch.FloatTensor(np.array(next_states)))  # Shape: [batch_size, 4]
    
    def __len__(self):
        return len(self.buffer)

class MDPModel(nn.Module):
    """Model for predicting next state and reward for CartPole"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.state_dim = 4
        self.action_dim = 2
        
        # Separate networks for state prediction and reward prediction
        self.state_network = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.state_dim)
        )
        
        self.reward_network = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        # One-hot encode the discrete action
        batch_size = state.size(0)
        action_onehot = torch.zeros(batch_size, self.action_dim, device=state.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=1)
        
        # Predict state differences and reward
        state_diff = self.state_network(x)
        next_state = state + state_diff  # Residual connection
        reward = self.reward_network(x).squeeze(-1)
        
        return next_state, reward
    
    def compute_prediction_error(self, states, actions, next_states):
        pred_next_states, _ = self(states, actions)
        return torch.mean((pred_next_states - next_states).pow(2))

class QNetwork(nn.Module):
    """Action-value function approximator for CartPole"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.state_dim = 4  # CartPole state dimension
        self.action_dim = 2  # CartPole actions (left/right)
        
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class Trainer:
    """Base class for training MBRL agents"""
    def __init__(self, config, logger):
        wandb.init(project="cartpole", job_type="debug", name=config['wandb_name'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = DynamicCartPole(
            change_frequency=config['env_change_freq']
        )
        
        # Initialize models
        self.mdp_model = MDPModel(config['hidden_dim']).to(self.device)
        self.q_network = QNetwork(config['hidden_dim']).to(self.device)
        
        # Optimizers
        self.outer_optimizer = torch.optim.Adam(self.mdp_model.parameters(), lr=config['outer_lr'])
        self.inner_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config['inner_lr'])
        
        # Replay buffer
        self.buffer = ReplayBuffer(config['buffer_capacity'])
        
        # Tracking metrics
        self.returns = []
        self.model_losses = []
        self.policy_losses = []
        
        # Training parameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.inner_iters = config['inner_iters']
        self.episodes = config['episodes']
        self.max_episode_steps = config['max_episode_steps']
        self.buffer_capacity = config['buffer_capacity']
        self.epsilon = config['epsilon_start']
        #self.epsilon_min = config['epsilon_min']
        #self.epsilon_decay = config['epsilon_decay']
        self.log_interval = config['log_interval']
        self.logger = logger
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def evaluate_policy(self, eval_episodes: int = 5, eval_steps: int = 1000) -> Tuple[float, float]:
        """
        Evaluate current policy without exploration
        
        Args:
            eval_episodes: Number of episodes to evaluate
            eval_steps: Maximum steps per evaluation episode
            
        Returns:
            Tuple of (average reward per step, average cart mass)
        """
        total_rewards = []
        all_masses = []
        
        for _ in range(eval_episodes):
            eval_state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < eval_steps:
                # Select action using only the policy network (no exploration)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(eval_state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    action = torch.argmax(q_values).item()
                
                # Execute action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                wandb.log({'cart_mass': self.env.unwrapped.masscart})
                episode_reward += reward
                
                # Update state and step count
                eval_state = next_state
                steps += 1
                
                # If episode ends, reset but continue counting steps
                if terminated or truncated:
                    eval_state, _ = self.env.reset()
            
            total_rewards.append(episode_reward)
        
        # Calculate averages across all episodes
        avg_reward = np.mean(total_rewards) / eval_steps
        
        return avg_reward
    
    def plan_actions(self, state: torch.Tensor) -> int:
        """Greedily plan an action using the dynamics model and Q-network."""
        state = state.to(self.device)
        if np.random.rand() < self.epsilon:  # Add epsilon-greedy exploration
            return self.env.action_space.sample()
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            action = torch.argmax(q_values).item()
        return action

    def train(self) -> None:
        """Main training loop implementing MBRL"""

        for episode in range(self.episodes):
            episode_total_reward = 0
            state, _ = self.env.reset()

            for _ in range(self.max_episode_steps):
                # 1. Environment Interaction with Planning
                if len(self.buffer) > self.batch_size:
                    # Plan action using dynamics model (and epsilon-greedy)
                    action = self.plan_actions(torch.FloatTensor(state))
                else:
                    # Fill the initial buffer with random actions
                    while len(self.buffer) <= self.batch_size:
                        action = self.env.action_space.sample()
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        self.buffer.push(state, action, reward, next_state)
                        state = next_state
                # Execute action in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_total_reward += reward
                wandb.log({'cart_mass': self.env.unwrapped.masscart})
                # Store transition
                self.buffer.push(state, action, reward, next_state)
                state = next_state
                # Check if episode is terminated
                if terminated or truncated:
                    break
                        
                # 2. Model Learning
                if len(self.buffer) > self.batch_size:
                    model_loss = self.train_dynamics_model()
                    wandb.log({'model_loss': model_loss})

                # 3. Policy Optimization (inner loop)
                if len(self.buffer) > self.batch_size:
                    for _ in range(self.inner_iters):
                        policy_loss = self.optimize_policy()
                        wandb.log({'policy_loss': policy_loss})

            wandb.log({'env_interaction_reward': episode_total_reward})