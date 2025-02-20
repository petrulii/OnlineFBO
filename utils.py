import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces
from collections import deque

def state_dict_to_tensor(model, device):
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

def tensor_to_state_dict(model, params, device):
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

def tensor_to_grad_list(model, params, device):
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
    """CartPole environment with time-varying dynamics"""
    def __init__(self, change_frequency=1000, noise_std=0.1):
        super().__init__(gym.make("CartPole-v1"))
        self.timesteps = 0
        self.change_frequency = change_frequency
        self.noise_std = noise_std
        
        # Default parameters from CartPole
        self.initial_masscart = 1.0
        self.initial_masspole = 0.1
        self.initial_force_mag = 10.0
        
        # Current parameters
        self.current_masscart = self.initial_masscart
        self.current_masspole = self.initial_masspole
        self.current_force_mag = self.initial_force_mag
        
    def step(self, action):
        # Update parameters periodically
        if self.timesteps % self.change_frequency == 0:
            # Sinusoidal variation in cart mass
            phase = 2 * math.pi * self.timesteps / (10 * self.change_frequency)
            self.current_masscart = self.initial_masscart * (1 + 0.3 * math.sin(phase))
            
            # Random walk in pole mass
            self.current_masspole += np.random.normal(0, self.noise_std * 0.1)
            self.current_masspole = max(0.05, min(0.2, self.current_masspole))
            
            # Update environment parameters
            self.env.unwrapped.masscart = self.current_masscart
            self.env.unwrapped.masspole = self.current_masspole
            self.env.unwrapped.total_mass = self.current_masscart + self.current_masspole
            self.env.unwrapped.polemass_length = self.current_masspole * self.env.unwrapped.length
        
        self.timesteps += 1
        return super().step(action)
    
    def reset(self, **kwargs):
        self.timesteps = 0
        return super().reset(**kwargs)

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

class PerformanceTracker:
    """Tracks performance metrics over a sliding window"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.prediction_errors = deque(maxlen=window_size)
        
    def update(self, episode_reward, episode_length, prediction_error):
        self.rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.prediction_errors.append(prediction_error)
    
    def get_metrics(self):
        if not self.rewards:
            return None
        return {
            'window_avg_reward': np.mean(self.rewards),
            'window_avg_length': np.mean(self.episode_lengths),
            'window_avg_pred_error': np.mean(self.prediction_errors)
        }

class Trainer:
    def __init__(self, config, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = DynamicCartPole(
            change_frequency=config['env_change_freq'],
            noise_std=config['env_noise_std']
        )
        
        # Initialize models
        self.mdp_model = MDPModel(config['hidden_dim']).to(self.device)
        self.q_network = QNetwork(config['hidden_dim']).to(self.device)
        self.target_q = QNetwork(config['hidden_dim']).to(self.device)
        self.target_q.load_state_dict(self.q_network.state_dict())
        
        # Performance tracking
        self.tracker = PerformanceTracker(window_size=config['window_size'])
        self.total_steps = 0
        
        # Optimizers
        self.outer_optimizer = torch.optim.Adam(self.mdp_model.parameters(), lr=config['outer_lr'])
        self.inner_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config['inner_lr'])
        
        # Replay buffer
        self.buffer = ReplayBuffer(config['buffer_capacity'])
        
        # Parameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.inner_iters = config['inner_iters']
        self.episodes = config['episodes']
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if torch.rand(1).item() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compute_bellman_error(self, q_val, reward, next_state, real=False):
        """Compute Bellman error for either real or predicted transitions"""
        with torch.no_grad():
            next_q = self.target_q(next_state)
            next_v = torch.logsumexp(next_q, dim=1)
        return 0.5 * (q_val - (reward + self.gamma * next_v)) ** 2
    
    def step_environment(self, state, episode_length, total_reward, prediction_errors=None):
        """Perform one step of environment interaction and data collection
        
        Args:
            state: Current environment state
            episode_length: Current episode length
            total_reward: Accumulated reward for current episode
            prediction_errors: Optional list to store prediction errors
            
        Returns:
            tuple: (next_state, total_reward, episode_length, done)
        """
        # Environment interaction
        action = self.select_action(state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        episode_length += 1
        total_reward += reward
        
        # Store transition and compute prediction error
        self.buffer.push(state, action, reward, next_state)
        
        # Compute prediction error if we have a model and enough samples
        if hasattr(self, 'mdp_model') and len(self.buffer) > 1:
            states, actions, _, next_states = self.buffer.sample(
                min(self.batch_size, len(self.buffer))
            )
            states, actions = states.to(self.device), actions.to(self.device)
            next_states = next_states.to(self.device)
            prediction_error = self.mdp_model.compute_prediction_error(
                states, actions, next_states
            )
            if prediction_errors is not None:
                prediction_errors.append(prediction_error.item())
        
        return next_state, total_reward, episode_length, done
        
    def optimize_networks(self):
        raise NotImplementedError

    def train_episode(self):
        """Run one episode with Dyna-style learning"""
        state, _ = self.env.reset()
        total_reward = 0
        episode_length = 0
        prediction_errors = []
        done = False
        
        while not done:
            state, total_reward, episode_length, done = self.step_environment(
                state, 
                episode_length, 
                total_reward, 
                prediction_errors
            )

            # Start training when we have enough samples
        if len(self.buffer) > self.batch_size:
            self.optimize_networks()

        # Compute average prediction error
        avg_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0
        
        # Update performance tracker
        self.tracker.update(total_reward, episode_length, avg_prediction_error)
        
        return total_reward, episode_length, avg_prediction_error

    def train(self):
        """Main training loop with performance tracking"""
        
        for episode in range(self.episodes):
            episode_reward, episode_length, pred_error = self.train_episode()
            self.update_epsilon()
            
            # Get sliding window metrics
            window_metrics = self.tracker.get_metrics()

    def evaluate_policy(self, num_episodes=5):
        """Evaluate current policy on real environment without exploration"""
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action greedily (no epsilon)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    action = torch.argmax(q_values).item()
                
                # Step in real environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                
            eval_rewards.append(episode_reward)
        
        # Return mean and std of evaluation rewards
        return np.mean(eval_rewards), np.std(eval_rewards)