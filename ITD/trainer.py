import torch
import torch.nn as nn
import sys
sys.path.append('/scratch3/clear/ipetruli/projects/BILO')
from utils import Trainer, DynamicCartPole, ReplayBuffer, MDPModel, QNetwork, PerformanceTracker
import wandb

class ITDTrainer(Trainer):
    def __init__(self, config, logger):
        wandb.init(project="cartpole", group="CartPole_ITD", job_type="debug", name="ITD")
        super().__init__(config, logger)

    def optimize_networks(self):
        """Perform ITD optimization - differentiate through last inner step"""

        # (1) Inner optimization (Q-network)

        for _ in range(self.inner_iters - 1):

            # Get current state and the set of possible actions
            states, actions, _, _ = self.buffer.sample(self.batch_size)
            states, actions = states.to(self.device), actions.to(self.device)
            # Get MDP predictions and compute loss
            pred_next_states, pred_rewards = self.mdp_model(states, actions).detach()
            q_values = self.q_network(states)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            loss = self.compute_bellman_error(q_selected, pred_rewards, pred_next_states).mean()
            # Optimize Q-network
            self.inner_optimizer.zero_grad()
            loss.backward()
            self.inner_optimizer.step()
            # Log Q-network training info
            wandb.log({
                'q_loss': loss.item(),
                'epsilon': self.epsilon,
                'masscart': self.env.current_masscart,
                'masspole': self.env.current_masspole
            })
                
        # (2) Inner optimization differentiable step (Q-network)

        # Get current state and the set of possible actions
        states, actions, _, _ = self.buffer.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        # Get MDP predictions and compute loss
        pred_next_states, pred_rewards = self.mdp_model(states, actions)
        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        # Optimize Q-network
        loss = self.compute_bellman_error(q_selected, pred_rewards, pred_next_states).mean()
        self.inner_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.inner_optimizer.step()
        # Log Q-network training info
        wandb.log({
            'q_loss': loss.item(),
            'epsilon': self.epsilon,
            'masscart': self.env.current_masscart,
            'masspole': self.env.current_masspole
        })

        # (3) Outer optimization (MDP model)

        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states = rewards.to(self.device), next_states.to(self.device)

        # Get Q-values and optimize MDP model
        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        pred_next_states, pred_rewards = self.mdp_model(states, actions)
        outer_loss = nn.MSELoss()(pred_next_states, next_states) + nn.MSELoss()(pred_rewards, rewards)

        # Optimize MDP model
        self.outer_optimizer.zero_grad()
        outer_loss.backward()
        self.outer_optimizer.step()

        # Log MDP model training info
        wandb.log({
            'model_loss': outer_loss.item(),
        })

        self.total_steps += 1
        
        # Evaluetate policy
        mean_reward, std_reward = self.evaluate_policy()
        wandb.log({
            'eval_reward_mean': mean_reward,
            'eval_reward_std': std_reward
        })