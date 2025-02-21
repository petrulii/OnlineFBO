import torch
import torch.nn as nn
from ..utils import Trainer

class DynaTrainer(Trainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def train_dynamics_model(self) -> float:
        """Train the dynamics model using transitions from buffer"""
        # Sample batch of transitions from buffer
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        # Get model predictions
        pred_next_states, pred_rewards = self.mdp_model(states, actions)
        
        # Compute loss
        state_loss = nn.MSELoss()(pred_next_states, next_states)
        reward_loss = nn.MSELoss()(pred_rewards.squeeze(), rewards)
        loss = state_loss + reward_loss
        
        # Update model
        self.outer_optimizer.zero_grad()
        loss.backward()
        self.outer_optimizer.step()

        self.model_losses.append(loss.item())
        return loss.mean().item()

    def optimize_policy(self) -> float:
        """
        Optimize policy using MDP model predictions for all possible actions.
        """
        # Sample states from buffer as starting points
        states, _, _, _ = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        
        # Compute Q-values for the current states
        q_values = self.q_network(states)
    
        # Consider all possible actions for each state
        batch_states = states.repeat_interleave(2, dim=0)  # Shape: [batch_size * 2, state_dim]
        batch_actions = torch.arange(2, device=self.device).repeat(self.batch_size)  # [0,1,0,1,...]
        
        # Use MDP model to predict outcomes for all state-action pairs
        with torch.no_grad():
            next_states_pred, rewards_pred = self.mdp_model(batch_states, batch_actions)
            next_q_values = self.q_network(next_states_pred)
            max_next_q_values = next_q_values.max(1)[0]
            predicted_returns = rewards_pred + self.gamma * max_next_q_values

        # Reshape predictions to [batch_size, n_actions]
        predicted_returns = predicted_returns.view(self.batch_size, 2)
    
        # Get actions that maximize predicted return
        best_actions = predicted_returns.argmax(dim=1)
        optimal_values = predicted_returns.max(dim=1)[0]
    
        # Update Q-network to match model-based predictions
        loss = nn.MSELoss()(
            q_values.gather(1, best_actions.unsqueeze(1)).squeeze(),
            optimal_values
        )

        # Optimize policy
        self.inner_optimizer.zero_grad()
        loss.backward()
        self.inner_optimizer.step()

        self.policy_losses.append(loss.item())
        return loss.item()