import torch
import torch.nn as nn
from utils import Trainer

class ITDTrainer(Trainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        
    def train_dynamics_model(self) -> float:
        """Train the dynamics model by unrolling through policy optimization"""
        # Sample batch of transitions from buffer
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        # Clear all gradients
        self.outer_optimizer.zero_grad()
        self.inner_optimizer.zero_grad()
        
        # First get direct model prediction loss
        pred_next_states, pred_rewards = self.mdp_model(states, actions)
        state_loss = nn.MSELoss()(pred_next_states, next_states)
        reward_loss = nn.MSELoss()(pred_rewards.squeeze(), rewards)
        direct_loss = state_loss + reward_loss
        
        # Now compute policy loss using model predictions
        policy_loss = self.optimize_policy(compute_grad=True)
        
        # Combined loss allows gradients to flow through both paths
        total_loss = direct_loss + policy_loss
        
        # Backpropagate through entire computational graph
        total_loss.backward()
        
        # Update model parameters
        self.outer_optimizer.step()

        return total_loss.item()

    def optimize_policy(self, compute_grad: bool = False) -> torch.Tensor:
        """
        Optimize policy using MDP model predictions with gradient computation option.
        Args:
            compute_grad: If True, maintain computational graph for backprop
        Returns:
            policy_loss: Loss tensor (detached if compute_grad=False)
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
        if not compute_grad:
            with torch.no_grad():
                next_states_pred, rewards_pred = self.mdp_model(batch_states, batch_actions)
                next_q_values = self.q_network(next_states_pred)
                max_next_q_values = next_q_values.max(1)[0]
                predicted_returns = rewards_pred + self.gamma * max_next_q_values
        # If gradient computation is required, maintain computational graph
        else:
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

        if not compute_grad:
            # Optimize policy
            self.inner_optimizer.zero_grad()
            loss.backward()
            self.inner_optimizer.step()

            self.policy_losses.append(loss.item())
            return loss.item()
        # If gradient computation is required, return loss tensor
        else:
            return loss