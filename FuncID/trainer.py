import torch
import torch.nn as nn
from utils import Trainer, state_dict_to_tensor, tensor_to_state_dict, tensor_to_grad_list

class FuncIDTrainer(Trainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
    
    def compute_grad_norm(self, parameters):
        """Compute the norm of gradients"""
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def train_dynamics_model(self) -> float:
        """Train the dynamics model using functional implicit differentiation"""
        # Sample batch of transitions from buffer
        states, _, _, _ = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        
        # Consider all possible actions for each state
        batch_states = states.repeat_interleave(2, dim=0)  # Shape: [batch_size * 2, state_dim]
        batch_actions = torch.arange(2, device=self.device).repeat(self.batch_size)  # [0,1,0,1,...]
        
        # Get initial Q-network predictions for computing outer loss
        q_values = self.q_network(states)
        
        # Convert model parameters to single tensor for VJP
        params = state_dict_to_tensor(self.mdp_model, self.device)
        
        # Define forward function for VJP that returns Q-value predictions
        def forward(argument):
            # Convert tensor back to state dict and make predictions
            state_dict = tensor_to_state_dict(self.mdp_model, argument, self.device)
            
            # Compute Q-predictions using the current parameters
            batch_states = states.repeat_interleave(2, dim=0)
            batch_actions = torch.arange(2, device=self.device).repeat(self.batch_size)
            
            next_states_pred, rewards_pred = torch.func.functional_call(
                self.mdp_model, state_dict, (batch_states, batch_actions))
            next_q_values = self.q_network(next_states_pred)
            max_next_q_values = next_q_values.max(1)[0]
            predicted_returns = rewards_pred + self.gamma * max_next_q_values
            
            # Return the reshaped predictions that match the adjoint dimensions
            return predicted_returns.view(self.batch_size, 2)
        
        # Define a function to get q-network predictions for computing gradient
        def get_q_predictions(states):
            batch_states = states.repeat_interleave(2, dim=0)
            batch_actions = torch.arange(2, device=self.device).repeat(self.batch_size)
            
            next_states_pred, rewards_pred = self.mdp_model(batch_states, batch_actions)
            next_q_values = self.q_network(next_states_pred)
            max_next_q_values = next_q_values.max(1)[0]
            predicted_returns = rewards_pred + self.gamma * max_next_q_values
            return predicted_returns.view(self.batch_size, 2)
        
        # Create tensor for Q predictions that requires grad
        q_predictions = get_q_predictions(states)
        q_predictions.requires_grad_(True)
        
        # Compute outer loss using these predictions
        best_actions = q_predictions.argmax(dim=1)
        optimal_values = q_predictions.max(dim=1)[0]
        outer_loss = nn.MSELoss()(
            q_values.gather(1, best_actions.unsqueeze(1)).squeeze(),
            optimal_values
        )
        
        # Compute adjoint as gradient of outer loss w.r.t q_predictions
        outer_loss.backward()
        adjoint = -q_predictions.grad  # Shape matches q_predictions

        # Compute VJP
        jvp_res = torch.autograd.functional.vjp(forward, params, adjoint)
        jvp_res_transformed = tensor_to_grad_list(self.mdp_model, jvp_res[1], self.device)
        
        # Assign gradients and update model
        self.outer_optimizer.zero_grad()
        for p, g in zip(self.mdp_model.parameters(), jvp_res_transformed):
            p.grad = g
        
        # Log gradient norm
        post_vjp_grad_norm = self.compute_grad_norm(self.mdp_model.parameters())
        
        # Update model parameters
        self.outer_optimizer.step()
        
        self.model_losses.append(outer_loss.item())
        return outer_loss.item()

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