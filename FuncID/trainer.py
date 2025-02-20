import torch
import torch.nn as nn
import sys
sys.path.append('/scratch3/clear/ipetruli/projects/BILO')
from utils import Trainer, DynamicCartPole, ReplayBuffer, MDPModel, QNetwork, PerformanceTracker, state_dict_to_tensor, tensor_to_state_dict, tensor_to_grad_list
import wandb

class FuncIDTrainer(Trainer):
    def __init__(self, config, logger):
        wandb.init(project="cartpole", group="CartPole_FuncID", job_type="debug", name="FuncID")
        super().__init__(config, logger)

    def optimize_networks(self):
        """Perform Functional Implicit Differentiation optimization"""

        # (1) Inner optimization (Q-network)

        for _ in range(self.inner_iters):

            # Get current state and the set of possible actions
            states, actions, _, _ = self.buffer.sample(self.batch_size)
            states, actions = states.to(self.device), actions.to(self.device)
            # Get MDP predictions and compute loss
            pred_next_states, pred_rewards = self.mdp_model(states, actions)
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

        # (2) Outer optimization (MDP model)
            
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states = rewards.to(self.device), next_states.to(self.device)

        # Get inner solution predictions
        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        pred_next_states, pred_rewards = self.mdp_model(states, actions)

        # Compute per-sample losses
        inner_losses = self.compute_bellman_error(q_selected, pred_rewards, pred_next_states)
        outer_losses = self.compute_bellman_error(q_selected, rewards, next_states)
        
        # Compute mean losses
        inner_loss = inner_losses.mean()
        outer_loss = outer_losses.mean()
        
        # Compute adjoint as per the paper's formula
        adjoint = -outer_losses # We don't take mean here since we need per-sample gradients

        # Compute jacobian vector product between adjoint and the inner loss w.r.t. the outer model parameters
        params = state_dict_to_tensor(self.mdp_model, self.device)
        
        def forward(argument):
            state_dict = tensor_to_state_dict(self.mdp_model, argument, self.device)
            next_states_pred, rewards_pred = torch.func.functional_call(self.mdp_model, state_dict, (states, actions))
            bellman_errors = self.compute_bellman_error(q_selected, rewards_pred, next_states_pred)
            return bellman_errors # Return per-sample losses

        jvp_res = torch.autograd.functional.vjp(forward, params, adjoint, create_graph=True)
        jvp_res_transformed = tensor_to_grad_list(self.mdp_model, jvp_res[1], self.device)

        # Assign jvp_res_transformed to the model parameter gradients
        self.outer_optimizer.zero_grad()
        for p, g in zip(self.mdp_model.parameters(), jvp_res_transformed):
            p.grad = g
        # Optimize MDP model
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

    def compute_adjoint(self, state, action, reward, next_state, q_val):
        """Compute closed-form adjoint solution"""
        pred_next_state, pred_reward = self.mdp_model(state, action)
        real_error = self.compute_bellman_error(q_val, reward, next_state, real=True)
        pred_error = self.compute_bellman_error(q_val, pred_reward, pred_next_state)
        return -(real_error - pred_error)