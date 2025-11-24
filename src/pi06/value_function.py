"""
Value Function for Credit Assignment in Recap

Predicts the value of a state (expected future return) to enable
credit assignment for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ValueFunction(nn.Module):
    """
    Value function that predicts expected future return from state features.
    
    Used for credit assignment: actions that increase value are good,
    actions that decrease value are bad.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256],
        activation: str = "gelu",
        output_type: str = "scalar",  # "scalar" or "distribution"
        num_bins: int = 255,  # For distribution output
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_type = output_type
        self.num_bins = num_bins
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = hidden_dim
        
        if output_type == "scalar":
            layers.append(nn.Linear(prev_dim, 1))
        elif output_type == "distribution":
            # Predict logits for value distribution
            layers.append(nn.Linear(prev_dim, num_bins))
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict value from features.
        
        Args:
            features: (B, T, D) or (B, D) tensor of state features
        
        Returns:
            Dictionary with value predictions
        """
        output = self.network(features)
        
        if self.output_type == "scalar":
            return {"value": output.squeeze(-1)}
        else:  # distribution
            return {"logits": output, "value": self._logits_to_value(output)}
    
    def _logits_to_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert distribution logits to scalar value (expected value)."""
        probs = F.softmax(logits, dim=-1)
        # Assume bins are evenly spaced from -1 to 1 (can be configured)
        bin_values = torch.linspace(-1, 1, self.num_bins, device=logits.device)
        if logits.dim() == 3:  # (B, T, num_bins)
            value = torch.sum(probs * bin_values.unsqueeze(0).unsqueeze(0), dim=-1)
        else:  # (B, num_bins)
            value = torch.sum(probs * bin_values.unsqueeze(0), dim=-1)
        return value
    
    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value function loss.
        
        Args:
            predictions: Output from forward pass
            targets: (B, T) or (B,) tensor of target values
            mask: Optional (B, T) or (B,) boolean mask for valid timesteps
        
        Returns:
            Loss value
        """
        if self.output_type == "scalar":
            pred_values = predictions["value"]
        else:
            pred_values = predictions["value"]
        
        # Ensure same shape
        if pred_values.dim() == 1 and targets.dim() == 2:
            pred_values = pred_values.unsqueeze(0).expand_as(targets)
        elif pred_values.dim() == 2 and targets.dim() == 1:
            targets = targets.unsqueeze(0).expand_as(pred_values)
        elif pred_values.dim() == 2 and targets.dim() == 2:
            # Both are (B, T), ensure they match
            if pred_values.shape != targets.shape:
                # Handle case where one might be (B, 1) and other is (B, T)
                if pred_values.shape[1] == 1:
                    pred_values = pred_values.expand_as(targets)
                elif targets.shape[1] == 1:
                    targets = targets.expand_as(pred_values)
        
        # MSE loss
        loss = F.mse_loss(pred_values, targets, reduction="none")
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            loss = loss * mask.float()
            loss = loss.sum() / (mask.sum().float() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute advantages using GAE (Generalized Advantage Estimation).
    
    Args:
        rewards: (B, T) tensor of rewards
        values: (B, T) tensor of predicted values
        dones: (B, T) tensor of done flags
        gamma: Discount factor
        lambda_: GAE lambda parameter
        normalize: Whether to normalize advantages
    
    Returns:
        advantages: (B, T) tensor of advantages
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Compute returns and advantages using GAE
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1] * (1 - dones[:, t].float())
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lambda_ * (1 - dones[:, t].float()) * gae
        advantages[:, t] = gae
    
    if normalize:
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    bootstrap_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute discounted returns.
    
    Args:
        rewards: (B, T) tensor of rewards
        dones: (B, T) tensor of done flags
        gamma: Discount factor
        bootstrap_value: Optional (B,) tensor of bootstrap values for last timestep
    
    Returns:
        returns: (B, T) tensor of returns
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    
    if bootstrap_value is not None:
        next_value = bootstrap_value
    else:
        next_value = torch.zeros(B, device=rewards.device)
    
    for t in reversed(range(T)):
        returns[:, t] = rewards[:, t] + gamma * (1 - dones[:, t].float()) * next_value
        next_value = returns[:, t]
    
    return returns

