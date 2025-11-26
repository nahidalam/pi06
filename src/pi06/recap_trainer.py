"""
Recap Training: RL with Experience & Corrections via Advantage-conditioned Policies

Implements the three-stage training process:
1. Demonstrations (supervised learning)
2. Corrections (expert interventions)
3. Autonomous experience (RL with advantage conditioning)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Union
import numpy as np
from tqdm import tqdm
import wandb

from .vla_model import VLAModel, TokenizerWrapper
from .value_function import ValueFunction, compute_advantages, compute_returns
from .dataset import LerobotDatasetV21


class RecapTrainer:
    """
    Trainer for Recap method.
    
    Training stages:
    1. Pretrain with offline RL (optional)
    2. Fine-tune with demonstrations
    3. Train with corrections and autonomous experience
    """
    
    def __init__(
        self,
        model: VLAModel,
        value_function: ValueFunction,
        tokenizer: TokenizerWrapper,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        value_lr: float = 1e-4,
        weight_decay: float = 0.0,
        use_wandb: bool = True,
        wandb_project: str = "recap",
        wandb_name: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.value_function = value_function.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.value_optimizer = optim.AdamW(
            self.value_function.parameters(),
            lr=value_lr,
            weight_decay=weight_decay,
        )
        
        # Loss functions
        self.action_loss_fn = nn.MSELoss()
        
        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "learning_rate": learning_rate,
                    "value_lr": value_lr,
                    "weight_decay": weight_decay,
                }
            )
    
    def train_demonstrations(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        log_every: int = 100,
    ):
        """
        Stage 1: Train with demonstrations (supervised learning).
        
        Args:
            dataloader: DataLoader with demonstration episodes
            num_epochs: Number of training epochs
            log_every: Log metrics every N steps
        """
        self.model.train()
        step = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in tqdm(dataloader, desc=f"Demonstration Epoch {epoch+1}"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Tokenize text
                text_inputs = None
                if batch["text"] is not None:
                    text_inputs = self.tokenizer.tokenize_text(batch["text"])
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                # Get images
                images = batch["images"][list(batch["images"].keys())[0]]  # Use first image key
                images = images.to(self.device)
                
                # Get actions
                actions = batch["actions"].to(self.device)
                
                # Forward pass
                action_preds = self.model(
                    images=images,
                    text_inputs=text_inputs,
                    advantage=None,  # No advantage for demonstrations
                    quality=None,
                )
                
                # Compute loss (MSE on action means)
                pred_actions = action_preds["mean"]
                loss = self.action_loss_fn(pred_actions, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                step += 1
                
                # Logging
                if step % log_every == 0:
                    avg_loss = np.mean(epoch_losses[-log_every:])
                    print(f"Step {step}, Loss: {avg_loss:.4f}")
                    if self.use_wandb:
                        wandb.log({
                            "train/demo_loss": avg_loss,
                            "train/step": step,
                        })
            
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            if self.use_wandb:
                wandb.log({
                    "train/demo_epoch_loss": avg_epoch_loss,
                    "train/epoch": epoch + 1,
                })
    
    def train_corrections(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        log_every: int = 100,
    ):
        """
        Stage 2: Train with corrections (expert interventions).
        
        Similar to demonstrations but focuses on recovery from mistakes.
        """
        # Corrections are treated similarly to demonstrations
        # but may have different weighting or sampling
        self.train_demonstrations(dataloader, num_epochs, log_every)
    
    def train_autonomous(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        log_every: int = 100,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        value_loss_weight: float = 0.5,
        policy_loss_weight: float = 1.0,
        entropy_weight: float = 0.01,
    ):
        """
        Stage 3: Train with autonomous experience (RL with advantage conditioning).
        
        Args:
            dataloader: DataLoader with autonomous episodes
            num_epochs: Number of training epochs
            log_every: Log metrics every N steps
            gamma: Discount factor
            lambda_: GAE lambda parameter
            value_loss_weight: Weight for value function loss
            policy_loss_weight: Weight for policy loss
            entropy_weight: Weight for entropy regularization
        """
        self.model.train()
        self.value_function.train()
        step = 0
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                "policy_loss": [],
                "value_loss": [],
                "entropy": [],
                "advantage_mean": [],
                "advantage_std": [],
            }
            
            for batch in tqdm(dataloader, desc=f"Autonomous Epoch {epoch+1}"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Get data
                images = batch["images"][list(batch["images"].keys())[0]]
                images = images.to(self.device)
                actions = batch["actions"].to(self.device)
                rewards = batch["rewards"].to(self.device)
                dones = batch["done"].to(self.device)
                
                # Tokenize text if available
                text_inputs = None
                if batch["text"] is not None:
                    text_inputs = self.tokenizer.tokenize_text(batch["text"])
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                B, T = actions.shape[:2]
                
                # Extract features for value function
                state_features = self.model.extract_features(images, text_inputs)
                
                # Predict values
                value_preds = self.value_function(state_features)
                values = value_preds["value"]  # (B, T)
                
                # Compute returns and advantages
                returns = compute_returns(rewards, dones, gamma=gamma)
                advantages = compute_advantages(
                    rewards, values, dones, gamma=gamma, lambda_=lambda_, normalize=True
                )
                
                # Value function loss
                value_loss = self.value_function.loss(value_preds, returns, mask=~dones)
                
                # Policy loss with advantage conditioning
                # Forward pass with advantage conditioning
                action_preds = self.model(
                    images=images,
                    text_inputs=text_inputs,
                    advantage=advantages,  # Condition on advantages
                    quality=None,
                )
                
                # Compute policy loss
                # Use advantage-weighted log probability
                pred_mean = action_preds["mean"]
                pred_std = action_preds["std"]
                
                # Compute log probability of actions
                dist = torch.distributions.Normal(pred_mean, pred_std)
                log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum over action dims
                
                # Advantage-weighted policy loss
                policy_loss = -(log_probs * advantages).mean()
                
                # Entropy regularization
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Total loss
                total_loss = (
                    policy_loss_weight * policy_loss +
                    value_loss_weight * value_loss -
                    entropy_weight * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.value_optimizer.step()
                
                # Track metrics
                epoch_metrics["policy_loss"].append(policy_loss.item())
                epoch_metrics["value_loss"].append(value_loss.item())
                epoch_metrics["entropy"].append(entropy.item())
                epoch_metrics["advantage_mean"].append(advantages.mean().item())
                epoch_metrics["advantage_std"].append(advantages.std().item())
                
                step += 1
                
                # Logging
                if step % log_every == 0:
                    avg_metrics = {k: np.mean(v[-log_every:]) for k, v in epoch_metrics.items()}
                    print(f"Step {step}")
                    for k, v in avg_metrics.items():
                        print(f"  {k}: {v:.4f}")
                    
                    if self.use_wandb:
                        log_dict = {f"train/{k}": v for k, v in avg_metrics.items()}
                        log_dict["train/step"] = step
                        wandb.log(log_dict)
            
            # Epoch summary
            avg_epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            print(f"Epoch {epoch+1} summary:")
            for k, v in avg_epoch_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            if self.use_wandb:
                log_dict = {f"train/epoch_{k}": v for k, v in avg_epoch_metrics.items()}
                log_dict["train/epoch"] = epoch + 1
                wandb.log(log_dict)
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        device_batch = {}
        for k, v in batch.items():
            if k == "images":
                device_batch[k] = {img_k: img_v.to(self.device) for img_k, img_v in v.items()}
            elif k == "text":
                device_batch[k] = v  # Keep as list
            elif k == "metadata":
                device_batch[k] = v  # Keep as list
            elif isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device)
            else:
                device_batch[k] = v
        return device_batch
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "value_function_state_dict": self.value_function.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.value_function.load_state_dict(checkpoint["value_function_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])

