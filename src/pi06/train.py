"""
Main training script for Recap.

Usage:
    python -m pi06.train --config configs/recap_config.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Optional

from .vla_model import VLAModel, TokenizerWrapper
from .value_function import ValueFunction
from .recap_trainer import RecapTrainer
from .dataset import create_dataloader


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert numeric strings to floats if needed (handles YAML parsing inconsistencies)
    if "training" in config:
        training = config["training"]
        # Ensure learning rates and other numeric values are floats
        numeric_keys = [
            "learning_rate", "value_lr", "weight_decay", "gamma", "lambda",
            "value_loss_weight", "policy_loss_weight", "entropy_weight"
        ]
        for key in numeric_keys:
            if key in training and training[key] is not None:
                training[key] = float(training[key])
    
    return config


def create_model(config: Dict) -> VLAModel:
    """Create VLA model from config."""
    return VLAModel(
        action_dim=config["model"]["action_dim"],
        vision_model_name=config["model"]["vision_model_name"],
        text_model_name=config["model"]["text_model_name"],
        hidden_dim=config["model"]["hidden_dim"],
        freeze_vision=config["model"].get("freeze_vision", False),
        freeze_text=config["model"].get("freeze_text", False),
        use_advantage_conditioning=config["model"].get("use_advantage_conditioning", True),
        use_quality_conditioning=config["model"].get("use_quality_conditioning", True),
    )


def create_value_function(config: Dict) -> ValueFunction:
    """Create value function from config."""
    return ValueFunction(
        input_dim=config["value_function"]["input_dim"],
        hidden_dims=config["value_function"]["hidden_dims"],
        activation=config["value_function"].get("activation", "gelu"),
        output_type=config["value_function"].get("output_type", "scalar"),
        num_bins=config["value_function"].get("num_bins", 255),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Recap model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Override dataset path from config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create tokenizer
    tokenizer = TokenizerWrapper(
        text_tokenizer_name=config["tokenizer"]["text_model_name"],
        vision_processor_name=config["tokenizer"]["vision_model_name"],
    )
    
    # Create model
    print("Creating VLA model...")
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create value function
    print("Creating value function...")
    value_function = create_value_function(config)
    print(f"Value function parameters: {sum(p.numel() for p in value_function.parameters()):,}")
    
    # Create trainer
    trainer = RecapTrainer(
        model=model,
        value_function=value_function,
        tokenizer=tokenizer,
        device=device,
        learning_rate=config["training"]["learning_rate"],
        value_lr=config["training"]["value_lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
        use_wandb=config["training"].get("use_wandb", True),
        wandb_project=config["training"].get("wandb_project", "recap"),
        wandb_name=config["training"].get("wandb_name", None),
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Get dataset paths
    dataset_path = args.dataset_path or config["dataset"]["path"]
    
    # Stage 1: Train with demonstrations
    if config["training"].get("train_demonstrations", True):
        print("\n" + "="*50)
        print("Stage 1: Training with Demonstrations")
        print("="*50)
        
        demo_dataloader = create_dataloader(
            dataset_path=dataset_path,
            episode_type="demo",
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"].get("num_workers", 0),
            image_keys=config["dataset"].get("image_keys", None),
            text_key=config["dataset"].get("text_key", "instruction"),
            action_key=config["dataset"].get("action_key", "action"),
            reward_key=config["dataset"].get("reward_key", "reward"),
            done_key=config["dataset"].get("done_key", "done"),
        )
        
        trainer.train_demonstrations(
            dataloader=demo_dataloader,
            num_epochs=config["training"]["demo_epochs"],
            log_every=config["training"].get("log_every", 100),
        )
        
        # Save checkpoint after demonstrations
        checkpoint_path = Path(config["training"]["checkpoint_dir"]) / "checkpoint_demo.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Stage 2: Train with corrections
    if config["training"].get("train_corrections", True):
        print("\n" + "="*50)
        print("Stage 2: Training with Corrections")
        print("="*50)
        
        correction_dataloader = create_dataloader(
            dataset_path=dataset_path,
            episode_type="correction",
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"].get("num_workers", 0),
            image_keys=config["dataset"].get("image_keys", None),
            text_key=config["dataset"].get("text_key", "instruction"),
            action_key=config["dataset"].get("action_key", "action"),
            reward_key=config["dataset"].get("reward_key", "reward"),
            done_key=config["dataset"].get("done_key", "done"),
        )
        
        trainer.train_corrections(
            dataloader=correction_dataloader,
            num_epochs=config["training"]["correction_epochs"],
            log_every=config["training"].get("log_every", 100),
        )
        
        # Save checkpoint after corrections
        checkpoint_path = Path(config["training"]["checkpoint_dir"]) / "checkpoint_correction.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Stage 3: Train with autonomous experience
    if config["training"].get("train_autonomous", True):
        print("\n" + "="*50)
        print("Stage 3: Training with Autonomous Experience (RL)")
        print("="*50)
        
        autonomous_dataloader = create_dataloader(
            dataset_path=dataset_path,
            episode_type="autonomous",
            batch_size=config["dataset"]["batch_size"],
            shuffle=True,
            num_workers=config["dataset"].get("num_workers", 0),
            image_keys=config["dataset"].get("image_keys", None),
            text_key=config["dataset"].get("text_key", "instruction"),
            action_key=config["dataset"].get("action_key", "action"),
            reward_key=config["dataset"].get("reward_key", "reward"),
            done_key=config["dataset"].get("done_key", "done"),
        )
        
        trainer.train_autonomous(
            dataloader=autonomous_dataloader,
            num_epochs=config["training"]["autonomous_epochs"],
            log_every=config["training"].get("log_every", 100),
            gamma=config["training"].get("gamma", 0.99),
            lambda_=config["training"].get("lambda", 0.95),
            value_loss_weight=config["training"].get("value_loss_weight", 0.5),
            policy_loss_weight=config["training"].get("policy_loss_weight", 1.0),
            entropy_weight=config["training"].get("entropy_weight", 0.01),
        )
        
        # Save final checkpoint
        checkpoint_path = Path(config["training"]["checkpoint_dir"]) / "checkpoint_final.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"Saved final checkpoint to {checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

