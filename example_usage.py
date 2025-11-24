"""
Example usage of Recap training.

This script demonstrates how to use the Recap implementation
with a Lerobot v3 dataset.
"""

import torch
from pi06 import (
    VLAModel,
    ValueFunction,
    TokenizerWrapper,
    RecapTrainer,
    create_dataloader,
)


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_dim = 7  # Adjust based on your robot
    dataset_path = "path/to/lerobot/dataset"
    
    # Create tokenizer
    tokenizer = TokenizerWrapper(
        text_tokenizer_name="bert-base-uncased",
        vision_processor_name="openai/clip-vit-base-patch32",
    )
    
    # You can change tokenizers dynamically:
    # tokenizer.set_text_tokenizer("distilbert-base-uncased")
    # tokenizer.set_vision_processor("openai/clip-vit-large-patch14")
    
    # Create VLA model
    model = VLAModel(
        action_dim=action_dim,
        vision_model_name="openai/clip-vit-base-patch32",
        text_model_name="bert-base-uncased",
        hidden_dim=512,
        freeze_vision=False,
        freeze_text=False,
        use_advantage_conditioning=True,
        use_quality_conditioning=True,
    )
    
    # Create value function
    value_function = ValueFunction(
        input_dim=512,  # Should match model.hidden_dim
        hidden_dims=[512, 256],
        activation="gelu",
        output_type="scalar",
    )
    
    # Create trainer
    trainer = RecapTrainer(
        model=model,
        value_function=value_function,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-4,
        value_lr=1e-4,
        weight_decay=0.0,
        use_wandb=True,
        wandb_project="recap",
        wandb_name="example_run",
    )
    
    # Stage 1: Train with demonstrations
    print("Training with demonstrations...")
    demo_dataloader = create_dataloader(
        dataset_path=dataset_path,
        episode_type="demo",
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    
    trainer.train_demonstrations(
        dataloader=demo_dataloader,
        num_epochs=10,
        log_every=100,
    )
    
    # Stage 2: Train with corrections
    print("Training with corrections...")
    correction_dataloader = create_dataloader(
        dataset_path=dataset_path,
        episode_type="correction",
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    
    trainer.train_corrections(
        dataloader=correction_dataloader,
        num_epochs=5,
        log_every=100,
    )
    
    # Stage 3: Train with autonomous experience
    print("Training with autonomous experience...")
    autonomous_dataloader = create_dataloader(
        dataset_path=dataset_path,
        episode_type="autonomous",
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    
    trainer.train_autonomous(
        dataloader=autonomous_dataloader,
        num_epochs=20,
        log_every=100,
        gamma=0.99,
        lambda_=0.95,
        value_loss_weight=0.5,
        policy_loss_weight=1.0,
        entropy_weight=0.01,
    )
    
    # Save final checkpoint
    trainer.save_checkpoint("checkpoint_final.pt")
    print("Training complete! Checkpoint saved.")


if __name__ == "__main__":
    main()

