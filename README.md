# RECAP: RL with Experience & Corrections via Advantage-conditioned Policies

**⚠️ Note: This is an unofficial, work-in-progress implementation.**

PyTorch implementation of RECAP based on the Physical Intelligence blog post: [π*0.6: a VLA that Learns from Experience](https://www.physicalintelligence.company/blog/pistar06)

This project is not affiliated with Physical Intelligence and is provided as-is for research purposes.

## Overview

RECAP implements a three-stage training process for Vision-Language-Action (VLA) models:

1. **Demonstrations**: Supervised learning from expert demonstrations
2. **Corrections**: Learning from expert interventions when the robot makes mistakes
3. **Autonomous Experience**: Reinforcement learning with advantage-conditioned policies

The key innovation is using a value function for credit assignment and conditioning the policy on advantage values, enabling the model to learn from both good and bad experiences.

## What it contains 

- Lerobot dataset v3 format support
- Configurable HuggingFace tokenizers for text
- Configurable CLIP ViT tokenizers for vision
- Advantage-conditioned policy training
- Value function for credit assignment
- Three-stage training pipeline

## Installation

The following steps have been tested with `CUDA Version: 12.4`.

1. Clone this repository and navigate to pi06 directory:
   ```bash
   git clone <repository-url>
   cd pi06
   ```

2. Install Package:
   ```bash
   conda create -n pi06 python=3.11 -y
   conda activate pi06
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   ```

3. Install additional packages for training (optional):
   ```bash
   pip install -e ".[train]"
   ```

## Usage

### Basic Training

1. Prepare your Lerobot v3 dataset (or use an existing one)

2. Create/edit the config file (`src/pi06/configs/recap_config.yaml`):
   ```yaml
   dataset:
     path: "path/to/your/dataset"
     batch_size: 1
   
   model:
     action_dim: 7  # Adjust for your robot
   
   training:
     demo_epochs: 10
     correction_epochs: 5
     autonomous_epochs: 20
   ```

3. Run training:
   ```bash
   python -m pi06.train --config src/pi06/configs/recap_config.yaml
   ```
   
   Or if the package is installed:
   ```bash
   python -m pi06.train --config pi06/configs/recap_config.yaml
   ```

### Changing Tokenizers

You can easily change tokenizers by modifying the config:

```yaml
tokenizer:
  text_model_name: "distilbert-base-uncased"  # Change text tokenizer
  vision_model_name: "openai/clip-vit-large-patch14"  # Change vision tokenizer
```

Or programmatically:

```python
from pi06 import TokenizerWrapper

tokenizer = TokenizerWrapper()
tokenizer.set_text_tokenizer("distilbert-base-uncased")
tokenizer.set_vision_processor("openai/clip-vit-large-patch14")
```

### Dataset Format

The implementation expects Lerobot v3 format with the following structure:

- **Images**: Multi-modal image observations (can have multiple camera views)
- **Text**: Language instructions/commands
- **Actions**: Robot actions
- **Rewards**: Task rewards
- **Done flags**: Episode termination flags
- **Episode metadata**: Episode type (demo/correction/autonomous)

See [Lerobot Dataset v3 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) for details.

## Architecture

### VLA Model

- **Vision Encoder**: CLIP ViT (configurable)
- **Language Encoder**: HuggingFace transformer (configurable)
- **Fusion Layer**: Combines vision and language features
- **Action Expert**: MLP head for action prediction
- **Conditioning**: Supports advantage and quality conditioning

### Value Function

- Predicts expected future return from state features
- Used for credit assignment via GAE (Generalized Advantage Estimation)
- Enables learning from both good and bad experiences

### Training Stages

1. **Demonstrations**: Supervised learning to match expert actions
2. **Corrections**: Learn recovery strategies from expert interventions
3. **Autonomous**: RL training with advantage-conditioned policy

## Configuration

Key configuration options in `recap_config.yaml`:

- `model.action_dim`: Dimension of action space
- `model.vision_model_name`: CLIP model for vision
- `model.text_model_name`: HuggingFace model for text
- `training.gamma`: Discount factor for RL
- `training.lambda`: GAE lambda parameter
- `training.value_loss_weight`: Weight for value function loss
- `training.policy_loss_weight`: Weight for policy loss

## Logging

Metrics are logged to WandB:

- `train/demo_loss`: Demonstration training loss
- `train/policy_loss`: Policy loss (autonomous training)
- `train/value_loss`: Value function loss
- `train/advantage_mean`: Mean advantage values
- `train/entropy`: Policy entropy

## Checkpoints

Checkpoints are saved after each training stage:

- `checkpoint_demo.pt`: After demonstration training
- `checkpoint_correction.pt`: After correction training
- `checkpoint_final.pt`: After autonomous training

Resume training with:
```bash
python -m pi06.train --config src/pi06/configs/recap_config.yaml --checkpoint checkpoints/checkpoint_demo.pt
```

## References

- [π*0.6 Blog Post](https://www.physicalintelligence.company/blog/pistar06)
- [Lerobot Dataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [CLIP](https://github.com/openai/CLIP)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## License

This implementation is provided as-is for research purposes.

