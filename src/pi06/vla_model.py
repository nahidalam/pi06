"""
Vision-Language-Action (VLA) Model for Recap

Based on π0.6 architecture with:
- Vision encoder (CLIP ViT)
- Language encoder (HuggingFace tokenizer/model)
- Action expert (MLP head)
- Support for heterogeneous prompts and conditioning (advantage, quality)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPProcessor,
    CLIPImageProcessor,
)


class VisionEncoder(nn.Module):
    """Vision encoder using CLIP ViT."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_backbone: bool = False,
        output_dim: int = 512,
    ):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.model_name = model_name
        
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Projection to output dimension
        vision_dim = self.vision_model.config.hidden_size
        self.projection = nn.Linear(vision_dim, output_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images.
        
        Args:
            images: (B, T, C, H, W) or (B, C, H, W) tensor of images
        
        Returns:
            vision_features: (B, T, D) or (B, D) tensor of vision features
        """
        if images.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
            vision_outputs = self.vision_model(pixel_values=images)
            vision_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
            vision_features = vision_features.view(B, T, -1)
        else:  # (B, C, H, W)
            vision_outputs = self.vision_model(pixel_values=images)
            vision_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        vision_features = self.projection(vision_features)
        return vision_features


class LanguageEncoder(nn.Module):
    """Language encoder using HuggingFace transformers."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze_backbone: bool = False,
        output_dim: int = 512,
    ):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        
        if freeze_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        # Projection to output dimension
        text_dim = self.text_model.config.hidden_size
        self.projection = nn.Linear(text_dim, output_dim)
        
    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode text.
        
        Args:
            text_inputs: Dictionary with 'input_ids', 'attention_mask', etc.
        
        Returns:
            text_features: (B, D) tensor of text features
        """
        text_outputs = self.text_model(**text_inputs)
        # Use CLS token or mean pooling
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        else:
            # Mean pooling over sequence
            attention_mask = text_inputs.get('attention_mask', None)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(text_outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(text_outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                text_features = sum_embeddings / sum_mask
            else:
                text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        text_features = self.projection(text_features)
        return text_features


class ActionExpert(nn.Module):
    """Action prediction head."""
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256],
        activation: str = "gelu",
    ):
        super().__init__()
        self.action_dim = action_dim
        
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
        
        # Output layer (predict mean and std for continuous actions)
        layers.append(nn.Linear(prev_dim, action_dim * 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict actions from features.
        
        Args:
            features: (B, T, D) or (B, D) tensor of features
        
        Returns:
            Dictionary with 'mean' and 'std' for action distribution
        """
        output = self.network(features)
        
        if output.dim() == 3:  # (B, T, action_dim * 2)
            B, T, _ = output.shape
            output = output.view(B, T, self.action_dim, 2)
        else:  # (B, action_dim * 2)
            B = output.shape[0]
            output = output.view(B, self.action_dim, 2)
        
        mean = output[..., 0]
        std = F.softplus(output[..., 1]) + 1e-5  # Ensure positive std
        
        return {"mean": mean, "std": std}


class VLAModel(nn.Module):
    """
    Vision-Language-Action (VLA) Model.
    
    Supports:
    - Heterogeneous prompts (text commands + conditioning info)
    - Advantage conditioning (for RL training)
    - Quality annotations
    """
    
    def __init__(
        self,
        action_dim: int,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "bert-base-uncased",
        hidden_dim: int = 512,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        use_advantage_conditioning: bool = True,
        use_quality_conditioning: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_advantage_conditioning = use_advantage_conditioning
        self.use_quality_conditioning = use_quality_conditioning
        
        # Encoders
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            freeze_backbone=freeze_vision,
            output_dim=hidden_dim,
        )
        
        self.language_encoder = LanguageEncoder(
            model_name=text_model_name,
            freeze_backbone=freeze_text,
            output_dim=hidden_dim,
        )
        
        # Conditioning embeddings
        if use_advantage_conditioning:
            self.advantage_embedding = nn.Linear(1, hidden_dim)
        
        if use_quality_conditioning:
            self.quality_embedding = nn.Linear(1, hidden_dim)
        
        # Fusion layer to combine vision and language
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Action expert
        self.action_expert = ActionExpert(
            input_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dims=[512, 256],
        )
        
    def forward(
        self,
        images: torch.Tensor,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        advantage: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: (B, T, C, H, W) or (B, C, H, W) tensor of images
            text_inputs: Optional dictionary with text tokenizer outputs
            advantage: Optional (B, T) or (B,) tensor of advantage values
            quality: Optional (B, T) or (B,) tensor of quality annotations
        
        Returns:
            Dictionary with action predictions
        """
        # Encode vision
        vision_features = self.vision_encoder(images)  # (B, T, D) or (B, D)
        
        # Encode language if provided
        if text_inputs is not None:
            text_features = self.language_encoder(text_inputs)  # (B, D)
            # Expand text features to match vision sequence length
            if vision_features.dim() == 3:  # (B, T, D)
                B, T, D = vision_features.shape
                text_features = text_features.unsqueeze(1).expand(B, T, D)
            else:  # (B, D)
                text_features = text_features
        else:
            # Use zero features if no text
            if vision_features.dim() == 3:
                B, T, D = vision_features.shape
                text_features = torch.zeros(B, T, D, device=vision_features.device, dtype=vision_features.dtype)
            else:
                B, D = vision_features.shape
                text_features = torch.zeros(B, D, device=vision_features.device, dtype=vision_features.dtype)
        
        # Combine vision and language
        combined = torch.cat([vision_features, text_features], dim=-1)
        features = self.fusion(combined)
        
        # Add conditioning if provided
        if self.use_advantage_conditioning and advantage is not None:
            if advantage.dim() == 1:
                advantage = advantage.unsqueeze(-1)  # (B, 1)
            elif advantage.dim() == 2:
                advantage = advantage.unsqueeze(-1)  # (B, T, 1)
            adv_emb = self.advantage_embedding(advantage)
            features = features + adv_emb
        
        if self.use_quality_conditioning and quality is not None:
            if quality.dim() == 1:
                quality = quality.unsqueeze(-1)  # (B, 1)
            elif quality.dim() == 2:
                quality = quality.unsqueeze(-1)  # (B, T, 1)
            qual_emb = self.quality_embedding(quality)
            features = features + qual_emb
        
        # Predict actions
        actions = self.action_expert(features)
        
        return actions
    
    def extract_features(
        self,
        images: torch.Tensor,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Extract state features without predicting actions.
        Useful for value function training.
        
        Args:
            images: (B, T, C, H, W) or (B, C, H, W) tensor of images
            text_inputs: Optional dictionary with text tokenizer outputs
        
        Returns:
            features: (B, T, D) or (B, D) tensor of state features
        """
        # Encode vision
        vision_features = self.vision_encoder(images)
        
        # Encode language if provided
        if text_inputs is not None:
            text_features = self.language_encoder(text_inputs)
            # Expand text features to match vision sequence length
            if vision_features.dim() == 3:  # (B, T, D)
                B, T, D = vision_features.shape
                text_features = text_features.unsqueeze(1).expand(B, T, D)
            else:  # (B, D)
                text_features = text_features
        else:
            # Use zero features if no text
            if vision_features.dim() == 3:
                B, T, D = vision_features.shape
                text_features = torch.zeros(B, T, D, device=vision_features.device, dtype=vision_features.dtype)
            else:
                B, D = vision_features.shape
                text_features = torch.zeros(B, D, device=vision_features.device, dtype=vision_features.dtype)
        
        # Combine vision and language
        combined = torch.cat([vision_features, text_features], dim=-1)
        features = self.fusion(combined)
        
        return features


class TokenizerWrapper:
    """Wrapper for configurable tokenizers."""
    
    def __init__(
        self,
        text_tokenizer_name: str = "bert-base-uncased",
        vision_processor_name: str = "openai/clip-vit-base-patch32",
    ):
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.vision_processor = CLIPProcessor.from_pretrained(vision_processor_name)
        self.text_tokenizer_name = text_tokenizer_name
        self.vision_processor_name = vision_processor_name
    
    def set_text_tokenizer(self, model_name: str):
        """Change text tokenizer."""
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_tokenizer_name = model_name
    
    def set_vision_processor(self, model_name: str):
        """Change vision processor."""
        self.vision_processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_processor_name = model_name
    
    def tokenize_text(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize text."""
        if isinstance(text, str):
            text = [text]
        return self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            **kwargs
        )
    
    def process_images(self, images, **kwargs):
        """Process images for CLIP."""
        return self.vision_processor(images=images, return_tensors="pt", **kwargs)

