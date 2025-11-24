"""
Lerobot Dataset v3 Loader for Recap Training

Handles loading and preprocessing of Lerobot dataset v3 format for:
- Demonstrations (supervised learning)
- Corrections (expert interventions)
- Autonomous episodes (RL training)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path
import json


class LerobotDatasetV3(Dataset):
    """
    Dataset loader for Lerobot v3 format.
    
    Lerobot v3 format structure:
    - Each episode is stored with observations, actions, rewards, etc.
    - Supports multi-modal data (images, text, proprioception)
    - Includes metadata about episode type (demo, correction, autonomous)
    """
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        episode_type: str = "all",  # "demo", "correction", "autonomous", "all"
        image_keys: Optional[List[str]] = None,
        text_key: str = "instruction",
        action_key: str = "action",
        reward_key: str = "reward",
        done_key: str = "done",
        advantage_key: Optional[str] = None,  # For precomputed advantages
        max_episode_length: Optional[int] = None,
        transform=None,
    ):
        """
        Initialize Lerobot v3 dataset.
        
        Args:
            dataset_path: Path to Lerobot dataset (HuggingFace dataset or local path)
            episode_type: Type of episodes to load
            image_keys: List of image observation keys (e.g., ["image", "image_0", "image_1"])
            text_key: Key for text instructions/commands
            action_key: Key for actions
            reward_key: Key for rewards
            done_key: Key for done flags
            advantage_key: Key for precomputed advantages (optional)
            max_episode_length: Maximum episode length to use
            transform: Optional transform to apply to images
        """
        self.dataset_path = Path(dataset_path)
        self.episode_type = episode_type
        self.image_keys = image_keys or ["image"]
        self.text_key = text_key
        self.action_key = action_key
        self.reward_key = reward_key
        self.done_key = done_key
        self.advantage_key = advantage_key
        self.max_episode_length = max_episode_length
        self.transform = transform
        
        # Try to load from HuggingFace datasets first
        try:
            from datasets import load_from_disk, load_dataset
            if self.dataset_path.exists() and self.dataset_path.is_dir():
                self.data = load_from_disk(str(self.dataset_path))
            elif isinstance(self.dataset_path, str) and '/' in str(self.dataset_path):
                # Assume it's a HuggingFace dataset name (e.g., "username/dataset_name")
                self.data = load_dataset(str(self.dataset_path))
            else:
                # Try loading as HuggingFace dataset name
                self.data = load_dataset(str(self.dataset_path))
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        except Exception as e:
            raise ValueError(f"Could not load dataset from {dataset_path}: {e}")
        
        # Filter episodes by type if specified
        if episode_type != "all":
            if "episode_type" in self.data.column_names:
                self.data = self.data.filter(lambda x: x["episode_type"] == episode_type)
            else:
                print(f"Warning: episode_type column not found, loading all episodes")
        
        # Build episode indices
        self.episode_indices = self._build_episode_indices()
        
    def _build_episode_indices(self):
        """Build indices for episode boundaries."""
        indices = []
        current_episode = []
        
        # Lerobot v3 uses episode_id or index to group steps
        if "episode_index" in self.data.column_names:
            episode_ids = self.data["episode_index"]
        elif "episode_id" in self.data.column_names:
            episode_ids = self.data["episode_id"]
        else:
            # Assume sequential episodes based on done flags
            done_flags = self.data[self.done_key] if self.done_key in self.data.column_names else None
            if done_flags is None:
                # No episode structure, treat as single episode
                return [(0, len(self.data))]
            episode_ids = []
            current_id = 0
            for done in done_flags:
                episode_ids.append(current_id)
                if done:
                    current_id += 1
        
        # Group by episode
        episodes = {}
        for idx, ep_id in enumerate(episode_ids):
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(idx)
        
        # Create (start, end) indices for each episode
        indices = []
        for ep_id, indices_list in sorted(episodes.items()):
            start = min(indices_list)
            end = max(indices_list) + 1
            if self.max_episode_length:
                end = min(end, start + self.max_episode_length)
            indices.append((start, end))
        
        return indices
    
    def __len__(self):
        return len(self.episode_indices)
    
    def __getitem__(self, idx):
        """Get a single episode."""
        start_idx, end_idx = self.episode_indices[idx]
        
        # Extract episode data
        episode_data = self.data.select(range(start_idx, end_idx))
        
        # Get images
        images = {}
        for key in self.image_keys:
            if key in episode_data.column_names:
                img_list = episode_data[key]
                # Convert PIL images to numpy/torch if needed
                if isinstance(img_list[0], np.ndarray):
                    images[key] = torch.from_numpy(np.stack(img_list))
                else:
                    # Assume PIL Image, convert to tensor
                    import torchvision.transforms as T
                    to_tensor = T.ToTensor()
                    images[key] = torch.stack([to_tensor(img) for img in img_list])
                    if self.transform:
                        images[key] = self.transform(images[key])
        
        # Get text instructions
        text = None
        if self.text_key in episode_data.column_names:
            text_list = episode_data[self.text_key]
            # Use first instruction if multiple
            text = text_list[0] if isinstance(text_list[0], str) else str(text_list[0])
        
        # Get actions
        actions = None
        if self.action_key in episode_data.column_names:
            action_list = episode_data[self.action_key]
            if isinstance(action_list[0], (list, np.ndarray)):
                actions = torch.from_numpy(np.array(action_list)).float()
            else:
                actions = torch.tensor(action_list).float()
        
        # Get rewards
        rewards = None
        if self.reward_key in episode_data.column_names:
            reward_list = episode_data[self.reward_key]
            rewards = torch.tensor(reward_list).float()
        
        # Get done flags
        done = None
        if self.done_key in episode_data.column_names:
            done_list = episode_data[self.done_key]
            done = torch.tensor(done_list).bool()
        
        # Get advantages if available
        advantages = None
        if self.advantage_key and self.advantage_key in episode_data.column_names:
            adv_list = episode_data[self.advantage_key]
            advantages = torch.tensor(adv_list).float()
        
        # Get episode metadata
        metadata = {
            "episode_index": idx,
            "episode_length": end_idx - start_idx,
        }
        if "episode_type" in episode_data.column_names:
            metadata["episode_type"] = episode_data["episode_type"][0]
        
        return {
            "images": images,
            "text": text,
            "actions": actions,
            "rewards": rewards,
            "done": done,
            "advantages": advantages,
            "metadata": metadata,
        }
    
    def get_batch(self, indices: List[int]) -> Dict:
        """Get a batch of episodes."""
        batch = [self[i] for i in indices]
        
        # Pad sequences to same length
        max_length = max(len(item["actions"]) for item in batch if item["actions"] is not None)
        
        padded_batch = {}
        for key in ["images", "actions", "rewards", "done", "advantages"]:
            if batch[0][key] is not None:
                if key == "images":
                    # Handle multiple image keys
                    padded_batch[key] = {}
                    for img_key in batch[0][key].keys():
                        padded_batch[key][img_key] = torch.stack([
                            self._pad_tensor(item[key][img_key], max_length, dim=0)
                            for item in batch
                        ])
                else:
                    padded_batch[key] = torch.stack([
                        self._pad_tensor(item[key], max_length, dim=0)
                        for item in batch
                    ])
            else:
                padded_batch[key] = None
        
        # Handle text (list of strings)
        if batch[0]["text"] is not None:
            padded_batch["text"] = [item["text"] for item in batch]
        else:
            padded_batch["text"] = None
        
        padded_batch["metadata"] = [item["metadata"] for item in batch]
        
        return padded_batch
    
    def _pad_tensor(self, tensor: torch.Tensor, target_length: int, dim: int = 0) -> torch.Tensor:
        """Pad tensor to target length."""
        current_length = tensor.shape[dim]
        if current_length >= target_length:
            return tensor[:target_length]
        
        pad_size = target_length - current_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        
        if dim == 0:
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=dim)
        else:
            raise NotImplementedError(f"Padding on dim {dim} not implemented")


def create_dataloader(
    dataset_path: Union[str, Path],
    episode_type: str = "all",
    batch_size: int = 1,  # Usually 1 for episodes, but can batch multiple episodes
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for Lerobot v3 dataset."""
    dataset = LerobotDatasetV3(dataset_path, episode_type=episode_type, **dataset_kwargs)
    
    def collate_fn(batch):
        """Custom collate function for episodes."""
        # batch is a list of episode dictionaries from __getitem__
        # Pad sequences to same length
        max_length = max(
            len(item["actions"]) if item["actions"] is not None else 0
            for item in batch
        )
        
        padded_batch = {}
        for key in ["images", "actions", "rewards", "done", "advantages"]:
            if batch[0][key] is not None:
                if key == "images":
                    # Handle multiple image keys
                    padded_batch[key] = {}
                    for img_key in batch[0][key].keys():
                        padded_batch[key][img_key] = torch.stack([
                            dataset._pad_tensor(item[key][img_key], max_length, dim=0)
                            for item in batch
                        ])
                else:
                    padded_batch[key] = torch.stack([
                        dataset._pad_tensor(item[key], max_length, dim=0)
                        for item in batch
                    ])
            else:
                padded_batch[key] = None
        
        # Handle text (list of strings)
        if batch[0]["text"] is not None:
            padded_batch["text"] = [item["text"] for item in batch]
        else:
            padded_batch["text"] = None
        
        padded_batch["metadata"] = [item["metadata"] for item in batch]
        
        return padded_batch
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

