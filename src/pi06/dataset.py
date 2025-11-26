"""
Lerobot Dataset v2.1 Loader for Recap Training

Handles loading and preprocessing of Lerobot dataset v2.1 format for:
- Demonstrations (supervised learning)
- Corrections (expert interventions)
- Autonomous episodes (RL training)

Dataset v2.1 structure:
<dataset_name>/
├── data/chunk-000/episode_*.parquet
├── videos/chunk-000/observation.images.*/episode_*.mp4
└── meta/episodes.jsonl
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path
import json
import re


class LerobotDatasetV21(Dataset):
    """
    Dataset loader for Lerobot v2.1 format.
    
    Lerobot v2.1 format structure:
    - Episodes stored as parquet files in data/chunk-*/episode_*.parquet
    - Videos stored as MP4 files in videos/chunk-*/observation.images.*/episode_*.mp4
    - Metadata in meta/episodes.jsonl
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
        chunk_id: str = "chunk-000",  # Which chunk to load
    ):
        """
        Initialize Lerobot v2.1 dataset.
        
        Args:
            dataset_path: Path to Lerobot dataset root directory or HuggingFace dataset name (e.g., "lerobot/svla_so100_pickplace")
            episode_type: Type of episodes to load
            image_keys: List of image observation keys (e.g., ["observation.images.main", "observation.images.secondary_0"])
            text_key: Key for text instructions/commands
            action_key: Key for actions
            reward_key: Key for rewards
            done_key: Key for done flags
            advantage_key: Key for precomputed advantages (optional)
            max_episode_length: Maximum episode length to use
            transform: Optional transform to apply to images
            chunk_id: Chunk identifier (default: "chunk-000")
        """
        # Check required dependencies first
        try:
            import pandas as pd
            self.pd = pd
        except ImportError:
            raise ImportError("Please install pandas: pip install pandas pyarrow")
        
        # Check for huggingface_hub (for dataset downloading)
        try:
            from huggingface_hub import snapshot_download
            self.snapshot_download = snapshot_download
        except ImportError:
            self.snapshot_download = None
        
        # Check if dataset_path is a HuggingFace dataset and download if needed
        dataset_path_str = str(dataset_path)
        dataset_path_obj = Path(dataset_path)
        
        # Detect HuggingFace dataset: contains '/' and doesn't exist as local path
        is_hf_dataset = (
            '/' in dataset_path_str and 
            not dataset_path_obj.exists() and
            not dataset_path_obj.is_absolute() and  # Not an absolute path
            len(dataset_path_str.split('/')) == 2  # Format: "org/dataset_name"
        )
        
        if is_hf_dataset:
            # Likely a HuggingFace dataset (e.g., "lerobot/svla_so100_pickplace")
            self.dataset_path = self._download_huggingface_dataset(dataset_path_str)
        else:
            self.dataset_path = dataset_path_obj
            if not self.dataset_path.exists():
                raise ValueError(
                    f"Dataset path does not exist: {dataset_path}\n"
                    f"If this is a HuggingFace dataset, make sure huggingface_hub is installed: "
                    f"pip install huggingface_hub"
                )
        
        self.episode_type = episode_type
        self.image_keys = image_keys or ["observation.images.main"]
        self.text_key = text_key
        self.action_key = action_key
        self.reward_key = reward_key
        self.done_key = done_key
        self.advantage_key = advantage_key
        self.max_episode_length = max_episode_length
        self.transform = transform
        self.chunk_id = chunk_id
        
        try:
            import decord
            self.decord = decord
            try:
                self.decord.bridge.set_bridge("torch")
            except:
                # If bridge setting fails, continue without it
                pass
            self.use_decord = True
        except ImportError:
            try:
                import torchvision.io as tvio
                self.torchvision_io = tvio
                self.use_decord = False
            except ImportError:
                raise ImportError("Please install decord for video loading: pip install decord")
            else:
                self.use_decord = False
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Discover episode files
        self.episode_files = self._discover_episodes()
        
        # Filter by episode type
        if episode_type != "all":
            self.episode_files = [
                ep_file for ep_file in self.episode_files
                if self._get_episode_type(ep_file) == episode_type
            ]
        
        if len(self.episode_files) == 0:
            raise ValueError(f"No episodes found for type '{episode_type}' in {dataset_path}")
    
    def _download_huggingface_dataset(self, dataset_name: str) -> Path:
        """
        Download a HuggingFace dataset and return the local cache path.
        
        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "lerobot/svla_so100_pickplace")
        
        Returns:
            Path to the downloaded dataset directory
        """
        if self.snapshot_download is None:
            raise ImportError(
                "huggingface_hub is required for downloading HuggingFace datasets. "
                "Install it with: pip install huggingface_hub"
            )
        
        print(f"Downloading HuggingFace dataset: {dataset_name}")
        print("This may take a while depending on the dataset size...")
        
        try:
            # Download the dataset to cache
            # By default, snapshot_download uses ~/.cache/huggingface/hub
            cache_dir = self.snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_files_only=False,  # Allow downloading
            )
            
            cache_path = Path(cache_dir)
            print(f"Dataset downloaded to: {cache_path}")
            
            # Verify the dataset structure
            if not (cache_path / "data").exists():
                raise ValueError(
                    f"Downloaded dataset does not have expected structure. "
                    f"Expected 'data' directory not found in {cache_path}"
                )
            
            return cache_path
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to download HuggingFace dataset '{dataset_name}': {e}\n"
                f"Make sure you have access to the dataset and have installed huggingface_hub."
            )
    
    def _load_metadata(self) -> Dict:
        """Load metadata from meta/episodes.jsonl."""
        metadata_path = self.dataset_path / "meta" / "episodes.jsonl"
        metadata = {}
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.strip():
                        ep_meta = json.loads(line)
                        # Use episode index or id as key
                        key = ep_meta.get("episode_index") or ep_meta.get("episode_id") or ep_meta.get("index")
                        if key is not None:
                            metadata[str(key)] = ep_meta
        
        return metadata
    
    def _discover_episodes(self) -> List[Path]:
        """Discover all episode parquet files in data/chunk-*/."""
        data_dir = self.dataset_path / "data" / self.chunk_id
        
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Find all episode parquet files - support both naming conventions
        episode_files = sorted(data_dir.glob("episode_*.parquet"))
        
        # If no episode_*.parquet files found, try file-*.parquet (Lerobot v2.1 format)
        if len(episode_files) == 0:
            episode_files = sorted(data_dir.glob("file-*.parquet"))
        
        if len(episode_files) == 0:
            raise ValueError(
                f"No episode files found in {data_dir}. "
                f"Expected files matching 'episode_*.parquet' or 'file-*.parquet'"
            )
        
        return episode_files
    
    def _get_episode_type(self, episode_file: Path) -> str:
        """Get episode type from metadata or filename."""
        # Extract episode index from filename
        # Support both episode_*.parquet and file-*.parquet naming conventions
        match = re.search(r'(?:episode_|file-)(\d+)', episode_file.name)
        if match:
            ep_idx = match.group(1)
            if ep_idx in self.metadata:
                return self.metadata[ep_idx].get("episode_type", "all")
        
        return "all"
    
    def _load_parquet_episode(self, episode_file: Path) -> Dict:
        """Load episode data from parquet file."""
        df = self.pd.read_parquet(episode_file)
        
        # Limit episode length if specified
        if self.max_episode_length and len(df) > self.max_episode_length:
            df = df.head(self.max_episode_length)
        
        return df
    
    def _load_video_frames(self, video_path: Path, num_frames: int) -> torch.Tensor:
        """Load frames from MP4 video file."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if self.use_decord:
            # Using decord for faster video loading
            vr = self.decord.VideoReader(str(video_path), ctx=self.decord.cpu(0))
            total_frames = len(vr)
            
            # Sample frames evenly
            if num_frames > total_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                frame = vr[idx].asnumpy()  # Convert to numpy
                frames.append(frame)
            
            # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        else:
            # Fallback to torchvision VideoReader
            video, audio, info = self.torchvision_io.read_video(str(video_path), output_format="TCHW")
            
            # Sample frames
            total_frames = video.shape[0]
            if num_frames > total_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = video[frame_indices].float() / 255.0
        
        return frames
    
    def _find_video_path(self, episode_idx: str, image_key: str) -> Optional[Path]:
        """Find video file path for given episode and image key."""
        # Convert image key to video directory name
        # e.g., "observation.images.main" -> "observation.images.main"
        video_dir = self.dataset_path / "videos" / self.chunk_id / image_key
        
        if not video_dir.exists():
            return None
        
        # Try episode_*.mp4 naming convention first
        video_file = video_dir / f"episode_{episode_idx}.mp4"
        if video_file.exists():
            return video_file
        
        # Try file-*.mp4 naming convention (Lerobot v2.1 format)
        video_file = video_dir / f"file-{episode_idx}.mp4"
        if video_file.exists():
            return video_file
        
        # Try matching any video file with the episode index
        # Some datasets might have different naming patterns
        for video_file in video_dir.glob(f"*{episode_idx}*.mp4"):
            return video_file
        
        return None
    
    def __len__(self):
        return len(self.episode_files)
    
    def __getitem__(self, idx):
        """Get a single episode."""
        episode_file = self.episode_files[idx]
        
        # Extract episode index from filename - support both naming conventions
        match = re.search(r'(?:episode_|file-)(\d+)', episode_file.name)
        episode_idx = match.group(1) if match else str(idx).zfill(6)
        
        # Load episode data from parquet
        df = self._load_parquet_episode(episode_file)
        num_steps = len(df)
        
        # Load images from videos
        images = {}
        for image_key in self.image_keys:
            video_path = self._find_video_path(episode_idx, image_key)
            if video_path:
                video_frames = self._load_video_frames(video_path, num_steps)
                if self.transform:
                    video_frames = self.transform(video_frames)
                images[image_key] = video_frames
            else:
                # Try loading from parquet if video not found
                if image_key in df.columns:
                    # If images are stored in parquet (less common)
                    img_data = df[image_key].values
                    if len(img_data) > 0 and img_data[0] is not None:
                        # Convert PIL/numpy images to tensor
                        import torchvision.transforms as T
                        to_tensor = T.ToTensor()
                        frames = []
                        for img in img_data:
                            if hasattr(img, 'numpy'):
                                img = img.numpy()
                            frames.append(to_tensor(img))
                        images[image_key] = torch.stack(frames)
        
        # Get text instructions
        text = None
        if self.text_key in df.columns:
            text_values = df[self.text_key].values
            # Use first non-null text if available
            for txt in text_values:
                if txt is not None and str(txt).strip():
                    text = str(txt)
                    break
        
        # Get actions
        actions = None
        if self.action_key in df.columns:
            action_data = df[self.action_key].values
            if len(action_data) > 0:
                # Handle nested lists/arrays
                if isinstance(action_data[0], (list, np.ndarray)):
                    actions = torch.from_numpy(np.array([np.array(a) for a in action_data])).float()
                else:
                    actions = torch.from_numpy(np.array(action_data)).float()
        
        # Get rewards
        rewards = None
        if self.reward_key in df.columns:
            reward_data = df[self.reward_key].values
            rewards = torch.from_numpy(np.array(reward_data)).float()
        
        # Get done flags
        done = None
        if self.done_key in df.columns:
            done_data = df[self.done_key].values
            done = torch.from_numpy(np.array(done_data)).bool()
        
        # Get advantages if available
        advantages = None
        if self.advantage_key and self.advantage_key in df.columns:
            adv_data = df[self.advantage_key].values
            advantages = torch.from_numpy(np.array(adv_data)).float()
        
        # Get episode metadata
        metadata = {
            "episode_index": idx,
            "episode_id": episode_idx,
            "episode_length": num_steps,
        }
        
        # Try to find metadata with flexible key matching
        # Try exact match first, then try without leading zeros, then as integer
        ep_meta = None
        if episode_idx in self.metadata:
            ep_meta = self.metadata[episode_idx]
        else:
            # Try without leading zeros
            episode_idx_int = str(int(episode_idx)) if episode_idx.isdigit() else episode_idx
            if episode_idx_int in self.metadata:
                ep_meta = self.metadata[episode_idx_int]
        
        if ep_meta:
            metadata["episode_type"] = ep_meta.get("episode_type", "all")
            metadata.update(ep_meta)
        
        return {
            "images": images,
            "text": text,
            "actions": actions,
            "rewards": rewards,
            "done": done,
            "advantages": advantages,
            "metadata": metadata,
        }
    
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
    
    def get_batch(self, indices: List[int]) -> Dict:
        """Get a batch of episodes."""
        batch = [self[i] for i in indices]
        
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


def create_dataloader(
    dataset_path: Union[str, Path],
    episode_type: str = "all",
    batch_size: int = 1,  # Usually 1 for episodes, but can batch multiple episodes
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """Create a DataLoader for Lerobot v2.1 dataset."""
    dataset = LerobotDatasetV21(dataset_path, episode_type=episode_type, **dataset_kwargs)
    
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
