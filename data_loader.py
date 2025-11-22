"""
Data loading and preprocessing for cat meow audio files
"""
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List
import config


class CatMeowDataset(Dataset):
    """Dataset for loading cat meow audio files"""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Directory containing subdirectories for each class
            transform: Optional transform to apply to audio
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}
        
        # Load all audio file paths and labels
        self._load_samples()
        
    def _load_samples(self):
        """Load all audio file paths from directory structure"""
        for class_name in config.CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory not found: {class_dir}")
                continue
                
            for audio_file in class_dir.glob("*.wav"):
                self.samples.append({
                    'path': str(audio_file),
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
            
            # Also support mp3 and m4a formats
            for ext in ['*.mp3', '*.m4a', '*.flac']:
                for audio_file in class_dir.glob(ext):
                    self.samples.append({
                        'path': str(audio_file),
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
        
        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and process audio file (keep original length - no padding)
        
        Returns:
            audio: Processed audio tensor (variable length)
            label: Class label
        """
        sample = self.samples[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(sample['path'])
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=config.SAMPLE_RATE
            )
            waveform = resampler(waveform)
        
        # Keep original length - no padding! Let ClapProcessor handle it
        
        # Apply transforms if any
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform.squeeze(0), sample['label']


def collate_variable_length(batch):
    """
    Custom collate function for variable-length audio
    Returns list of audio tensors instead of stacked tensor
    """
    audios = [item[0] for item in batch]  # Keep as list (variable lengths)
    labels = torch.tensor([item[1] for item in batch])
    return audios, labels


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0  
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = CatMeowDataset(train_dir)
    val_dataset = CatMeowDataset(val_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to avoid BatchNorm issues
        collate_fn=collate_variable_length  # Handle variable-length audio
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to avoid BatchNorm issues
        collate_fn=collate_variable_length  # Handle variable-length audio
    )
    
    return train_loader, val_loader


def create_sample_data_structure():
    """
    Create sample directory structure for organizing data
    This helps users understand how to organize their audio files
    """
    dirs_to_create = [
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR
    ]
    
    for base_dir in dirs_to_create:
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    print("Created data directory structure:")
    print(f"  {config.DATA_DIR}/")
    print("    ├── train/")
    print("    ├── val/")
    print("    └── test/")
    for class_name in config.CLASS_NAMES:
        print(f"        ├── {class_name}/")
    print("\nPlace your .wav, .mp3, or .m4a files in the appropriate directories.")


if __name__ == "__main__":
    # Create sample directory structure
    create_sample_data_structure()

