"""
Inference script for testing the trained model
"""
import torch
import torchaudio
import numpy as np
import os
from typing import List, Tuple

import config
from model import create_model, load_clap_processor


class CatMeowInference:
    """Inference class for cat meow classification"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        # Load model and processor
        self.model = self._load_model()
        self.processor = load_clap_processor()
        
    def _load_model(self):
        """Load model from checkpoint"""
        print(f"Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        model = create_model(num_classes=config.NUM_CLASSES)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            processed_audio: Preprocessed audio tensor
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=config.SAMPLE_RATE
            )
            waveform = resampler(waveform)
        
        # Trim or pad to max length
        max_samples = config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform.squeeze(0)
    
    def predict(self, audio_path: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Make prediction on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            predicted_class: Predicted class name
            confidence: Confidence score
            all_probabilities: List of (class_name, probability) tuples
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        
        # Process with CLAP processor
        inputs = self.processor(
            audios=audio.numpy(),
            sampling_rate=config.SAMPLE_RATE,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, dim=-1)
        predicted_class = config.CLASS_NAMES[predicted_idx.item()]
        confidence = confidence.item()
        
        # Get all probabilities
        all_probs = [(config.CLASS_NAMES[i], prob.item()) 
                     for i, prob in enumerate(probabilities[0])]
        all_probs.sort(key=lambda x: x[1], reverse=True)
        
        return predicted_class, confidence, all_probs
    
    def batch_predict(self, audio_paths: List[str]) -> List[dict]:
        """
        Make predictions on multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            predictions: List of prediction dictionaries
        """
        results = []
        
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                continue
            
            pred_class, confidence, all_probs = self.predict(audio_path)
            
            results.append({
                'file': audio_path,
                'predicted_class': pred_class,
                'confidence': confidence,
                'all_probabilities': all_probs
            })
        
        return results


def main():
    """Test inference on sample audio files"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cat Meow Classifier Inference')
    parser.add_argument('audio_file', type=str, help='Path to audio file')
    parser.add_argument('--checkpoint', type=str, 
                       default=os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'),
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first using train.py")
        return
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    # Create inference object
    print("Loading model...")
    inference = CatMeowInference(args.checkpoint, args.device)
    
    # Make prediction
    print(f"\nAnalyzing: {args.audio_file}")
    print("-" * 60)
    
    predicted_class, confidence, all_probs = inference.predict(args.audio_file)
    
    print(f"\n✓ Predicted Class: {predicted_class}")
    print(f"✓ Confidence: {confidence*100:.2f}%")
    print("\nAll Probabilities:")
    for class_name, prob in all_probs:
        bar = '█' * int(prob * 50)
        print(f"  {class_name:12s}: {prob*100:5.2f}% {bar}")


if __name__ == "__main__":
    main()

