"""
Cat Meow Classifier model based on CLAP
"""
import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor
import config


class CatMeowClassifier(nn.Module):
    """
    Cat meow classifier using pretrained CLAP model
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, freeze_encoder: bool = False):
        """
        Args:
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze the CLAP encoder weights
        """
        super().__init__()
        
        self.freeze_encoder = freeze_encoder
        
        # Load pretrained CLAP model
        self.clap = ClapModel.from_pretrained(config.MODEL_NAME)
        
        # Freeze encoder if specified
        if freeze_encoder:
            # Freeze the entire CLAP model (both audio and text encoders)
            for param in self.clap.parameters():
                param.requires_grad = False
            # Set to eval mode to fix BatchNorm issues
            self.clap.eval()
        
        # Get the audio embedding dimension
        self.audio_embed_dim = self.clap.config.projection_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.audio_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def train(self, mode: bool = True):
        """
        Override train() to handle BatchNorm properly
        """
        super().train(mode)
        if self.freeze_encoder:
            # Keep entire CLAP in eval mode when frozen
            self.clap.eval()
        else:
            # When unfrozen for fine-tuning, keep BatchNorm in eval mode
            # This uses pretrained BatchNorm statistics (common practice)
            for module in self.clap.modules():
                if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        return self
        
    def forward(self, audio_input):
        """
        Forward pass
        
        Args:
            audio_input: Processed features dict from ClapProcessor
            
        Returns:
            logits: Classification logits
        """
        # Get audio embeddings from CLAP
        # Pass the full dict to get_audio_features
        audio_embeds = self.clap.get_audio_features(**audio_input)
        
        # Normalize embeddings
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
        
        # Classification
        logits = self.classifier(audio_embeds)
        
        return logits
    
    def get_embeddings(self, audio_input):
        """
        Get audio embeddings without classification
        
        Args:
            audio_input: Processed features dict from ClapProcessor
            
        Returns:
            embeddings: Audio embeddings
        """
        with torch.no_grad():
            audio_embeds = self.clap.get_audio_features(**audio_input)
            audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
        return audio_embeds


def load_clap_processor():
    """Load CLAP processor for audio preprocessing"""
    processor = ClapProcessor.from_pretrained(config.MODEL_NAME)
    return processor


def create_model(num_classes: int = config.NUM_CLASSES, freeze_encoder: bool = False):
    """
    Create and return a new model instance
    
    Args:
        num_classes: Number of output classes
        freeze_encoder: Whether to freeze encoder weights
        
    Returns:
        model: CatMeowClassifier instance
    """
    model = CatMeowClassifier(num_classes=num_classes, freeze_encoder=freeze_encoder)
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

