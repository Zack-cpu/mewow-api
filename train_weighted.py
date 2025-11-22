"""
Training with class weights to focus on struggling classes
"""
import torch
import torch.nn as nn
from train import main as original_main
from train import Trainer
import config


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for struggling classes"""
    
    def __init__(self, model, train_loader, val_loader, processor, device, class_weights=None):
        super().__init__(model, train_loader, val_loader, processor, device)
        
        # Override criterion with weighted loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted loss: {class_weights.cpu().numpy()}")


def main():
    """Main training with class weights"""
    
    # Define class weights (higher = more important)
    # Classes: Angry, Defence, Fighting, Happy, HuntingMind, Mating, MotherCall, Paining, Resting, Warning
    class_weights = [
        1.0,  # Angry      - OK (85% recall)
        1.0,  # Defence    - Excellent (100% recall)
        1.0,  # Fighting   - OK (92% recall)
        1.0,  # Happy      - Good (92% recall)
        1.0,  # HuntingMind- Good (94% recall)
        5.0,  # Mating     - STRUGGLING (5% recall) ← 5x weight!
        1.0,  # MotherCall - Good (95% recall)
        5.0,  # Paining    - STRUGGLING (14% recall) ← 5x weight!
        1.0,  # Resting    - Excellent (99% recall)
        1.0,  # Warning    - Good (94% recall)
    ]
    
    print("="*60)
    print("TRAINING WITH CLASS WEIGHTS")
    print("="*60)
    print("Focusing on struggling classes:")
    print(f"  Mating: 5x weight")
    print(f"  Paining: 5x weight")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import here to avoid circular import
    from model import create_model, load_clap_processor
    from data_loader import create_data_loaders
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # Load processor
    print("Loading CLAP processor...")
    processor = load_clap_processor()
    
    # Create model
    print("Creating model...")
    freeze_encoder = True  # Start with frozen
    model = create_model(num_classes=config.NUM_CLASSES, freeze_encoder=freeze_encoder)
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create weighted trainer
    trainer = WeightedTrainer(
        model, train_loader, val_loader, processor, device,
        class_weights=class_weights
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

