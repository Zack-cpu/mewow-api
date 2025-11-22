"""
Resume training with class weights to focus on struggling classes
"""
import torch
import torch.nn as nn
import os
from train import Trainer
from model import create_model, load_clap_processor
from data_loader import create_data_loaders
import config


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss"""
    
    def __init__(self, model, train_loader, val_loader, processor, device, class_weights=None):
        super().__init__(model, train_loader, val_loader, processor, device)
        
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"\nUsing weighted loss for struggling classes:")
            for i, (name, weight) in enumerate(zip(config.CLASS_NAMES, class_weights)):
                if weight > 1.0:
                    print(f"  {name}: {weight:.1f}x weight")


def resume_with_weights(checkpoint_path, additional_epochs=10, class_weights=None):
    """Resume training with class weights"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Previous training:")
    print(f"  - Completed epochs: {checkpoint['epoch'] + 1}")
    print(f"  - Best accuracy: {checkpoint['accuracy']:.2f}%")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # Load processor
    print("Loading CLAP processor...")
    processor = load_clap_processor()
    
    # Create model (unfrozen for fine-tuning)
    print("Creating model...")
    model = create_model(num_classes=config.NUM_CLASSES, freeze_encoder=False)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create weighted trainer
    trainer = WeightedTrainer(
        model, train_loader, val_loader, processor, device,
        class_weights=class_weights
    )
    
    # Restore optimizer state
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Set starting values
    start_epoch = checkpoint['epoch'] + 1
    trainer.best_val_acc = checkpoint['accuracy']
    
    print(f"\nResuming training from epoch {start_epoch + 1}...")
    print(f"Training for {additional_epochs} more epochs")
    
    # Update config
    original_epochs = config.NUM_EPOCHS
    config.NUM_EPOCHS = start_epoch + additional_epochs
    
    # Train
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc = trainer.validate(epoch)
        
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        is_best = val_acc > trainer.best_val_acc
        if is_best:
            trainer.best_val_acc = val_acc
        
        trainer.save_checkpoint(epoch, val_acc, is_best=is_best)
    
    print(f"\nTraining completed! Best validation accuracy: {trainer.best_val_acc:.2f}%")
    trainer.writer.close()
    
    config.NUM_EPOCHS = original_epochs


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume with class weights')
    parser.add_argument('--checkpoint', type=str,
                       default=os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'),
                       help='Path to checkpoint')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of additional epochs')
    
    args = parser.parse_args()
    
    # Define class weights
    # Focus heavily on Mating and Paining (struggling classes)
    class_weights = [
        1.0,  # Angry
        1.0,  # Defence
        1.0,  # Fighting
        1.0,  # Happy
        1.0,  # HuntingMind
        5.0,  # Mating ← 5x weight!
        1.0,  # MotherCall
        5.0,  # Paining ← 5x weight!
        1.0,  # Resting
        1.0,  # Warning
    ]
    
    print("="*60)
    print("RESUME WITH CLASS WEIGHTS")
    print("="*60)
    print("Focusing on struggling classes")
    
    resume_with_weights(
        checkpoint_path=args.checkpoint,
        additional_epochs=args.epochs,
        class_weights=class_weights
    )


if __name__ == "__main__":
    main()

