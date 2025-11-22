"""
Resume training from a checkpoint
"""
import torch
import os
from train import Trainer
from model import create_model, load_clap_processor
from data_loader import create_data_loaders
import config


def resume_training(checkpoint_path, additional_epochs=5, freeze_encoder=True):
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        additional_epochs: Number of additional epochs to train
        freeze_encoder: Keep encoder frozen or unfreeze for fine-tuning
    """
    # Set device
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
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes=config.NUM_CLASSES, freeze_encoder=freeze_encoder)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, processor, device)
    
    # Restore optimizer and scheduler states
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Set starting epoch and best accuracy
    start_epoch = checkpoint['epoch'] + 1
    trainer.best_val_acc = checkpoint['accuracy']
    
    print(f"\nResuming training from epoch {start_epoch + 1}...")
    print(f"Training for {additional_epochs} more epochs")
    
    # Update config for additional epochs
    original_epochs = config.NUM_EPOCHS
    config.NUM_EPOCHS = start_epoch + additional_epochs
    
    # Train for additional epochs
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = trainer.train_epoch(epoch)
        
        # Validate
        val_loss, val_acc = trainer.validate(epoch)
        
        # Update learning rate
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save checkpoint
        is_best = val_acc > trainer.best_val_acc
        if is_best:
            trainer.best_val_acc = val_acc
        
        trainer.save_checkpoint(epoch, val_acc, is_best=is_best)
    
    print(f"\nTraining completed! Best validation accuracy: {trainer.best_val_acc:.2f}%")
    trainer.writer.close()
    
    # Restore original config
    config.NUM_EPOCHS = original_epochs


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str,
                       default=os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'),
                       help='Path to checkpoint')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of additional epochs to train')
    parser.add_argument('--unfreeze', action='store_true',
                       help='Unfreeze encoder for full fine-tuning')
    
    args = parser.parse_args()
    
    freeze_encoder = not args.unfreeze
    
    print("="*60)
    print("RESUME TRAINING")
    print("="*60)
    print(f"Mode: {'Frozen Encoder' if freeze_encoder else 'Full Fine-='}")
    
    resume_training(
        checkpoint_path=args.checkpoint,
        additional_epochs=args.epochs,
        freeze_encoder=freeze_encoder
    )


if __name__ == "__main__":
    main()

