"""
Training script for Cat Meow Classifier
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from model import create_model, load_clap_processor
from data_loader import create_data_loaders


class Trainer:
    """Training class for Cat Meow Classifier"""
    
    def __init__(self, model, train_loader, val_loader, processor, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-7
        )
        
        # TensorBoard writer
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        self.writer = SummaryWriter(config.LOGS_DIR)
        
        # Best validation accuracy
        self.best_val_acc = 0.0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for batch_idx, (audio, labels) in enumerate(pbar):
            # Process audio with CLAP processor (handles variable-length audio)
            # audio is a list of tensors with different lengths
            inputs = self.processor(
                audios=[a.numpy() for a in audio],  # Convert list of tensors to list of numpy arrays
                sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Accuracy', 100.*correct/total, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
            
            for audio, labels in pbar:
                # Process audio (handles variable-length audio)
                inputs = self.processor(
                    audios=[a.numpy() for a in audio],  # Convert list of tensors to list of numpy arrays
                    sampling_rate=config.SAMPLE_RATE,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        # Generate classification report
        if epoch == config.NUM_EPOCHS - 1:
            report = classification_report(
                all_labels, all_preds,
                target_names=config.CLASS_NAMES,
                digits=4
            )
            print("\nClassification Report:")
            print(report)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(all_labels, all_preds, epoch)
        
        return avg_loss, accuracy
    
    def plot_confusion_matrix(self, true_labels, pred_labels, epoch):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.OUTPUT_DIR, f'confusion_matrix_epoch_{epoch+1}.png'))
        plt.close()
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Save model checkpoint"""
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': {
                'num_classes': config.NUM_CLASSES,
                'class_names': config.CLASS_NAMES,
                'model_name': config.MODEL_NAME
            }
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best=is_best)
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        self.writer.close()


def main():
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Create model (start with frozen encoder for faster initial training)
    print("Creating model...")
    freeze_encoder = True  # Set to False for full fine-tuning
    model = create_model(num_classes=config.NUM_CLASSES, freeze_encoder=freeze_encoder)
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, processor, device)
    trainer.train()


if __name__ == "__main__":
    main()

