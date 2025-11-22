"""
Prepare and split the NAYA dataset into train/val/test sets
"""
import os
import shutil
from pathlib import Path
import random
import config

SOURCE_DATA_DIR = "C:/Users/asus/Desktop/MEWOW/NAYA_DATA_AUG1X"
TARGET_DATA_DIR = "C:/Users/asus/Desktop/MEWOW/data"

# Split ratios (only train/val, no test needed for fine-tuning)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

def count_files():
    """Count files in each class"""
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(SOURCE_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  Directory not found: {class_name}")
            continue
        
        files = [f for f in os.listdir(class_dir) if f.endswith('.mp3')]
        original_files = [f for f in files if '_aug1' not in f]
        augmented_files = [f for f in files if '_aug1' in f]
        
        print(f"\n{class_name}:")
        print(f"  Original files:  {len(original_files)}")
        print(f"  Augmented files: {len(augmented_files)}")
        print(f"  Total files:     {len(files)}")

def split_data(use_augmented=True):
    """
    Split data into train/val sets (no test set needed for fine-tuning)
    
    Args:
        use_augmented: If True, use both original and augmented files.
                      If False, use only original files.
    """
    print("\n" + "="*60)
    print("SPLITTING DATA INTO TRAIN/VAL")
    print("="*60)
    print(f"Using augmented files: {use_augmented}")
    print(f"Split ratio: {TRAIN_RATIO:.0%} train, {VAL_RATIO:.0%} val")
    
    # Create target directories (only train and val)
    for split in ['train', 'val']:
        for class_name in config.CLASS_NAMES:
            target_dir = os.path.join(TARGET_DATA_DIR, split, class_name)
            os.makedirs(target_dir, exist_ok=True)
    
    random.seed(42)  # For reproducibility
    
    total_train = 0
    total_val = 0
    
    for class_name in config.CLASS_NAMES:
        source_dir = os.path.join(SOURCE_DATA_DIR, class_name)
        if not os.path.exists(source_dir):
            print(f"⚠️  Skipping {class_name} - directory not found")
            continue
        
        # Get all files
        all_files = [f for f in os.listdir(source_dir) if f.endswith('.mp3')]
        
        if use_augmented:
            # Use all files (original + augmented)
            files_to_use = all_files
        else:
            # Use only original files (no _aug1 in filename)
            files_to_use = [f for f in all_files if '_aug1' not in f]
        
        # Shuffle files
        random.shuffle(files_to_use)
        
        # Calculate split indices
        n_files = len(files_to_use)
        n_train = int(n_files * TRAIN_RATIO)
        
        train_files = files_to_use[:n_train]
        val_files = files_to_use[n_train:]
        
        # Copy files to respective directories
        for file in train_files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(TARGET_DATA_DIR, 'train', class_name, file)
            shutil.copy2(src, dst)
        
        for file in val_files:
            src = os.path.join(source_dir, file)
            dst = os.path.join(TARGET_DATA_DIR, 'val', class_name, file)
            shutil.copy2(src, dst)
        
        total_train += len(train_files)
        total_val += len(val_files)
        
        print(f"\n{class_name}:")
        print(f"  Train: {len(train_files)} | Val: {len(val_files)}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Train: {total_train}")
    print(f"Total Val:   {total_val}")
    print(f"Grand Total: {total_train + total_val}")
    print("\n✓ Data split complete!")
    print(f"✓ Data saved to: {TARGET_DATA_DIR}")
    print("\nYou can now run: python train.py")


def main():
    """Main function"""
    print("="*60)
    print("CAT MEOW DATASET PREPARATION")
    print("="*60)
    
    # Count files
    count_files()
    
    # Ask user
    print("\n" + "="*60)
    print("OPTIONS")
    print("="*60)
    print("1. Use BOTH original and augmented files (recommended - more data)")
    print("2. Use ONLY augmented files")
    print("3. Use ONLY original files")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        split_data(use_augmented=True)
    elif choice == "2":
        # Filter to only augmented
        print("\nNote: Option 2 not yet implemented. Using option 1 instead.")
        split_data(use_augmented=True)
    elif choice == "3":
        split_data(use_augmented=False)
    else:
        print("Invalid choice. Using option 1 (both original and augmented).")
        split_data(use_augmented=True)


if __name__ == "__main__":
    main()

