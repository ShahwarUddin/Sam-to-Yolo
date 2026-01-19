#!/usr/bin/env python3
"""
YOLO Training Pipeline for SAM3 Annotated Data

This script continues the work after ANNOTATION_PIPELINE.py by:
1. Taking annotated data from training_data folder
2. Splitting it into train/test sets
3. Creating YAML configuration file
4. Training a YOLOv8n model

Usage:
    python train_yolo_model.py

Requirements:
    pip install ultralytics

Author: Generated for SAM3 pipeline
"""

import os
import shutil
import random
from pathlib import Path
import yaml
from datetime import datetime
import argparse

def setup_directories(base_dir="yolo_training", train_ratio=0.8):
    """
    Create directory structure for YOLO training
    
    Args:
        base_dir: Base directory for YOLO training setup
        train_ratio: Ratio of data to use for training (rest goes to test)
    
    Returns:
        Dictionary with directory paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = f"{base_dir}_{timestamp}"
    
    dirs = {
        'base': training_dir,
        'train_images': os.path.join(training_dir, 'train', 'images'),
        'train_labels': os.path.join(training_dir, 'train', 'labels'),
        'test_images': os.path.join(training_dir, 'test', 'images'),
        'test_labels': os.path.join(training_dir, 'test', 'labels'),
        'val_images': os.path.join(training_dir, 'val', 'images'),
        'val_labels': os.path.join(training_dir, 'val', 'labels'),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return dirs

def read_classes_file(classes_file="training_data/classes.txt"):
    """Read class names from classes.txt file"""
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(classes)} classes: {classes}")
    return classes

def split_dataset(source_images_dir="training_data/images", 
                 source_labels_dir="training_data/labels",
                 target_dirs=None, 
                 train_ratio=0.8, 
                 val_ratio=0.1,
                 seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_images_dir: Source directory with images
        source_labels_dir: Source directory with labels
        target_dirs: Dictionary with target directory paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (test gets the rest)
        seed: Random seed for reproducible splits
    """
    random.seed(seed)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(source_images_dir).glob(ext))
    
    # Filter to only include images that have corresponding label files
    valid_pairs = []
    for img_path in image_files:
        label_path = Path(source_labels_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid image-label pairs found!")
    
    # Shuffle the pairs
    random.shuffle(valid_pairs)
    
    # Calculate split indices
    total = len(valid_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    
    # Copy files to respective directories
    def copy_pairs(pairs, img_dir, label_dir, split_name):
        for img_path, label_path in pairs:
            # Copy image
            shutil.copy2(img_path, img_dir)
            # Copy label
            shutil.copy2(label_path, label_dir)
        print(f"Copied {len(pairs)} {split_name} pairs")
    
    copy_pairs(train_pairs, target_dirs['train_images'], target_dirs['train_labels'], "train")
    copy_pairs(val_pairs, target_dirs['val_images'], target_dirs['val_labels'], "val")
    copy_pairs(test_pairs, target_dirs['test_images'], target_dirs['test_labels'], "test")
    
    return {
        'train': len(train_pairs),
        'val': len(val_pairs), 
        'test': len(test_pairs),
        'total': total
    }

def create_yaml_config(target_dirs, classes, yaml_path=None):
    """
    Create YAML configuration file for YOLO training
    
    Args:
        target_dirs: Dictionary with directory paths
        classes: List of class names
        yaml_path: Path to save YAML file (auto-generated if None)
    
    Returns:
        Path to created YAML file
    """
    if yaml_path is None:
        yaml_path = os.path.join(target_dirs['base'], 'dataset.yaml')
    
    # Create YAML configuration
    config = {
        'path': os.path.abspath(target_dirs['base']),  # Absolute path to dataset root
        'train': 'train/images',  # Relative path from dataset root
        'val': 'val/images',      # Relative path from dataset root
        'test': 'test/images',    # Relative path from dataset root
        'nc': len(classes),       # Number of classes
        'names': classes          # Class names
    }
    
    # Save YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created YAML config: {yaml_path}")
    print("YAML Contents:")
    print("-" * 40)
    with open(yaml_path, 'r') as f:
        print(f.read())
    print("-" * 40)
    
    return yaml_path

def train_yolo_model(yaml_config_path, 
                    model_name='yolov8n.pt',
                    epochs=100,
                    img_size=640,
                    batch_size=16,
                    device='auto',
                    save_dir=None):
    """
    Train YOLOv8 model using the prepared dataset
    
    Args:
        yaml_config_path: Path to YAML configuration file
        model_name: Pre-trained model to start from
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Training batch size
        device: Device to use ('auto', 'cpu', 'cuda', etc.)
        save_dir: Directory to save training results
    """
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except ImportError:
        print("❌ Error: ultralytics not installed!")
        print("Please install it with: pip install ultralytics")
        return None
    
    # Load pretrained model
    print(f"Loading pre-trained model: {model_name}")
    model = YOLO(model_name)
    
    # Set up training arguments
    train_args = {
        'data': yaml_config_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'verbose': True,
        'save': True,
        'plots': True,
    }
    
    if save_dir:
        train_args['project'] = save_dir
    
    print(f"Starting training with parameters:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\n" + "="*50)
    print("🚀 STARTING YOLO TRAINING")
    print("="*50)
    
    try:
        results = model.train(**train_args)
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        return results
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        return None

def validate_training_data(source_dir="training_data"):
    """
    Validate the training data before processing
    """
    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "labels") 
    classes_file = os.path.join(source_dir, "classes.txt")
    
    print("🔍 Validating training data...")
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    
    # Count files
    image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    
    print(f"✓ Found {image_count} images")
    print(f"✓ Found {label_count} label files")
    
    if image_count == 0:
        raise ValueError("No images found in training data!")
    if label_count == 0:
        raise ValueError("No label files found in training data!")
    
    # Sample validation - check a few image-label pairs
    sample_size = min(5, image_count)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:sample_size]
    
    valid_pairs = 0
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            valid_pairs += 1
    
    print(f"✓ Validated {valid_pairs}/{sample_size} sample image-label pairs")
    
    if valid_pairs == 0:
        raise ValueError("No matching image-label pairs found!")
    
    return True

def main():
    """Main function to orchestrate the YOLO training pipeline"""
    
    parser = argparse.ArgumentParser(description='Train YOLO model on SAM3 annotated data')
    parser.add_argument('--source-dir', default='training_data', help='Source directory with annotated data')
    parser.add_argument('--output-dir', default='yolo_training', help='Output directory for YOLO training setup')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Ratio of data for validation (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size (default: 16)')
    parser.add_argument('--model', default='yolov8n.pt', help='Pre-trained model to use (default: yolov8n.pt)')
    parser.add_argument('--device', default='auto', help='Device to use for training (default: auto)')
    parser.add_argument('--no-train', action='store_true', help='Only prepare data, do not train model')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 SAM3 → YOLO TRAINING PIPELINE")
    print("="*60)
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train/Val/Test split: {args.train_ratio:.1%}/{args.val_ratio:.1%}/{1-args.train_ratio-args.val_ratio:.1%}")
    print("-"*60)
    
    try:
        # Step 1: Validate training data
        print("\n📋 STEP 1: Validating training data...")
        validate_training_data(args.source_dir)
        
        # Step 2: Read classes
        print("\n📝 STEP 2: Reading class definitions...")
        classes_file = os.path.join(args.source_dir, "classes.txt")
        classes = read_classes_file(classes_file)
        
        # Step 3: Setup directories
        print("\n📁 STEP 3: Setting up training directories...")
        target_dirs = setup_directories(args.output_dir, args.train_ratio)
        
        # Step 4: Split dataset
        print("\n✂️ STEP 4: Splitting dataset...")
        split_stats = split_dataset(
            source_images_dir=os.path.join(args.source_dir, "images"),
            source_labels_dir=os.path.join(args.source_dir, "labels"),
            target_dirs=target_dirs,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        
        # Step 5: Create YAML config
        print("\n⚙️ STEP 5: Creating YAML configuration...")
        yaml_path = create_yaml_config(target_dirs, classes)
        
        print("\n📊 DATASET SUMMARY:")
        print(f"  Total samples: {split_stats['total']}")
        print(f"  Training: {split_stats['train']} ({split_stats['train']/split_stats['total']:.1%})")
        print(f"  Validation: {split_stats['val']} ({split_stats['val']/split_stats['total']:.1%})")
        print(f"  Test: {split_stats['test']} ({split_stats['test']/split_stats['total']:.1%})")
        print(f"  Classes: {len(classes)} → {classes}")
        
        if args.no_train:
            print("\n⏸️ Stopping here (--no-train flag specified)")
            print(f"YOLO dataset ready at: {target_dirs['base']}")
            print(f"YAML config: {yaml_path}")
            return
        
        # Step 6: Train YOLO model
        print("\n🚀 STEP 6: Training YOLO model...")
        print("This may take a while depending on your hardware and dataset size...")
        
        results = train_yolo_model(
            yaml_config_path=yaml_path,
            model_name=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=os.path.join(target_dirs['base'], 'training_results')
        )
        
        if results:
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📂 Training setup: {target_dirs['base']}")
            print(f"📄 YAML config: {yaml_path}")
            print(f"🔗 Training results should be in: {target_dirs['base']}/training_results/")
        else:
            print("\n⚠️ Training failed, but data preparation was successful")
            print(f"You can manually train using: {yaml_path}")
            
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        print("Please check the error message above and fix any issues")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
