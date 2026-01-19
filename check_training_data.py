 #!/usr/bin/env python3
"""
Quick script to analyze the current training data from ANNOTATION_PIPELINE.py

Usage:
    python check_training_data.py
"""

import os
from pathlib import Path
from collections import Counter

def analyze_training_data(training_dir="training_data"):
    """Analyze the current training data structure and content"""
    
    print("🔍 TRAINING DATA ANALYSIS")
    print("="*50)
    
    images_dir = os.path.join(training_dir, "images")
    labels_dir = os.path.join(training_dir, "labels")
    classes_file = os.path.join(training_dir, "classes.txt")
    
    # Check structure
    print("📁 Directory Structure:")
    for path, expected in [(training_dir, "training_data"), (images_dir, "images"), 
                          (labels_dir, "labels"), (classes_file, "classes.txt")]:
        status = "✓" if os.path.exists(path) else "❌"
        print(f"  {status} {expected}: {path}")
    
    if not os.path.exists(training_dir):
        print("❌ Training data directory not found!")
        return
    
    # Read classes
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"\n📝 Classes ({len(classes)}): {classes}")
    else:
        print("\n❌ Classes file not found!")
        return
    
    # Count files
    image_files = []
    label_files = []
    
    if os.path.exists(images_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(images_dir).glob(ext))
    
    if os.path.exists(labels_dir):
        label_files = list(Path(labels_dir).glob('*.txt'))
    
    print(f"\n📊 File Counts:")
    print(f"  Images: {len(image_files)}")
    print(f"  Labels: {len(label_files)}")
    
    # Check matching pairs
    matched_pairs = []
    unmatched_images = []
    unmatched_labels = []
    
    image_stems = {img.stem for img in image_files}
    label_stems = {lbl.stem for lbl in label_files}
    
    for img in image_files:
        if img.stem in label_stems:
            matched_pairs.append(img.stem)
        else:
            unmatched_images.append(img.name)
    
    for lbl in label_files:
        if lbl.stem not in image_stems:
            unmatched_labels.append(lbl.name)
    
    print(f"\n🔗 Data Matching:")
    print(f"  Valid pairs: {len(matched_pairs)}")
    print(f"  Unmatched images: {len(unmatched_images)}")
    print(f"  Unmatched labels: {len(unmatched_labels)}")
    
    if unmatched_images:
        print(f"  ⚠️ Unmatched images: {unmatched_images[:5]}{'...' if len(unmatched_images) > 5 else ''}")
    if unmatched_labels:
        print(f"  ⚠️ Unmatched labels: {unmatched_labels[:5]}{'...' if len(unmatched_labels) > 5 else ''}")
    
    # Analyze annotations
    if label_files and matched_pairs:
        print(f"\n📋 Annotation Analysis:")
        total_annotations = 0
        class_counts = Counter()
        
        for label_file in label_files[:10]:  # Sample first 10 files
            try:
                with open(label_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    total_annotations += len(lines)
                    
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id < len(classes):
                                class_counts[classes[class_id]] += 1
            except Exception as e:
                print(f"    ⚠️ Error reading {label_file.name}: {e}")
        
        print(f"  Total annotations (sampled): {total_annotations}")
        print(f"  Class distribution: {dict(class_counts)}")
        
        # Calculate average annotations per image
        if matched_pairs:
            avg_annotations = total_annotations / min(len(matched_pairs), 10)
            print(f"  Avg annotations per image: {avg_annotations:.2f}")
    
    # File size analysis
    if image_files:
        sizes = []
        for img_file in image_files[:10]:  # Sample first 10 files
            try:
                size_mb = img_file.stat().st_size / (1024 * 1024)
                sizes.append(size_mb)
            except:
                pass
        
        if sizes:
            print(f"\n💾 File Size Analysis (sampled):")
            print(f"  Avg image size: {sum(sizes)/len(sizes):.2f} MB")
            print(f"  Size range: {min(sizes):.2f} - {max(sizes):.2f} MB")
    
    # Training readiness
    print(f"\n🎯 Training Readiness:")
    ready = True
    
    if len(matched_pairs) < 10:
        print(f"  ⚠️ Low sample count: {len(matched_pairs)} (recommend 50+)")
        ready = False
    else:
        print(f"  ✓ Sample count: {len(matched_pairs)}")
    
    if len(classes) == 0:
        print(f"  ❌ No classes defined")
        ready = False
    else:
        print(f"  ✓ Classes defined: {len(classes)}")
    
    if len(matched_pairs) != len(image_files):
        print(f"  ⚠️ Some images missing labels")
    else:
        print(f"  ✓ All images have labels")
    
    print(f"\n{'✅ READY FOR TRAINING!' if ready else '⚠️ NEEDS ATTENTION'}")
    
    if ready:
        print(f"\nNext steps:")
        print(f"  1. Install requirements: pip install -r yolo_requirements.txt")
        print(f"  2. Run training: python train_yolo_model.py")
        print(f"  3. Or prepare only: python train_yolo_model.py --no-train")

if __name__ == "__main__":
    analyze_training_data()
