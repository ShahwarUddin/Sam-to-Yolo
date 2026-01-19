import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from datetime import datetime
from PIL import Image

"""
SAM3 Tank Detection with Training Data Collection

This script performs focused video segmentation to detect tanks using SAM3 model.
It can optionally save training data in YOLO format for later training.

Training Data Features:
- YOLO format: images/ and labels/ directories with normalized bounding boxes
- Accumulative: Multiple runs add more data instead of replacing
- Configurable confidence threshold for training data quality
- Timestamped filenames to avoid conflicts
- classes.txt file with class definitions

Usage:
    Modify the CONFIG section below to customize settings
    Set SAVE_FOR_TRAINING=True to enable training data collection
    Adjust TRAINING_THRESHOLD to control data quality
"""

#################################### CONFIGURATION SECTION ####################################
# Video processing settings
VIDEO_PATH = "tank_videos/v4.mp4"
MAX_FRAMES = 20
OUTPUT_DIR = "focused_video_results"

# Training data settings
SAVE_FOR_TRAINING = True  # Set to False to disable training data saving
TRAINING_THRESHOLD = 0.6  # Minimum confidence for training data
TRAINING_DIR = "training_data"
IMAGE_QUALITY = 95  # JPEG quality for saved training images (1-100)

# Target objects and their prompts
TARGET_PROMPTS = [
    # Format: (label, prompt)
    # ("goggles", "goggles"),
    # ("mask", "mask"), 
    # ("gloves", "gloves"),
    # ("cap", "protective cap"),
    # ("gown", "protective gown")
    ("tank", "army tank")
]

# Confidence thresholds for different object types
CONFIDENCE_THRESHOLDS = {
    # Adjust these based on detection difficulty for each object type
    # "gloves": 0.5,    # Lower threshold for harder-to-detect objects
    # "mask": 0.8,      # Higher threshold for easily detectable objects
    # "goggles": 0.7,   # Medium threshold
    # "cap": 0.6,
    # "gown": 0.7,
    "tank": 0.5
}

# Default confidence threshold for objects not in CONFIDENCE_THRESHOLDS
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Visualization colors for different object types (RGB values 0-1)
VISUALIZATION_COLORS = {
    # "goggles": [1.0, 0.2, 0.2],  # Bright Red
    # "mask": [0.2, 1.0, 0.2],     # Bright Green
    # "gloves": [0.2, 0.2, 1.0],   # Bright Blue
    # "cap": [1.0, 0.5, 0.0],      # Orange
    # "gown": [0.8, 0.0, 0.8],     # Magenta
    "tank": [0.0, 1.0, 0.5],       # Cyan
}

# Default color for objects not in VISUALIZATION_COLORS
DEFAULT_COLOR = [0.7, 0.7, 0.7]  # Light gray

# Class mapping for YOLO training (class_name: class_id)
CLASS_MAPPING = {
    "tank": 0,
    # Add more classes as needed:
    # "goggles": 1,
    # "mask": 2,
    # "gloves": 3,
    # "cap": 4,
    # "gown": 5
}

# Visualization settings
FIGURE_SIZE = (12, 6)
DPI = 150
MASK_ALPHA = 0.7  # Transparency for mask overlays
BOX_LINE_WIDTH = 3
FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16

#################################### END CONFIGURATION ####################################

#################################### Frame-by-Frame Video Processing ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def process_focused_video_segmentation(video_path=None, target_prompts=None, output_dir=None, 
                                      max_frames=None, save_for_training=None, training_threshold=None, training_dir=None):
    """
    Process video segmentation ONLY for specific target objects with high precision
    
    Args:
        video_path: Path to video file (uses VIDEO_PATH if None)
        target_prompts: List of (label, prompt) tuples (uses TARGET_PROMPTS if None)
        output_dir: Directory for visualization results (uses OUTPUT_DIR if None)
        max_frames: Maximum number of frames to process (uses MAX_FRAMES if None)
        save_for_training: If True, save annotations in YOLO format (uses SAVE_FOR_TRAINING if None)
        training_threshold: Minimum confidence threshold for saving training data (uses TRAINING_THRESHOLD if None)
        training_dir: Directory to save training data (uses TRAINING_DIR if None)
    """
    # Use configuration defaults if parameters not provided
    video_path = video_path or VIDEO_PATH
    target_prompts = target_prompts or TARGET_PROMPTS
    output_dir = output_dir or OUTPUT_DIR
    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    save_for_training = save_for_training if save_for_training is not None else SAVE_FOR_TRAINING
    training_threshold = training_threshold if training_threshold is not None else TRAINING_THRESHOLD
    training_dir = training_dir or TRAINING_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup training data directories if needed
    training_images_dir = None
    training_labels_dir = None
    class_mapping = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_for_training:
        training_images_dir = os.path.join(training_dir, "images")
        training_labels_dir = os.path.join(training_dir, "labels")
        os.makedirs(training_images_dir, exist_ok=True)
        os.makedirs(training_labels_dir, exist_ok=True)
        
        # Create class mapping (use configuration)
        class_mapping = CLASS_MAPPING.copy()
        
        # Create or update classes.txt
        classes_file = os.path.join(training_dir, "classes.txt")
        if not os.path.exists(classes_file):
            with open(classes_file, 'w') as f:
                for class_name in sorted(class_mapping.keys(), key=lambda x: class_mapping[x]):
                    f.write(f"{class_name}\n")
            print(f"Created classes file: {classes_file}")
        
        print(f"Training data will be saved to: {training_dir}")
        print(f"Training threshold: {training_threshold}")
    
    # Load the image model
    print("Loading SAM3 image model for focused detection...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    print(f"FOCUSED MODE: Only detecting -> {[prompt for _, prompt in target_prompts]}")
    
    # Select frames to process
    if max_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    print(f"Processing {len(frame_indices)} frames: {frame_indices}")
    
    all_results = {}
    
    for i, frame_idx in enumerate(frame_indices):
        print(f"\n=== Frame {frame_idx} ({i+1}/{len(frame_indices)}) ===")
        
        # Read specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert and create PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Set image for processing
        inference_state = processor.set_image(pil_image)
        
        frame_results = {}
        
        for label, prompt in target_prompts:
            print(f"  Searching for: '{prompt}'")
            
            try:
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                
                # ADAPTIVE confidence filtering - use configuration
                threshold = CONFIDENCE_THRESHOLDS.get(label, DEFAULT_CONFIDENCE_THRESHOLD)
                print(f"    Using confidence threshold: {threshold} for {label}")
                
                high_conf_masks, high_conf_boxes, high_conf_scores = [], [], []
                all_detections = []
                
                for mask, box, score in zip(output["masks"], output["boxes"], output["scores"]):
                    score_val = float(score.cpu() if hasattr(score, 'cpu') else score)
                    all_detections.append(score_val)
                    
                    if score_val > threshold:
                        high_conf_masks.append(mask)
                        high_conf_boxes.append(box)
                        high_conf_scores.append(score)
                
                # Show all detections for debugging
                if all_detections:
                    print(f"    All {label} detections: {[f'{s:.3f}' for s in sorted(all_detections, reverse=True)]}")
                else:
                    print(f"    No {label} detections at all")
                
                if high_conf_masks:
                    print(f"    ✓ Found {len(high_conf_masks)} high-confidence {label}(s)")
                    for j, score in enumerate(high_conf_scores):
                        print(f"      {j+1}. Confidence: {score:.3f}")
                else:
                    print(f"    ✗ No high-confidence {label} detected")
                
                frame_results[label] = {
                    "masks": high_conf_masks,
                    "boxes": high_conf_boxes,
                    "scores": high_conf_scores,
                    "prompt": prompt
                }
                
            except Exception as e:
                print(f"    Error: {e}")
                frame_results[label] = {"masks": [], "boxes": [], "scores": [], "prompt": prompt}
        
        all_results[frame_idx] = {
            "frame": frame_rgb,
            "results": frame_results
        }
        
        # Save training data if enabled
        if save_for_training:
            saved_annotations = save_training_data(
                frame_rgb, frame_results, frame_idx, 
                training_images_dir, training_labels_dir, 
                class_mapping, training_threshold, timestamp
            )
            if saved_annotations > 0:
                print(f"    💾 Added {saved_annotations} training samples")
        
        # Clear memory
        torch.cuda.empty_cache()
    
    cap.release()
    return all_results

def visualize_focused_results(all_results, output_dir=None):
    """
    Create clean visualizations showing ONLY the target objects
    """
    output_dir = output_dir or OUTPUT_DIR
    
    def draw_detection(mask, box, score, label, ax, color):
        # Draw mask
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        color_with_alpha = color + [MASK_ALPHA]  # Use configurable alpha
        h, w = mask_np.shape
        mask_image = mask_np.reshape(h, w, 1) * np.array(color_with_alpha).reshape(1, 1, -1)
        ax.imshow(mask_image)
        
        # Draw bounding box
        if hasattr(box, 'cpu'):
            box_np = box.cpu().numpy()
        else:
            box_np = np.array(box)
        
        x0, y0, x1, y1 = box_np
        w, h = x1 - x0, y1 - y0
        rect = plt.Rectangle((x0, y0), w, h, linewidth=BOX_LINE_WIDTH, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Label with confidence
        score_val = float(score.cpu() if hasattr(score, 'cpu') else score)
        ax.text(x0, y0-10, f"{label.upper()}: {score_val:.2f}", 
               color=color, fontsize=FONT_SIZE, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    for frame_idx, frame_data in all_results.items():
        frame_rgb = frame_data["frame"]
        frame_results = frame_data["results"]
        
        # Count total target detections
        total_targets = sum(len(results["masks"]) for results in frame_results.values())
        
        if total_targets == 0:
            print(f"Frame {frame_idx}: No target objects detected - skipping visualization")
            continue
        
        # Create focused visualization
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        fig.suptitle(f'FOCUSED DETECTION - Frame {frame_idx}', fontsize=TITLE_FONT_SIZE, fontweight='bold')
        
        # Original frame
        axes[0].imshow(frame_rgb)
        axes[0].set_title('Original Frame', fontsize=LABEL_FONT_SIZE)
        axes[0].axis('off')
        
        # TARGET OBJECTS ONLY
        axes[1].imshow(frame_rgb)
        
        detected_objects = []
        for label, results in frame_results.items():
            if results["masks"]:  # Only process if objects were found
                color = VISUALIZATION_COLORS.get(label, DEFAULT_COLOR)
                
                for mask, box, score in zip(results["masks"], results["boxes"], results["scores"]):
                    draw_detection(mask, box, score, label, axes[1], color)
                    detected_objects.append(f"{label}({score:.2f})")
        
        title = f"TARGET OBJECTS FOUND: {', '.join(detected_objects)}" if detected_objects else "NO TARGETS"
        axes[1].set_title(title, fontsize=FONT_SIZE, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save result
        output_path = os.path.join(output_dir, f'focused_frame_{frame_idx:04d}.png')
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved focused results: {output_path}")
        plt.close()

def save_training_data(frame_rgb, frame_results, frame_idx, training_images_dir, training_labels_dir, 
                      class_mapping, training_threshold, timestamp):
    """
    Save training data in YOLO format (image + .txt with normalized bounding boxes)
    """
    saved_count = 0
    
    # Collect all high-confidence detections for this frame
    training_annotations = []
    
    for label, results in frame_results.items():
        if label not in class_mapping:
            continue
            
        class_id = class_mapping[label]
        
        for mask, box, score in zip(results["masks"], results["boxes"], results["scores"]):
            score_val = float(score.cpu() if hasattr(score, 'cpu') else score)
            
            if score_val >= training_threshold:
                # Convert box to normalized YOLO format
                if hasattr(box, 'cpu'):
                    box_np = box.cpu().numpy()
                else:
                    box_np = np.array(box)
                
                x0, y0, x1, y1 = box_np
                img_height, img_width = frame_rgb.shape[:2]
                
                # Convert to YOLO format: (center_x, center_y, width, height) all normalized
                center_x = (x0 + x1) / 2 / img_width
                center_y = (y0 + y1) / 2 / img_height
                width = (x1 - x0) / img_width
                height = (y1 - y0) / img_height
                
                training_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    # Only save if we have annotations
    if training_annotations:
        # Create unique filename with timestamp and frame info
        base_name = f"tank_frame_{frame_idx:04d}_{timestamp}"
        
        # Save image
        img_path = os.path.join(training_images_dir, f"{base_name}.jpg")
        img_pil = Image.fromarray(frame_rgb)
        img_pil.save(img_path, "JPEG", quality=IMAGE_QUALITY)
        
        # Save annotations
        label_path = os.path.join(training_labels_dir, f"{base_name}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(training_annotations))
        
        saved_count = len(training_annotations)
        print(f"    ✓ Saved training data: {saved_count} annotations -> {base_name}")
    
    return saved_count

def main():
    print("=== SAM3 FOCUSED VIDEO SEGMENTATION ===")
    print("Only detecting specific target objects with high confidence")
    
    print(f"Video: {VIDEO_PATH}")
    print(f"Target objects: {[f'{label} ({prompt})' for label, prompt in TARGET_PROMPTS]}")
    print(f"Max frames: {MAX_FRAMES}")
    
    if SAVE_FOR_TRAINING:
        print(f"🎯 Training data collection: ENABLED")
        print(f"   Training threshold: {TRAINING_THRESHOLD}")
        print(f"   Training directory: {TRAINING_DIR}")
    else:
        print("🎯 Training data collection: DISABLED")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        return
    
    try:
        # Process with focused detection using configuration
        results = process_focused_video_segmentation()
        
        # Create clean visualizations
        visualize_focused_results(results)
        
        # Summary of what was actually found
        print("\n" + "="*50)
        print("FOCUSED DETECTION SUMMARY")
        print("="*50)
        
        total_detections = 0
        for frame_idx, frame_data in results.items():
            frame_results = frame_data["results"]
            frame_targets = []
            
            for label, data in frame_results.items():
                count = len(data["masks"])
                if count > 0:
                    avg_conf = np.mean([float(s.cpu() if hasattr(s, 'cpu') else s) for s in data["scores"]])
                    frame_targets.append(f"{count} {label}(s) @{avg_conf:.2f}")
                    total_detections += count
            
            if frame_targets:
                print(f"Frame {frame_idx:3d}: {', '.join(frame_targets)}")
            else:
                print(f"Frame {frame_idx:3d}: No target objects")
        
        print(f"\nTotal target detections: {total_detections}")
        print("Check 'focused_video_results/' for clean visualizations!")
        
        # Training data summary
        if SAVE_FOR_TRAINING:
            print("\n" + "="*50)
            print("TRAINING DATA SUMMARY")
            print("="*50)
            
            training_images_dir = os.path.join(TRAINING_DIR, "images")
            training_labels_dir = os.path.join(TRAINING_DIR, "labels")
            
            if os.path.exists(training_images_dir):
                image_count = len([f for f in os.listdir(training_images_dir) if f.endswith(('.jpg', '.png'))])
                label_count = len([f for f in os.listdir(training_labels_dir) if f.endswith('.txt')])
                print(f"📂 Training images: {image_count}")
                print(f"📄 Annotation files: {label_count}")
                print(f"💾 Data saved to: {TRAINING_DIR}/")
                print("   Structure:")
                print(f"   ├── images/     ({image_count} files)")
                print(f"   ├── labels/     ({label_count} files)")
                print(f"   └── classes.txt (class definitions)")
            else:
                print("❌ No training data was saved (no detections above threshold)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
CONFIGURATION EXAMPLES:

# Example 1: Basic tank detection
VIDEO_PATH = "tank_videos/v4.mp4"
TARGET_PROMPTS = [("tank", "army tank")]
MAX_FRAMES = 20

# Example 2: PPE detection setup
VIDEO_PATH = "ppe_video.mp4"
TARGET_PROMPTS = [
    ("goggles", "goggles"),
    ("mask", "face mask"),
    ("gloves", "safety gloves"),
    ("cap", "hard hat")
]
CONFIDENCE_THRESHOLDS = {
    "goggles": 0.6,
    "mask": 0.8,
    "gloves": 0.5,  # Lower threshold for harder detection
    "cap": 0.7
}
VISUALIZATION_COLORS = {
    "goggles": [1.0, 0.2, 0.2],  # Red
    "mask": [0.2, 1.0, 0.2],     # Green
    "gloves": [0.2, 0.2, 1.0],   # Blue
    "cap": [1.0, 0.5, 0.0]       # Orange
}

# Example 3: High-quality training data collection
SAVE_FOR_TRAINING = True
TRAINING_THRESHOLD = 0.8  # Higher threshold for better quality
IMAGE_QUALITY = 100  # Maximum quality
MAX_FRAMES = 100  # More frames for better dataset

# Example 4: Quick testing setup
SAVE_FOR_TRAINING = False
MAX_FRAMES = 5
TRAINING_THRESHOLD = 0.5  # Lower threshold to see more detections
"""
