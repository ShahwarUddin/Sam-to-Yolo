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
    Set save_for_training=True in main() to enable training data collection
    Adjust training_threshold to control data quality (default: 0.6)
"""

#################################### Frame-by-Frame Video Processing ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def process_focused_video_segmentation(video_path, target_prompts, output_dir="focused_video_results", 
                                      max_frames=5, save_for_training=False, training_threshold=0.7, training_dir="training_data"):
    """
    Process video segmentation ONLY for specific target objects with high precision
    
    Args:
        video_path: Path to video file
        target_prompts: List of (label, prompt) tuples
        output_dir: Directory for visualization results
        max_frames: Maximum number of frames to process
        save_for_training: If True, save annotations in YOLO format for training
        training_threshold: Minimum confidence threshold for saving training data
        training_dir: Directory to save training data
    """
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
        
        # Create class mapping (tank = 0, can be extended later)
        class_mapping = {"tank": 0}
        
        # Create or update classes.txt
        classes_file = os.path.join(training_dir, "classes.txt")
        if not os.path.exists(classes_file):
            with open(classes_file, 'w') as f:
                f.write("tank\n")
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
                
                # ADAPTIVE confidence filtering - different thresholds for different objects
                confidence_thresholds = {
                    # 'gloves': 0.5,    # Lower threshold for gloves (harder to detect)
                    # 'mask': 0.8,      # High threshold for masks
                    # 'goggles': 0.7,   # Medium threshold for goggles
                    "tank": 0.5
                }
                
                threshold = confidence_thresholds.get(label, 0.8)
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

def visualize_focused_results(all_results, output_dir="focused_video_results"):
    """
    Create clean visualizations showing ONLY the target objects
    """
    # Colors for target objects only
    target_colors = {
        # 'goggles': [1.0, 0.2, 0.2],         # Bright Red
        # 'mask': [0.2, 1.0, 0.2],            # Bright Green
        # 'gloves': [0.2, 0.2, 1.0],          # Bright Blue
        # 'cap': [1.0, 0.5, 0.0],             # Orange
        'tank': [0.0, 1.0, 0.5], # Cyan
    }
    
    def draw_detection(mask, box, score, label, ax, color):
        # Draw mask
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        color_with_alpha = color + [0.7]  # Strong alpha for visibility
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
        rect = plt.Rectangle((x0, y0), w, h, linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Label with confidence
        score_val = float(score.cpu() if hasattr(score, 'cpu') else score)
        ax.text(x0, y0-10, f"{label.upper()}: {score_val:.2f}", 
               color=color, fontsize=12, fontweight='bold',
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'FOCUSED DETECTION - Frame {frame_idx}', fontsize=16, fontweight='bold')
        
        # Original frame
        axes[0].imshow(frame_rgb)
        axes[0].set_title('Original Frame', fontsize=14)
        axes[0].axis('off')
        
        # TARGET OBJECTS ONLY
        axes[1].imshow(frame_rgb)
        
        detected_objects = []
        for label, results in frame_results.items():
            if results["masks"]:  # Only process if objects were found
                color = target_colors.get(label, [0.7, 0.7, 0.7])
                
                for mask, box, score in zip(results["masks"], results["boxes"], results["scores"]):
                    draw_detection(mask, box, score, label, axes[1], color)
                    detected_objects.append(f"{label}({score:.2f})")
        
        title = f"TARGET OBJECTS FOUND: {', '.join(detected_objects)}" if detected_objects else "NO TARGETS"
        axes[1].set_title(title, fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save result
        output_path = os.path.join(output_dir, f'focused_frame_{frame_idx:04d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
        img_pil.save(img_path, "JPEG", quality=95)
        
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
    
    # Configuration
    video_path = "tank_videos/v4.mp4"
    max_frames = 20
    
    # Training data parameters
    save_for_training = True  # Set to False to disable training data saving
    training_threshold = 0.6  # Minimum confidence for training data (adjustable)
    training_dir = "training_data"  # Directory for YOLO training data
    
    # ONLY the objects you want to detect - try multiple prompts for gloves
    target_prompts = [
        # ("goggles", "goggles"),
        # ("mask", "mask"), 
        # ("gloves", "gloves"),                # Try simple "gloves" first
        # ("cap","protective cap"),
        # ("gown","protective gown")
        ("tank", "army tank")
    ]
    
    print(f"Video: {video_path}")
    print(f"Target objects: {[f'{label} ({prompt})' for label, prompt in target_prompts]}")
    print(f"Max frames: {max_frames}")
    
    if save_for_training:
        print(f"🎯 Training data collection: ENABLED")
        print(f"   Training threshold: {training_threshold}")
        print(f"   Training directory: {training_dir}")
    else:
        print("🎯 Training data collection: DISABLED")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    try:
        # Process with focused detection
        results = process_focused_video_segmentation(
            video_path, target_prompts, max_frames=max_frames,
            save_for_training=save_for_training, 
            training_threshold=training_threshold,
            training_dir=training_dir
        )
        
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
        if save_for_training:
            print("\n" + "="*50)
            print("TRAINING DATA SUMMARY")
            print("="*50)
            
            training_images_dir = os.path.join(training_dir, "images")
            training_labels_dir = os.path.join(training_dir, "labels")
            
            if os.path.exists(training_images_dir):
                image_count = len([f for f in os.listdir(training_images_dir) if f.endswith(('.jpg', '.png'))])
                label_count = len([f for f in os.listdir(training_labels_dir) if f.endswith('.txt')])
                print(f"📂 Training images: {image_count}")
                print(f"📄 Annotation files: {label_count}")
                print(f"💾 Data saved to: {training_dir}/")
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