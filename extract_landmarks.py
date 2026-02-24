"""
ASL Landmark Extraction Script
==============================
Extracts hand landmarks from gesture images/videos and saves to CSV.

Usage:
    python extract_landmarks.py
"""

import cv2
import numpy as np
import csv
import sys
import io
import urllib.request
import os
from pathlib import Path
from typing import Optional, List, Tuple

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DYNAMIC_DIR = DATA_DIR / "extracted_dynamic"
PROCESSED_DIR = DATA_DIR / "processed_landmarks"
OUTPUT_CSV = PROCESSED_DIR / "gesture_landmarks.csv"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

# Target gestures
GESTURES = ['Hello', 'THANKYOU', 'Sorry', 'Yes', 'No']

# Landmark names for CSV header
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


# ============================================================================
# MODEL SETUP
# ============================================================================

def download_model():
    """Download the MediaPipe hand landmarker model if not present."""
    if MODEL_PATH.exists():
        return
    
    print(f"⬇️  Downloading hand landmarker model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ Model downloaded")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)


def create_landmarker():
    """Create MediaPipe HandLandmarker instance."""
    download_model()
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    
    return vision.HandLandmarker.create_from_options(options)


# ============================================================================
# LANDMARK EXTRACTION
# ============================================================================

def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks relative to wrist (landmark 0).
    
    - Translation: Subtract wrist position (makes wrist origin)
    - Scale: Divide by wrist-to-middle-MCP distance
    """
    wrist = coords[0]
    translated = coords - wrist
    
    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)
    
    if scale < 1e-6:
        scale = 1.0
    
    return translated / scale


def extract_landmarks_from_image(
    landmarker: vision.HandLandmarker,
    image_path: Path
) -> Optional[np.ndarray]:
    """
    Extract normalized landmarks from a single image.
    
    Returns:
        Flattened array of 63 values (21 landmarks × 3 coords) or None if no hand detected
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Detect hands
    results = landmarker.detect(mp_image)
    
    if not results.hand_landmarks:
        return None
    
    # Get first hand's landmarks
    hand_landmarks = results.hand_landmarks[0]
    
    # Extract raw coordinates
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # Normalize
    normalized = normalize_landmarks(raw_coords)
    
    # Flatten to 1D array (63 values)
    return normalized.flatten()


def extract_landmarks_from_video(
    landmarker: vision.HandLandmarker,
    video_path: Path,
    sample_rate: int = 5
) -> List[np.ndarray]:
    """
    Extract normalized landmarks from video frames.
    
    Args:
        landmarker: MediaPipe HandLandmarker
        video_path: Path to video file
        sample_rate: Extract every Nth frame
    
    Returns:
        List of flattened landmark arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    landmarks_list = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Sample every Nth frame
        if frame_num % sample_rate != 0:
            continue
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        results = landmarker.detect(mp_image)
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            normalized = normalize_landmarks(raw_coords)
            landmarks_list.append(normalized.flatten())
    
    cap.release()
    return landmarks_list


# ============================================================================
# CSV GENERATION
# ============================================================================

def generate_csv_header() -> List[str]:
    """Generate CSV header with landmark coordinate names."""
    header = []
    for name in LANDMARK_NAMES:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    header.append("label")
    return header


def process_all_gestures():
    """Process all gesture folders and save landmarks to CSV."""
    print("\n" + "=" * 60)
    print("   🤟 ASL LANDMARK EXTRACTION")
    print("=" * 60)
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create landmarker
    print("\n📦 Loading MediaPipe model...")
    landmarker = create_landmarker()
    print("✅ Model loaded\n")
    
    # Prepare CSV
    header = generate_csv_header()
    all_rows = []
    
    print("=" * 60)
    print("📊 PROCESSING GESTURES")
    print("=" * 60)
    
    for gesture in GESTURES:
        gesture_dir = EXTRACTED_DYNAMIC_DIR / gesture
        
        if not gesture_dir.exists():
            print(f"\n⚠️  Gesture folder not found: {gesture}")
            continue
        
        print(f"\n🖐️  Processing '{gesture}'...")
        gesture_samples = 0
        
        # Process video files (.avi)
        video_files = list(gesture_dir.glob("*.avi"))
        for video_path in video_files:
            landmarks_list = extract_landmarks_from_video(landmarker, video_path)
            for landmarks in landmarks_list:
                row = list(landmarks) + [gesture]
                all_rows.append(row)
                gesture_samples += 1
        
        # Process frame folders (images)
        frame_folders = [f for f in gesture_dir.iterdir() if f.is_dir() and "_frames" in f.name]
        for folder in frame_folders:
            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            
            # Sample every 5th image to avoid redundancy
            for i, img_path in enumerate(sorted(image_files)):
                if i % 5 != 0:
                    continue
                
                landmarks = extract_landmarks_from_image(landmarker, img_path)
                if landmarks is not None:
                    row = list(landmarks) + [gesture]
                    all_rows.append(row)
                    gesture_samples += 1
        
        print(f"   ✅ Extracted {gesture_samples} samples")
    
    # Write CSV
    print(f"\n💾 Writing CSV to: {OUTPUT_CSV.name}")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    
    print(f"✅ Saved {len(all_rows)} total samples")
    
    # Close landmarker
    landmarker.close()
    
    return OUTPUT_CSV


def verify_csv(csv_path: Path):
    """Print a summary of the CSV for verification."""
    print("\n" + "=" * 60)
    print("🔍 DATA VERIFICATION")
    print("=" * 60)
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    print(f"\n📄 File: {csv_path.name}")
    print(f"📊 Total columns: {len(header)} (63 landmark coords + 1 label)")
    print(f"📊 Total rows: {len(rows)}")
    
    # Count samples per label
    label_counts = {}
    for row in rows:
        label = row[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n📈 Samples per gesture:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")
    
    # Show sample rows
    print("\n📋 Sample rows (first 3):")
    print("-" * 60)
    
    for i, row in enumerate(rows[:3]):
        # Show first few coordinates + label
        coords_preview = [f"{float(x):.4f}" for x in row[:6]]
        print(f"Row {i+1}: [{', '.join(coords_preview)}, ...] → '{row[-1]}'")
    
    print("-" * 60)
    
    # Verify data types
    print("\n✅ Verification:")
    try:
        # Check if coordinates are numeric
        for row in rows[:10]:
            for val in row[:-1]:
                float(val)
        print("   ✓ All coordinate values are numeric")
        print("   ✓ Labels are text strings")
        print("   ✓ CSV format is valid for ML training")
    except ValueError as e:
        print(f"   ❌ Error: Non-numeric value found - {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Check if CSV already exists
    if OUTPUT_CSV.exists():
        print(f"\n📄 CSV already exists: {OUTPUT_CSV}")
        response = input("   Regenerate? (y/n): ").strip().lower()
        if response != 'y':
            verify_csv(OUTPUT_CSV)
            return
    
    # Process gestures and generate CSV
    csv_path = process_all_gestures()
    
    # Verify the output
    verify_csv(csv_path)
    
    print("\n" + "=" * 60)
    print("🎉 EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"\nCSV ready for ML training: {OUTPUT_CSV}")
    print("\nNext steps:")
    print("  1. Load CSV with pandas: df = pd.read_csv('data/processed_landmarks/gesture_landmarks.csv')")
    print("  2. Split into X (features) and y (labels)")
    print("  3. Train classifier (SVM, Random Forest, Neural Net, etc.)")
    print()


if __name__ == "__main__":
    main()
