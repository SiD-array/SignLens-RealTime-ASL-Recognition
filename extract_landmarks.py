"""
ASL Landmark Extraction Script
==============================
Extracts hand landmarks from gesture videos and saves 30-frame sequences
as .npy files for LSTM-based action recognition.

Usage:
    python extract_landmarks.py
"""

import cv2
import numpy as np
import sys
import io
import json
import urllib.request
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
SEQUENCE_DIR = DATA_DIR / "sequences_lstm"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

LABELS_PATH = PROJECT_ROOT / "labels.json"
SEQUENCE_LENGTH = 30  # Number of frames per gesture clip (matches LSTM input)


# ============================================================================
# MODEL SETUP
# ============================================================================

def load_labels() -> List[str]:
    """Load gesture labels from labels.json."""
    if not LABELS_PATH.exists():
        print(f"\n❌ Error: labels.json not found at {LABELS_PATH}")
        print("   Create labels.json as a JSON list of label strings, e.g.:")
        print('   ["Hello", "THANKYOU", "Sorry", "Yes", "No"]')
        sys.exit(1)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        print("\n❌ Error: labels.json must be a JSON list of label strings.")
        sys.exit(1)

    return labels


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
    sample_rate: int = 2
) -> List[np.ndarray]:
    """
    Extract normalized landmarks from video frames.
    
    Args:
        landmarker: MediaPipe HandLandmarker
        video_path: Path to video file
        sample_rate: Extract every Nth frame
    
    Returns:
        List of flattened landmark arrays (one per sampled frame)
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
    # SEQUENCE GENERATION
    # ============================================================================


def chunk_into_sequences(frames: List[np.ndarray], sequence_length: int) -> List[np.ndarray]:
    """
    Split a list of frame-level landmark vectors into fixed-length sequences.

    Uses an overlapping sliding window so that fast, dynamic gestures
    (like 'No') are captured multiple times across the motion.
    """
    sequences: List[np.ndarray] = []
    num_frames = len(frames)

    if num_frames < sequence_length:
        return sequences

    # Step size smaller than sequence length → overlapping clips
    step = max(1, sequence_length // 5)  # e.g. 30 // 5 = 6

    for start in range(0, num_frames - sequence_length + 1, step):
        window = frames[start : start + sequence_length]
        if len(window) == sequence_length:
            seq = np.stack(window, axis=0).astype(np.float32)  # (sequence_length, 63)
            sequences.append(seq)

    return sequences


def process_all_gestures() -> Path:
    """Process all gesture folders and save 30-frame sequences as .npy clips."""
    print("\n" + "=" * 60)
    print("   🤟 ASL LANDMARK EXTRACTION (SEQUENCES)")
    print("=" * 60)

    labels = load_labels()

    # Ensure output directory exists
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Create landmarker
    print("\n📦 Loading MediaPipe model...")
    landmarker = create_landmarker()
    print("✅ Model loaded\n")

    print("=" * 60)
    print("📊 PROCESSING GESTURES INTO 30-FRAME CLIPS")
    print("=" * 60)

    total_sequences = 0

    for gesture in labels:
        gesture_dir = EXTRACTED_DYNAMIC_DIR / gesture

        if not gesture_dir.exists():
            print(f"\n⚠️  Gesture folder not found: {gesture}")
            continue

        output_dir = SEQUENCE_DIR / gesture
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🖐️  Processing '{gesture}'...")
        gesture_sequences = 0

        # ---------------------------------------------------------------------
        # From raw video files (.avi)
        # ---------------------------------------------------------------------
        video_files = list(gesture_dir.glob("*.avi"))
        for video_path in video_files:
            frame_landmarks = extract_landmarks_from_video(landmarker, video_path)
            sequences = chunk_into_sequences(frame_landmarks, SEQUENCE_LENGTH)

            for seq_idx, seq in enumerate(sequences):
                out_path = output_dir / f"{video_path.stem}_seq{seq_idx:03d}.npy"
                np.save(out_path, seq)
                gesture_sequences += 1
                total_sequences += 1

        # ---------------------------------------------------------------------
        # From pre-extracted frame folders (images)
        # ---------------------------------------------------------------------
        frame_folders = [f for f in gesture_dir.iterdir() if f.is_dir() and "_frames" in f.name]
        for folder in frame_folders:
            image_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
            frame_landmarks: List[np.ndarray] = []

            for img_path in image_files:
                landmarks = extract_landmarks_from_image(landmarker, img_path)
                if landmarks is not None:
                    frame_landmarks.append(landmarks)

            sequences = chunk_into_sequences(frame_landmarks, SEQUENCE_LENGTH)
            for seq_idx, seq in enumerate(sequences):
                out_path = output_dir / f"{folder.name}_seq{seq_idx:03d}.npy"
                np.save(out_path, seq)
                gesture_sequences += 1
                total_sequences += 1

        print(f"   ✅ Saved {gesture_sequences} sequences for '{gesture}'")

    # Close landmarker
    landmarker.close()

    print(f"\n✅ Saved {total_sequences} total sequences to {SEQUENCE_DIR}")
    return SEQUENCE_DIR


def summarize_sequences(base_dir: Path):
    """Print a summary of generated .npy sequences."""
    print("\n" + "=" * 60)
    print("🔍 SEQUENCE DATA SUMMARY")
    print("=" * 60)

    if not base_dir.exists():
        print(f"\n❌ No sequences found at {base_dir}")
        return

    label_counts = {}
    example_shape = None

    for gesture_dir in base_dir.iterdir():
        if not gesture_dir.is_dir():
            continue

        npy_files = list(gesture_dir.glob("*.npy"))
        if not npy_files:
            continue

        label = gesture_dir.name
        label_counts[label] = len(npy_files)

        # Peek at first file to confirm shape
        if example_shape is None:
            sample = np.load(npy_files[0])
            example_shape = sample.shape

    if not label_counts:
        print("\n⚠️  No .npy sequence files found.")
        return

    print("\n📈 Sequences per gesture:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")

    if example_shape is not None:
        print(f"\n🧱 Example sequence shape: {example_shape} (frames, features)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Check if sequences already exist
    existing = SEQUENCE_DIR.exists() and any(SEQUENCE_DIR.rglob("*.npy"))
    if existing:
        print(f"\n📄 Sequence data already exists under: {SEQUENCE_DIR}")
        response = input("   Regenerate sequences from raw videos/frames? (y/n): ").strip().lower()
        if response != "y":
            summarize_sequences(SEQUENCE_DIR)
            return

    # Process gestures and generate sequences
    seq_dir = process_all_gestures()

    # Summarize the output
    summarize_sequences(seq_dir)

    print("\n" + "=" * 60)
    print("🎉 EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"\n30-frame .npy sequences ready for LSTM training in: {seq_dir}")
    print()


if __name__ == "__main__":
    main()
