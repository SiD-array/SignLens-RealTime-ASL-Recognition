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
# WLASL-style: one 30-frame sequence per video, flat naming
SEQUENCE_DIR_MULTI_HAND = DATA_DIR / "sequences_multi_hand"
EXTRACTION_ERRORS_FILE = DATA_DIR / "extraction_errors.txt"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

LABELS_PATH = PROJECT_ROOT / "labels.json"
SEQUENCE_LENGTH = 30  # Frames per clip (matches main.py LSTM)
# Multi-hand: 126 features (21×3×2) for WLASL two-hand signs; must match main.py MULTI_HAND
MULTI_HAND = True
FEATURE_SIZE_ONE_HAND = 63
FEATURE_SIZE = 126 if MULTI_HAND else FEATURE_SIZE_ONE_HAND
MODEL_TRAINED_HAND = "Right"  # Mirror other hand to match (same as main.py)


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
    Scale: Divide by wrist-to-middle-MCP distance (same as main.py).
    """
    wrist = coords[0]
    translated = coords - wrist
    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)
    if scale < 1e-6:
        scale = 1.0
    return translated / scale


def _one_hand_to_63(hand_landmarks, handedness: str) -> np.ndarray:
    """Single-hand: normalize, mirror if needed (match main.py), return 63-dim."""
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    normalized = normalize_landmarks(raw)
    if handedness != MODEL_TRAINED_HAND:
        normalized[:, 0] = -normalized[:, 0]
    return normalized.flatten().astype(np.float32)


def _results_to_126(results) -> Optional[np.ndarray]:
    """
    126-dim: Hand 0 → features 0–62, Hand 1 → 63–125 (canonical order: Left then Right).
    One hand detected → other 63 values zero-padded. Distance normalization per hand.
    """
    if not results.hand_landmarks:
        return None
    hands: List[Tuple[str, np.ndarray]] = []
    for i, hand_landmarks in enumerate(results.hand_landmarks):
        handedness = "Right"
        if results.handedness and i < len(results.handedness) and results.handedness[i]:
            handedness = results.handedness[i][0].category_name
        hands.append((handedness, _one_hand_to_63(hand_landmarks, handedness)))
    hands.sort(key=lambda x: (0 if x[0] == "Left" else 1, x[0]))
    if len(hands) == 2:
        return np.concatenate([hands[0][1], hands[1][1]], axis=0)
    if len(hands) == 1:
        return np.concatenate([hands[0][1], np.zeros(FEATURE_SIZE_ONE_HAND, dtype=np.float32)], axis=0)
    return None


def extract_landmarks_from_image(
    landmarker: vision.HandLandmarker,
    image_path: Path
) -> Optional[np.ndarray]:
    """
    Extract landmarks from one image. Returns 126-dim if MULTI_HAND else 63-dim, or None.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = landmarker.detect(mp_image)
    if MULTI_HAND:
        return _results_to_126(results)
    if not results.hand_landmarks:
        return None
    hand_landmarks = results.hand_landmarks[0]
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    normalized = normalize_landmarks(raw_coords)
    return normalized.flatten().astype(np.float32)


def extract_landmarks_from_video(
    landmarker: vision.HandLandmarker,
    video_path: Path,
    sample_rate: int = 2
) -> List[np.ndarray]:
    """
    Extract landmarks from video frames. Each frame → 126-dim if MULTI_HAND else 63-dim.
    Uses sample_rate to skip frames; use sample_rate=1 for every frame (WLASL mode).
    """
    cap = cv2.VideoCapture(str(video_path))
    landmarks_list: List[np.ndarray] = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % sample_rate != 0:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_image)
        if MULTI_HAND:
            vec = _results_to_126(results)
            if vec is not None:
                landmarks_list.append(vec)
        elif results.hand_landmarks:
            raw = np.array([[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0]])
            normalized = normalize_landmarks(raw)
            landmarks_list.append(normalized.flatten().astype(np.float32))
    cap.release()
    return landmarks_list


def frames_to_one_sequence(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Turn a list of frame vectors into exactly one (30, FEATURE_SIZE) sequence.
    - If len(frames) >= 30: sample 30 frames evenly (e.g. linspace).
    - If len(frames) < 30: pad at the beginning with zeros (LSTM masking will ignore).
    - If len(frames) == 0: return None (caller should log and skip).
    """
    n = len(frames)
    if n == 0:
        return None
    out = np.zeros((SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
    if n >= SEQUENCE_LENGTH:
        indices = np.linspace(0, n - 1, SEQUENCE_LENGTH, dtype=int)
        out[:] = np.stack([frames[i] for i in indices], axis=0)
    else:
        out[SEQUENCE_LENGTH - n :] = np.stack(frames, axis=0)
    return out


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
            seq = np.stack(window, axis=0).astype(np.float32)  # (sequence_length, FEATURE_SIZE)
            sequences.append(seq)

    return sequences


def process_wlasl_videos() -> Path:
    """
    WLASL-style extraction: one 30-frame sequence per video.
    - 126 features (Hand 0: 0–62, Hand 1: 63–125); one hand → zeros for the other.
    - >30 frames: sample 30 evenly; <30: zero-pad at start (LSTM masking).
    - Output: data/sequences_multi_hand/{gloss}_{video_id}.npy
    - Failed videos (no hands in any frame) logged to extraction_errors.txt.
    """
    print("\n" + "=" * 60)
    print("   🤟 WLASL MULTI-HAND EXTRACTION (1 sequence per video)")
    print("=" * 60)

    labels = load_labels()
    SEQUENCE_DIR_MULTI_HAND.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📦 Loading MediaPipe model (num_hands=2)...")
    landmarker = create_landmarker()
    print("✅ Model loaded\n")
    print(f"📊 Output: {SEQUENCE_DIR_MULTI_HAND} (shape 30×{FEATURE_SIZE})")
    print("📊 Distance normalization: wrist → middle-finger MCP scale")
    print("=" * 60)

    total_saved = 0
    error_entries: List[str] = []

    for gesture in labels:
        gesture_dir = EXTRACTED_DYNAMIC_DIR / gesture
        if not gesture_dir.exists():
            print(f"\n⚠️  Skipping missing folder: {gesture}")
            continue

        video_files = list(gesture_dir.glob("*.mp4")) + list(gesture_dir.glob("*.avi"))
        print(f"\n🖐️  {gesture} ({len(video_files)} videos)")

        gesture_saved = 0
        for video_path in video_files:
            video_id = video_path.stem
            frame_landmarks = extract_landmarks_from_video(
                landmarker, video_path, sample_rate=1
            )
            seq = frames_to_one_sequence(frame_landmarks)

            if seq is None:
                error_entries.append(f"{gesture}/{video_path.name}")
                continue

            out_path = SEQUENCE_DIR_MULTI_HAND / f"{gesture}_{video_id}.npy"
            np.save(out_path, seq)
            total_saved += 1
            gesture_saved += 1

        print(f"   ✅ Saved {gesture_saved} sequences")

    if error_entries:
        with open(EXTRACTION_ERRORS_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(error_entries))
        print(f"\n⚠️  Logged {len(error_entries)} failed videos to {EXTRACTION_ERRORS_FILE.name}")

    landmarker.close()
    print(f"\n✅ Total sequences saved: {total_saved} → {SEQUENCE_DIR_MULTI_HAND}")
    return SEQUENCE_DIR_MULTI_HAND


def process_all_gestures() -> Path:
    """Legacy: multiple overlapping sequences per video → sequences_lstm/{gloss}/."""
    print("\n" + "=" * 60)
    print("   🤟 ASL LANDMARK EXTRACTION (SEQUENCES)")
    print("=" * 60)

    labels = load_labels()
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📦 Loading MediaPipe model...")
    landmarker = create_landmarker()
    print("✅ Model loaded\n")
    print("=" * 60)
    print(f"📊 PROCESSING GESTURES INTO 30-FRAME CLIPS (shape 30×{FEATURE_SIZE})")
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

        video_files = list(gesture_dir.glob("*.avi")) + list(gesture_dir.glob("*.mp4"))
        for video_path in video_files:
            frame_landmarks = extract_landmarks_from_video(landmarker, video_path)
            sequences = chunk_into_sequences(frame_landmarks, SEQUENCE_LENGTH)
            for seq_idx, seq in enumerate(sequences):
                out_path = output_dir / f"{video_path.stem}_seq{seq_idx:03d}.npy"
                np.save(out_path, seq)
                gesture_sequences += 1
                total_sequences += 1

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

    landmarker.close()
    print(f"\n✅ Saved {total_sequences} total sequences to {SEQUENCE_DIR}")
    return SEQUENCE_DIR


def summarize_sequences(base_dir: Path):
    """Print a summary of generated .npy sequences (supports per-gloss dirs or flat gloss_id.npy)."""
    print("\n" + "=" * 60)
    print("🔍 SEQUENCE DATA SUMMARY")
    print("=" * 60)

    if not base_dir.exists():
        print(f"\n❌ No sequences found at {base_dir}")
        return

    label_counts = {}
    example_shape = None

    # Flat layout: gloss_videoid.npy
    npy_flat = list(base_dir.glob("*.npy"))
    if npy_flat:
        for p in npy_flat:
            # "gloss_videoid.npy" -> gloss is everything before last _
            parts = p.stem.rsplit("_", 1)
            gloss = parts[0] if len(parts) == 2 else p.stem
            label_counts[gloss] = label_counts.get(gloss, 0) + 1
        if example_shape is None and npy_flat:
            example_shape = np.load(npy_flat[0]).shape
    else:
        # Nested layout: base_dir/gloss/*.npy
        for gesture_dir in base_dir.iterdir():
            if not gesture_dir.is_dir():
                continue
            npy_files = list(gesture_dir.glob("*.npy"))
            if not npy_files:
                continue
            label = gesture_dir.name
            label_counts[label] = len(npy_files)
            if example_shape is None:
                example_shape = np.load(npy_files[0]).shape

    if not label_counts:
        print("\n⚠️  No .npy sequence files found.")
        return

    print("\n📈 Sequences per gesture:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")
    print(f"   Total: {sum(label_counts.values())}")

    if example_shape is not None:
        print(f"\n🧱 Example sequence shape: {example_shape} (frames, features)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # WLASL-style: one sequence per video → sequences_multi_hand
    existing = SEQUENCE_DIR_MULTI_HAND.exists() and any(SEQUENCE_DIR_MULTI_HAND.glob("*.npy"))
    if existing:
        print(f"\n📄 Sequence data already exists under: {SEQUENCE_DIR_MULTI_HAND}")
        response = input("   Regenerate from videos? (y/n): ").strip().lower()
        if response != "y":
            summarize_sequences(SEQUENCE_DIR_MULTI_HAND)
            if EXTRACTION_ERRORS_FILE.exists():
                print(f"\n⚠️  Previous errors: {EXTRACTION_ERRORS_FILE}")
            return

    seq_dir = process_wlasl_videos()
    summarize_sequences(seq_dir)

    print("\n" + "=" * 60)
    print("🎉 EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"\n30-frame .npy (30×126) → {seq_dir}")
    if EXTRACTION_ERRORS_FILE.exists():
        print(f"Failed videos logged: {EXTRACTION_ERRORS_FILE}")
    print()


if __name__ == "__main__":
    main()
