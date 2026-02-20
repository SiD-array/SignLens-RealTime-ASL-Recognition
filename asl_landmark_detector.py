"""
ASL Landmark Detector - Proof of Concept
=========================================
Extracts hand landmarks from video using MediaPipe for ASL gesture recognition.

Usage:
    python asl_landmark_detector.py                     # Use webcam
    python asl_landmark_detector.py --video path.mp4   # Use local video file
    python asl_landmark_detector.py --youtube URL      # Download and use YouTube video
"""

import cv2
import numpy as np
import argparse
import subprocess
import os
import sys
import io
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# MediaPipe Tasks API (new API for MediaPipe 0.10.30+)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# MediaPipe landmark names
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)            # Palm
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HandLandmarks:
    """Container for raw and normalized hand landmark data."""
    raw_coords: np.ndarray          # Shape: (21, 3) - original (x, y, z)
    normalized_coords: np.ndarray   # Shape: (21, 3) - wrist-relative normalized
    handedness: str                 # "Left" or "Right"


# ============================================================================
# MODEL DOWNLOAD
# ============================================================================

def download_model():
    """Download the MediaPipe hand landmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists: {MODEL_PATH}")
        return
    
    print(f"⬇️  Downloading hand landmarker model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✅ Model downloaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)


# ============================================================================
# VIDEO SOURCE CONFIGURATION (Modular Design)
# ============================================================================

def get_video_source(
    webcam_index: int = 0,
    video_path: Optional[str] = None,
    youtube_url: Optional[str] = None
) -> cv2.VideoCapture:
    """
    Create a video capture object from various sources.
    
    MODULAR DESIGN: To swap between sources, just change the arguments:
        - Webcam:  get_video_source(webcam_index=0)
        - File:    get_video_source(video_path="path/to/video.mp4")
        - YouTube: get_video_source(youtube_url="https://youtube.com/...")
    
    Args:
        webcam_index: Camera device index (default 0 for primary webcam)
        video_path: Path to local video file
        youtube_url: YouTube URL to download and process
    
    Returns:
        cv2.VideoCapture object ready for frame extraction
    """
    if youtube_url:
        video_path = download_youtube_video(youtube_url)
        if not video_path:
            print("Failed to download YouTube video. Falling back to webcam.")
            return cv2.VideoCapture(webcam_index)
    
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        print(f"📹 Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print(f"📷 Opening webcam (index {webcam_index})")
        cap = cv2.VideoCapture(webcam_index)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")
    
    return cap


def download_youtube_video(url: str, output_dir: str = ".") -> Optional[str]:
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded video
    
    Returns:
        Path to downloaded video file, or None if failed
    """
    output_template = os.path.join(output_dir, "youtube_video.%(ext)s")
    output_file = os.path.join(output_dir, "youtube_video.mp4")
    
    # Remove existing file if present
    if os.path.exists(output_file):
        print(f"Using cached video: {output_file}")
        return output_file
    
    print(f"⬇️  Downloading video from YouTube...")
    print(f"   URL: {url}")
    
    try:
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",  # Prefer MP4 format
            "-o", output_template,
            "--no-playlist",              # Don't download playlists
            url
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Find the downloaded file (extension might vary)
        for ext in ["mp4", "webm", "mkv"]:
            potential_file = os.path.join(output_dir, f"youtube_video.{ext}")
            if os.path.exists(potential_file):
                print(f"✅ Download complete: {potential_file}")
                return potential_file
        
        print("❌ Download completed but file not found")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ yt-dlp failed: {e.stderr}")
        return None
    except FileNotFoundError:
        print("❌ yt-dlp not found. Install it with: pip install yt-dlp")
        return None


# ============================================================================
# LANDMARK EXTRACTION & NORMALIZATION
# ============================================================================

def extract_landmarks(
    hand_landmarks,
    handedness: str,
    image_width: int,
    image_height: int
) -> HandLandmarks:
    """
    Extract raw and normalized coordinates from MediaPipe hand landmarks.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks (list of NormalizedLandmark)
        handedness: "Left" or "Right"
        image_width: Frame width in pixels
        image_height: Frame height in pixels
    
    Returns:
        HandLandmarks dataclass with raw and normalized coordinates
    """
    # Extract raw (x, y, z) for all 21 landmarks
    raw_coords = np.array([
        [lm.x, lm.y, lm.z] for lm in hand_landmarks
    ])
    
    # Normalize coordinates
    normalized_coords = normalize_landmarks(raw_coords)
    
    return HandLandmarks(
        raw_coords=raw_coords,
        normalized_coords=normalized_coords,
        handedness=handedness
    )


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks relative to the wrist (landmark 0).
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  NORMALIZATION MATH EXPLAINED                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  STEP 1: TRANSLATION NORMALIZATION (Wrist-Relative)                     │
    │  ─────────────────────────────────────────────────────                  │
    │  For each landmark i with coordinates (x_i, y_i, z_i):                  │
    │                                                                         │
    │      x'_i = x_i - x_wrist                                               │
    │      y'_i = y_i - y_wrist                                               │
    │      z'_i = z_i - z_wrist                                               │
    │                                                                         │
    │  This makes the wrist the origin (0, 0, 0), removing the effect of      │
    │  where the hand appears in the frame. The gesture "Hello" looks the     │
    │  same whether your hand is on the left or right side of the screen.     │
    │                                                                         │
    │  STEP 2: SCALE NORMALIZATION                                            │
    │  ─────────────────────────────                                          │
    │  We divide by the distance from wrist to middle finger MCP joint        │
    │  (landmark 9), which is a stable reference distance:                    │
    │                                                                         │
    │      scale = ||landmark_9 - landmark_0|| = sqrt(Σ(coord_9 - coord_0)²)  │
    │                                                                         │
    │      x''_i = x'_i / scale                                               │
    │      y''_i = y'_i / scale                                               │
    │      z''_i = z'_i / scale                                               │
    │                                                                         │
    │  This makes the data scale-invariant: a hand close to the camera        │
    │  (appearing larger) produces the same normalized values as a hand       │
    │  farther away (appearing smaller).                                      │
    │                                                                         │
    │  WHY WRIST + MIDDLE MCP?                                                │
    │  • Wrist (landmark 0): Stable base, doesn't move with finger gestures   │
    │  • Middle MCP (landmark 9): Central point, provides consistent scale    │
    │                                                                         │
    │  RESULT: Translation + Scale invariant features for ML classification   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Args:
        coords: Raw coordinates array of shape (21, 3)
    
    Returns:
        Normalized coordinates array of shape (21, 3)
    """
    # Step 1: Translate so wrist (landmark 0) is at origin
    wrist = coords[0]  # (x, y, z) of wrist
    translated = coords - wrist  # Subtract wrist from all landmarks
    
    # Step 2: Scale by distance from wrist to middle finger MCP (landmark 9)
    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)  # Euclidean distance
    
    # Avoid division by zero (if hand detection is poor)
    if scale < 1e-6:
        scale = 1.0
    
    normalized = translated / scale
    
    return normalized


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_landmarks_on_frame(
    frame: np.ndarray,
    hand_landmarks,
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Draw hand landmarks and connections on the frame.
    
    Args:
        frame: BGR image from OpenCV
        hand_landmarks: List of NormalizedLandmark from MediaPipe
        image_width: Frame width in pixels
        image_height: Frame height in pixels
    
    Returns:
        Frame with landmarks drawn
    """
    annotated = frame.copy()
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        points.append((x, y))
    
    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(
            annotated,
            points[start_idx],
            points[end_idx],
            (0, 255, 0),  # Green
            2
        )
    
    # Draw landmarks
    for i, (x, y) in enumerate(points):
        # Different colors for different finger groups
        if i == 0:
            color = (255, 0, 0)     # Blue for wrist
        elif i <= 4:
            color = (255, 128, 0)   # Orange for thumb
        elif i <= 8:
            color = (255, 255, 0)   # Cyan for index
        elif i <= 12:
            color = (0, 255, 0)     # Green for middle
        elif i <= 16:
            color = (0, 255, 255)   # Yellow for ring
        else:
            color = (0, 128, 255)   # Orange for pinky
        
        cv2.circle(annotated, (x, y), 5, color, -1)
        cv2.circle(annotated, (x, y), 7, (255, 255, 255), 1)
    
    return annotated


def print_landmark_data(landmarks: HandLandmarks, frame_num: int):
    """
    Print formatted landmark data to console.
    
    Args:
        landmarks: HandLandmarks object with raw and normalized data
        frame_num: Current frame number
    """
    print("\n" + "=" * 70)
    print(f"🖐️  HAND DETECTED - Frame #{frame_num} ({landmarks.handedness} Hand)")
    print("=" * 70)
    
    print("\n📍 RAW COORDINATES (MediaPipe normalized 0-1):")
    print("-" * 70)
    print(f"{'#':<3} {'Landmark':<14} {'X':>10} {'Y':>10} {'Z':>10}")
    print("-" * 70)
    
    for i, (name, coord) in enumerate(zip(LANDMARK_NAMES, landmarks.raw_coords)):
        print(f"{i:<3} {name:<14} {coord[0]:>10.6f} {coord[1]:>10.6f} {coord[2]:>10.6f}")
    
    print("\n📐 NORMALIZED COORDINATES (Wrist-relative, scale-invariant):")
    print("-" * 70)
    print(f"{'#':<3} {'Landmark':<14} {'X':>10} {'Y':>10} {'Z':>10}")
    print("-" * 70)
    
    for i, (name, coord) in enumerate(zip(LANDMARK_NAMES, landmarks.normalized_coords)):
        print(f"{i:<3} {name:<14} {coord[0]:>10.6f} {coord[1]:>10.6f} {coord[2]:>10.6f}")
    
    print("\n" + "=" * 70)
    print("✅ First hand detection complete! Continuing video processing...")
    print("   Press 'Q' to quit the video window.")
    print("=" * 70 + "\n")


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def process_video(cap: cv2.VideoCapture, print_first_detection: bool = True):
    """
    Main video processing loop with hand landmark detection.
    
    Args:
        cap: OpenCV VideoCapture object
        print_first_detection: If True, print landmarks for first detection only
    """
    # Download model if needed
    download_model()
    
    # Create hand landmarker with new Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    frame_num = 0
    first_detection_done = False
    
    # Get video FPS for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default fallback
    
    print("\n🎬 Starting video processing...")
    print("   Window will open. Press 'Q' to quit.\n")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("📼 End of video or failed to read frame.")
                break
            
            frame_num += 1
            h, w, _ = frame.shape
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calculate timestamp in milliseconds
            timestamp_ms = int((frame_num / fps) * 1000)
            
            # Process frame with MediaPipe
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Create display frame
            display_frame = frame.copy()
            
            if results.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # Get handedness
                    handedness = "Unknown"
                    if results.handedness and hand_idx < len(results.handedness):
                        handedness = results.handedness[hand_idx][0].category_name
                    
                    # Draw landmarks on display frame
                    display_frame = draw_landmarks_on_frame(
                        display_frame,
                        hand_landmarks,
                        w, h
                    )
                    
                    # Extract and print landmarks (first detection only if flag set)
                    if print_first_detection and not first_detection_done:
                        landmarks = extract_landmarks(
                            hand_landmarks, handedness, w, h
                        )
                        print_landmark_data(landmarks, frame_num)
                        first_detection_done = True
            
            # Add frame info overlay
            cv2.putText(
                display_frame,
                f"Frame: {frame_num}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            status = "Hand Detected!" if results.hand_landmarks else "No Hand"
            color = (0, 255, 0) if results.hand_landmarks else (0, 0, 255)
            cv2.putText(
                display_frame,
                status,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
            # Display the frame
            cv2.imshow("ASL Landmark Detection", display_frame)
            
            # Press 'Q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 User quit.")
                break
    
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Processing complete.")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ASL Landmark Detector - Extract hand landmarks from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asl_landmark_detector.py                           # Use webcam
  python asl_landmark_detector.py --video sign_language.mp4 # Use local file
  python asl_landmark_detector.py --youtube URL             # Use YouTube video
  python asl_landmark_detector.py --webcam 1                # Use secondary webcam
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to local video file"
    )
    parser.add_argument(
        "--youtube", "-y",
        type=str,
        help="YouTube URL to download and process"
    )
    parser.add_argument(
        "--webcam", "-w",
        type=int,
        default=0,
        help="Webcam index (default: 0)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("   🤟 ASL LANDMARK DETECTOR - Proof of Concept")
    print("=" * 70)
    
    try:
        # Get video source based on arguments
        cap = get_video_source(
            webcam_index=args.webcam,
            video_path=args.video,
            youtube_url=args.youtube
        )
        
        # Process video
        process_video(cap)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
