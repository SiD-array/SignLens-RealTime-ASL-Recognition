"""
SignLens - Real-Time ASL Recognition
====================================
Live webcam ASL gesture recognition with temporal smoothing.

Usage:
    python main.py

Controls:
    Q - Quit the application
"""

import cv2
import numpy as np
import sys
import io
import json
import time
import urllib.request
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List

import tensorflow as tf

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
MODEL_PATH = PROJECT_ROOT / "lstm_model.keras"
LABELS_PATH = PROJECT_ROOT / "labels.json"
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

# Webcam settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Temporal sequence settings (for LSTM input)
SEQUENCE_LENGTH = 30       # Number of frames in each sequence
FEATURE_SIZE_ONE_HAND = 63 # 21 landmarks × 3 coordinates
# Multi-hand: 2 hands → 126 features (WLASL-ready for signs like 'Book').
# When MULTI_HAND is True, train with train_lstm.py using 126-dim sequences (see train_lstm/extract_landmarks).
MULTI_HAND = True
FEATURE_SIZE = 126 if MULTI_HAND else FEATURE_SIZE_ONE_HAND

# Temporal smoothing (lower = less lag, faster UI updates)
BUFFER_SIZE = 6
MIN_AGREEMENT = 3
CONFIDENCE_THRESHOLD = 0.70

# UI settings
OVERLAY_HEIGHT = 80       # Height of bottom overlay
OVERLAY_ALPHA = 0.7       # Transparency of overlay (0-1)

# Handedness settings
# Set this to the hand the model was primarily trained on
# The script will mirror coordinates for the opposite hand
MODEL_TRAINED_HAND = "Right"  # "Right" or "Left"


# ============================================================================
# MODEL LOADING
# ============================================================================

def download_mediapipe_model():
    """Download MediaPipe hand landmarker model if not present."""
    if MEDIAPIPE_MODEL_PATH.exists():
        return
    
    print("Downloading MediaPipe hand landmarker model...")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    print("Model downloaded.")


def load_labels():
    """Load gesture labels from labels.json (dynamic; add WLASL words without code changes)."""
    if not LABELS_PATH.exists():
        print(f"Error: labels.json not found at {LABELS_PATH}")
        print("Create labels.json (a JSON list of label strings) before running this script.")
        sys.exit(1)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        print('Error: labels.json must be a JSON list of label strings, e.g. ["Hello", "No", ...].')
        sys.exit(1)

    return labels


def load_lstm_model():
    """Load the trained LSTM and label set. Categories come from labels.json only (WLASL-ready)."""
    if not MODEL_PATH.exists():
        print(f"Error: LSTM model not found at {MODEL_PATH}")
        print("Run train_lstm.py first to train the action recognition model.")
        sys.exit(1)

    labels = load_labels()
    model = tf.keras.models.load_model(MODEL_PATH)

    # Ensure model expects current feature size (63 one-hand vs 126 two-hand)
    expected_shape = (None, SEQUENCE_LENGTH, FEATURE_SIZE)
    if model.input_shape[1:] != (SEQUENCE_LENGTH, FEATURE_SIZE):
        print(f"Error: Model input shape is {model.input_shape[1:]}, but main.py expects (30, {FEATURE_SIZE}).")
        print("Retrain with train_lstm.py using the same MULTI_HAND / FEATURE_SIZE setup.")
        sys.exit(1)

    print(f"Loaded LSTM model; classes from labels.json: {labels}")
    return model, labels


def create_hand_landmarker():
    """Create MediaPipe HandLandmarker for real-time detection."""
    download_mediapipe_model()
    
    base_options = python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,  # Multi-hand for WLASL (e.g. two-hand signs like 'Book')
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    
    return vision.HandLandmarker.create_from_options(options)


# ============================================================================
# LANDMARK PROCESSING
# ============================================================================

def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks relative to wrist (landmark 0).
    
    - Translation: Subtract wrist position
    - Scale: Divide by wrist-to-middle-MCP distance
    """
    wrist = coords[0]
    translated = coords - wrist
    
    middle_mcp = translated[9]
    scale = np.linalg.norm(middle_mcp)
    
    if scale < 1e-6:
        scale = 1.0
    
    return translated / scale


def extract_landmarks(hand_landmarks, handedness: str, is_selfie_view: bool = True) -> Optional[np.ndarray]:
    """
    Extract and normalize landmarks from MediaPipe detection.
    
    Includes handedness invariance: if the detected hand is opposite to what
    the model was trained on, we mirror the x-coordinates to make the 
    landmark geometry consistent.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        handedness: "Left" or "Right" as reported by MediaPipe
        is_selfie_view: True if frame was horizontally flipped (mirror mode)
    
    Returns:
        Flattened array of 63 values or None if extraction fails
    """
    if not hand_landmarks:
        return None
    
    # Extract raw coordinates
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # Normalize (wrist-relative + scale)
    normalized = normalize_landmarks(raw_coords)
    
    # =========================================================================
    # HANDEDNESS INVARIANCE
    # =========================================================================
    # 
    # In selfie/mirror view (cv2.flip), MediaPipe reports:
    #   - "Left" when the user shows their LEFT hand (appears on RIGHT of screen)
    #   - "Right" when the user shows their RIGHT hand (appears on LEFT of screen)
    #
    # If the model was trained on right-hand data, left-hand landmarks have
    # mirrored geometry. To fix this, we flip the x-coordinates:
    #
    #   x_mirrored = -x_normalized
    #
    # This transforms left-hand geometry to match right-hand geometry.
    # =========================================================================
    
    # Determine if we need to mirror based on handedness
    # In selfie view, MediaPipe's reported handedness matches the user's actual hand
    detected_hand = handedness
    
    # Mirror if detected hand differs from model's training hand
    needs_mirror = (detected_hand != MODEL_TRAINED_HAND)
    
    if needs_mirror:
        # Mirror x-coordinates (multiply by -1)
        # normalized shape is (21, 3) where columns are [x, y, z]
        normalized[:, 0] = -normalized[:, 0]
    
    # Flatten to 1D array (63 values)
    return normalized.flatten()


def _one_hand_to_63(hand_landmarks, handedness: str) -> np.ndarray:
    """Single-hand landmarks: normalize, mirror, return 63-dim."""
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    normalized = normalize_landmarks(raw)
    if handedness != MODEL_TRAINED_HAND:
        normalized[:, 0] = -normalized[:, 0]
    return normalized.flatten().astype(np.float32)


def extract_landmarks_multi_hand(results) -> Optional[np.ndarray]:
    """
    Extract features for 0, 1, or 2 hands. Always returns 126-dim for WLASL (two-hand signs).
    Canonical order: [Left_63, Right_63]. One hand: other half zero-padded.
    """
    if not results.hand_landmarks:
        return None

    hands: List[Tuple[str, np.ndarray]] = []
    for i, hand_landmarks in enumerate(results.hand_landmarks):
        handedness = "Right"
        if results.handedness and i < len(results.handedness) and results.handedness[i]:
            handedness = results.handedness[i][0].category_name
        vec = _one_hand_to_63(hand_landmarks, handedness)
        hands.append((handedness, vec))

    # Canonical order: Left first, Right second (for consistent model input)
    hands.sort(key=lambda x: (0 if x[0] == "Left" else 1, x[0]))

    if len(hands) == 2:
        return np.concatenate([hands[0][1], hands[1][1]], axis=0)  # (126,)
    if len(hands) == 1:
        return np.concatenate([hands[0][1], np.zeros(FEATURE_SIZE_ONE_HAND, dtype=np.float32)], axis=0)  # (126,)
    return None


# ============================================================================
# TEMPORAL SMOOTHING
# ============================================================================

class PredictionBuffer:
    """
    Maintains a buffer of recent predictions for temporal smoothing.
    
    Only updates the displayed gesture if the same prediction appears
    in at least MIN_AGREEMENT out of BUFFER_SIZE recent frames.
    """
    
    def __init__(self, buffer_size: int = BUFFER_SIZE, min_agreement: int = MIN_AGREEMENT):
        self.buffer_size = buffer_size
        self.min_agreement = min_agreement
        self.predictions = deque(maxlen=buffer_size)
        self.confidences = deque(maxlen=buffer_size)
        self.current_gesture = None
        self.current_confidence = 0.0
    
    def add_prediction(self, gesture: str, confidence: float):
        """Add a new prediction to the buffer."""
        self.predictions.append(gesture)
        self.confidences.append(confidence)
        self._update_stable_prediction()
    
    def _update_stable_prediction(self):
        """Update the stable prediction based on buffer contents."""
        if len(self.predictions) < self.min_agreement:
            return
        
        # Count occurrences of each gesture
        gesture_counts = {}
        gesture_confidences = {}
        
        for gesture, conf in zip(self.predictions, self.confidences):
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
                gesture_confidences[gesture] = []
            gesture_counts[gesture] += 1
            gesture_confidences[gesture].append(conf)
        
        # Find the most common gesture
        most_common = max(gesture_counts, key=gesture_counts.get)
        count = gesture_counts[most_common]
        
        # Only update if it meets the agreement threshold
        if count >= self.min_agreement:
            avg_confidence = np.mean(gesture_confidences[most_common])
            self.current_gesture = most_common
            self.current_confidence = avg_confidence
    
    def get_stable_prediction(self) -> Tuple[Optional[str], float]:
        """
        Get the current stable prediction.
        
        Returns:
            (gesture_name, confidence) or (None, 0.0) if no stable prediction
        """
        if self.current_confidence < CONFIDENCE_THRESHOLD:
            return None, self.current_confidence
        return self.current_gesture, self.current_confidence
    
    def clear(self):
        """Clear the prediction buffer."""
        self.predictions.clear()
        self.confidences.clear()
        self.current_gesture = None
        self.current_confidence = 0.0


# ============================================================================
# UI DRAWING
# ============================================================================

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks, width: int, height: int):
    """Draw hand landmarks and connections on the frame."""
    if not hand_landmarks:
        return
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for lm in hand_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))
    
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    
    # Draw landmarks
    for i, (x, y) in enumerate(points):
        color = (255, 0, 0) if i == 0 else (0, 255, 255)
        cv2.circle(frame, (x, y), 5, color, -1)


def draw_overlay(
    frame: np.ndarray,
    gesture: Optional[str],
    confidence: float,
    hand_detected: bool,
    handedness: Optional[str] = None,
    fps: Optional[float] = None,
):
    """
    Draw overlay: gesture, confidence, hand indicator, handedness, optional FPS.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create overlay region
    overlay = frame.copy()
    
    # Draw semi-transparent black rectangle at bottom
    cv2.rectangle(overlay, (0, h - OVERLAY_HEIGHT), (w, h), (0, 0, 0), -1)
    
    # Blend overlay with original frame
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
    
    # Determine display text
    if gesture and confidence >= CONFIDENCE_THRESHOLD:
        display_text = gesture
        confidence_text = f"{confidence * 100:.0f}%"
        text_color = (255, 255, 255)  # White
    elif hand_detected:
        display_text = "Recognizing..."
        confidence_text = f"{confidence * 100:.0f}%" if confidence > 0 else ""
        text_color = (200, 200, 200)  # Light gray
    else:
        display_text = "Show your hand"
        confidence_text = ""
        text_color = (150, 150, 150)  # Gray
    
    # Draw gesture text (large, bold)
    font_scale = 1.5
    thickness = 3
    
    # Get text size for centering
    (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = h - OVERLAY_HEIGHT // 2 + text_h // 4
    
    # Draw text shadow for better visibility
    cv2.putText(frame, display_text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, display_text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # Draw confidence text (smaller, right side)
    if confidence_text:
        conf_font_scale = 0.8
        conf_thickness = 2
        cv2.putText(frame, confidence_text, (w - 80, h - OVERLAY_HEIGHT // 2 + 10), 
                    font, conf_font_scale, (100, 255, 100), conf_thickness)
    
    # Draw hand detection indicator
    indicator_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.circle(frame, (30, h - OVERLAY_HEIGHT // 2), 10, indicator_color, -1)
    
    # Draw "LIVE" indicator
    cv2.putText(frame, "LIVE", (10, 30), font, 0.6, (0, 0, 255), 2)
    cv2.circle(frame, (65, 25), 5, (0, 0, 255), -1)

    # FPS counter (top-left, below LIVE)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 58), font, 0.55, (200, 200, 200), 2)

    # Handedness in top-right
    if handedness:
        hand_text = f"Hand: {handedness}"
        # Color code: Green if matches model, Yellow if mirrored
        is_mirrored = (handedness != MODEL_TRAINED_HAND)
        hand_color = (0, 255, 255) if is_mirrored else (0, 255, 0)  # Yellow if mirrored, Green if native
        
        cv2.putText(frame, hand_text, (w - 130, 30), font, 0.6, (0, 0, 0), 3)  # Shadow
        cv2.putText(frame, hand_text, (w - 130, 30), font, 0.6, hand_color, 2)
        
        # Show if mirroring is applied
        if is_mirrored:
            cv2.putText(frame, "(mirrored)", (w - 130, 55), font, 0.5, (0, 200, 255), 1)
    else:
        cv2.putText(frame, "Hand: None", (w - 130, 30), font, 0.6, (100, 100, 100), 2)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    print("\n" + "=" * 50)
    print("   SignLens - Real-Time ASL Recognition")
    print("=" * 50)
    
    # Initialize components
    print("\nInitializing...")
    model, labels = load_lstm_model()
    landmarker = create_hand_landmarker()
    prediction_buffer = PredictionBuffer()
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Initialize webcam
    print(f"Opening webcam (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("\nPress 'Q' to quit")
    print("=" * 50 + "\n")

    # FPS profiling
    t_prev = time.perf_counter()
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break

            # Flip frame horizontally (mirror/selfie effect)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hands (up to 2 for multi-hand / WLASL)
            results = landmarker.detect(mp_image)

            hand_detected = bool(results.hand_landmarks)
            current_confidence = 0.0
            detected_handedness = None

            # Draw all detected hands
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks, w, h)
                # Handedness for UI: show SCREEN left/right, not anatomical.
                # In selfie view, anatomical Left appears on the RIGHT side of the screen.
                # So we invert labels for display only:
                #   anatomical Left  -> "R" on screen
                #   anatomical Right -> "L" on screen
                if results.handedness and len(results.handedness) >= 1:
                    names = [results.handedness[i][0].category_name for i in range(len(results.handedness))]
                    # Sort by anatomical Left/Right for stable ordering, then map to screen labels
                    sorted_names = sorted(names, key=lambda x: (0 if x == "Left" else 1))
                    screen_parts = ["R" if n == "Left" else "L" for n in sorted_names]
                    detected_handedness = "+".join(screen_parts)
                else:
                    detected_handedness = "?"

            # Feature extraction: multi-hand (126-dim) or single-hand (63-dim)
            if MULTI_HAND:
                features = extract_landmarks_multi_hand(results)
            else:
                features = None
                if results.hand_landmarks and results.handedness and len(results.handedness) > 0:
                    mp_handedness = results.handedness[0][0].category_name
                    features = extract_landmarks(
                        results.hand_landmarks[0],
                        handedness=mp_handedness,
                        is_selfie_view=True,
                    )

            if features is not None:
                sequence_buffer.append(features.astype(np.float32))

                # Sliding-window inference (step=1): predict every frame once buffer is full
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence_array = np.array(sequence_buffer, dtype=np.float32).reshape(
                        1, SEQUENCE_LENGTH, FEATURE_SIZE
                    )
                    proba = model.predict(sequence_array, verbose=0)[0]
                    predicted_idx = int(np.argmax(proba))

                    if 0 <= predicted_idx < len(labels):
                        predicted_gesture = labels[predicted_idx]
                    else:
                        predicted_gesture = str(predicted_idx)

                    current_confidence = float(proba[predicted_idx])
                    prediction_buffer.add_prediction(predicted_gesture, current_confidence)
            else:
                sequence_buffer.clear()

            # FPS (exponential moving average)
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            stable_gesture, stable_confidence = prediction_buffer.get_stable_prediction()
            draw_overlay(frame, stable_gesture, stable_confidence, hand_detected, detected_handedness, fps=fps_smooth)

            cv2.imshow("SignLens - ASL Recognition", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    finally:
        # Cleanup
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()
