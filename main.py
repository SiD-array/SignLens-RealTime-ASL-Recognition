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
import pickle
import sys
import io
import urllib.request
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

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
MODEL_PATH = PROJECT_ROOT / "sign_language_model.pkl"
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

# Webcam settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Temporal smoothing settings
BUFFER_SIZE = 10          # Number of frames to consider
MIN_AGREEMENT = 7         # Minimum frames with same prediction
CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence to display gesture

# UI settings
OVERLAY_HEIGHT = 80       # Height of bottom overlay
OVERLAY_ALPHA = 0.7       # Transparency of overlay (0-1)


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


def load_classifier():
    """Load the trained gesture classification model."""
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run train_model.py first to train the classifier.")
        sys.exit(1)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded classifier with classes: {list(model.classes_)}")
    return model


def create_hand_landmarker():
    """Create MediaPipe HandLandmarker for real-time detection."""
    download_mediapipe_model()
    
    base_options = python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
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


def extract_landmarks(hand_landmarks) -> Optional[np.ndarray]:
    """
    Extract and normalize landmarks from MediaPipe detection.
    
    Returns:
        Flattened array of 63 values or None if extraction fails
    """
    if not hand_landmarks:
        return None
    
    # Extract raw coordinates
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # Normalize
    normalized = normalize_landmarks(raw_coords)
    
    # Flatten to 1D array (63 values)
    return normalized.flatten()


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


def draw_overlay(frame: np.ndarray, gesture: Optional[str], confidence: float, hand_detected: bool):
    """
    Draw a semi-transparent overlay at the bottom with gesture info.
    
    Args:
        frame: The video frame to draw on
        gesture: The recognized gesture name (or None)
        confidence: The prediction confidence (0-1)
        hand_detected: Whether a hand was detected in the current frame
    """
    h, w = frame.shape[:2]
    
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
    font = cv2.FONT_HERSHEY_SIMPLEX
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


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    print("\n" + "=" * 50)
    print("   SignLens - Real-Time ASL Recognition")
    print("=" * 50)
    
    # Initialize components
    print("\nInitializing...")
    classifier = load_classifier()
    landmarker = create_hand_landmarker()
    prediction_buffer = PredictionBuffer()
    
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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            results = landmarker.detect(mp_image)
            
            hand_detected = bool(results.hand_landmarks)
            current_confidence = 0.0
            
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                
                # Draw hand landmarks
                draw_hand_landmarks(frame, hand_landmarks, w, h)
                
                # Extract and normalize landmarks
                features = extract_landmarks(hand_landmarks)
                
                if features is not None:
                    # Get prediction probabilities
                    proba = classifier.predict_proba([features])[0]
                    predicted_idx = np.argmax(proba)
                    predicted_gesture = classifier.classes_[predicted_idx]
                    current_confidence = proba[predicted_idx]
                    
                    # Add to buffer for temporal smoothing
                    prediction_buffer.add_prediction(predicted_gesture, current_confidence)
            
            # Get stable prediction from buffer
            stable_gesture, stable_confidence = prediction_buffer.get_stable_prediction()
            
            # Draw UI overlay
            draw_overlay(frame, stable_gesture, stable_confidence, hand_detected)
            
            # Display frame
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
