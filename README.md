# 🤟 SignLens - Real-Time ASL Recognition

A proof-of-concept system for real-time American Sign Language (ASL) recognition using computer vision and machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98.45%25-brightgreen.svg)

## 🎯 Project Goal

Capture webcam video, extract hand landmarks using MediaPipe, and classify them into ASL gestures to display live captions.

### Supported Gestures

| Gesture | Samples | F1-Score |
|---------|---------|----------|
| Hello | 478 | 0.98 |
| Thank You | 491 | 0.98 |
| Sorry | 386 | 1.00 |
| Yes | 137 | 0.97 |
| No | 116 | 0.98 |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SiD-array/SignLens-RealTime-ASL-Recognition.git
cd SignLens-RealTime-ASL-Recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Live Recognition

```bash
python main.py
```

Press **Q** to quit.

## 📁 Project Structure

```
SignLens-RealTime-ASL-Recognition/
├── main.py                    # 🎬 Live webcam recognition app
├── train_model.py             # 🏋️ Model training script
├── extract_landmarks.py       # 📐 Landmark extraction from dataset
├── setup_dataset.py           # 📁 Dataset extraction from ZIPs
├── fix_rotated_frames.py      # 🔧 Fix rotated frames in dataset
├── asl_landmark_detector.py   # 🔍 Video/webcam landmark detector
├── sign_language_model.pkl    # 🤖 Trained Random Forest model
├── hand_landmarker.task       # 🖐️ MediaPipe model (auto-downloaded)
├── requirements.txt           # 📦 Python dependencies
├── README.md                  # 📖 This file
└── data/
    ├── extracted_dynamic/     # Gesture video frames
    │   ├── Hello/
    │   ├── THANKYOU/
    │   ├── Sorry/
    │   ├── Yes/
    │   └── No/
    ├── processed_landmarks/   # Extracted landmark CSV
    │   └── gesture_landmarks.csv
    └── raw_images/            # SignAlphaSet (A-Z letters)
```

## 🛠️ Scripts Overview

### `main.py` - Live Recognition
Real-time ASL recognition with webcam.

**Features:**
- Temporal smoothing (7/10 frame agreement)
- Handedness invariance (left/right hand support)
- Clean UI with confidence display
- 640x480 resolution

```bash
python main.py
```

### `train_model.py` - Model Training
Train a Random Forest classifier on extracted landmarks.

```bash
python train_model.py
```

**Output:** `sign_language_model.pkl`

### `extract_landmarks.py` - Landmark Extraction
Extract normalized hand landmarks from gesture images/videos.

```bash
python extract_landmarks.py
```

**Output:** `data/processed_landmarks/gesture_landmarks.csv`

### `setup_dataset.py` - Dataset Setup
Extract gesture data from ZIP files.

```bash
python setup_dataset.py
```

**Requires:** `ASL_dynamic.zip` and `SignAlphaSet.zip` in project root.

### `fix_rotated_frames.py` - Fix Rotated Frames
Correct orientation of misaligned frames in dataset.

```bash
python fix_rotated_frames.py
```

### `asl_landmark_detector.py` - Video Landmark Detector
Process video files or YouTube URLs for landmark visualization.

```bash
# Webcam
python asl_landmark_detector.py

# Video file
python asl_landmark_detector.py --video path/to/video.mp4

# YouTube
python asl_landmark_detector.py --youtube "https://youtube.com/..."
```

## 📐 Landmark Normalization

The system extracts 21 hand landmarks and normalizes them for consistent recognition:

### Two-Stage Normalization

**Stage 1: Translation (Wrist-Relative)**
```
x'ᵢ = xᵢ - x_wrist
y'ᵢ = yᵢ - y_wrist
z'ᵢ = zᵢ - z_wrist
```

**Stage 2: Scale Normalization**
```
scale = ||landmark₉ - landmark₀||
normalized = translated / scale
```

### Handedness Invariance

The model supports both left and right hands by mirroring x-coordinates when needed:

```python
if detected_hand != MODEL_TRAINED_HAND:
    normalized[:, 0] = -normalized[:, 0]  # Mirror x-coords
```

## ⚠️ Known Limitations

### Motion-Based Gestures

Some ASL signs (like **"Yes"** and **"No"**) differ primarily by **motion**, not hand shape:
- **Yes** = Closed fist + nodding motion
- **No** = Closed fist + side-to-side motion

Since the current model uses single-frame landmarks (static shape), it may confuse motion-based gestures.

### Solutions
1. **Use distinct hand shapes** for Yes/No (recommended for PoC)
2. **Implement sequence modeling** with LSTM/RNN for temporal patterns

## 🗺️ Roadmap

- [x] Hand landmark extraction with MediaPipe
- [x] Coordinate normalization (translation + scale)
- [x] Multi-source video input (webcam, file, YouTube)
- [x] Dataset extraction and preprocessing
- [x] Random Forest classifier training
- [x] Real-time inference with temporal smoothing
- [x] Handedness invariance (left/right hand support)
- [x] Clean UI with confidence display
- [ ] LSTM/RNN for motion-based gesture recognition
- [ ] More gesture classes
- [ ] Mobile deployment

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Video capture and processing |
| mediapipe | ≥0.10.0 | Hand landmark detection |
| numpy | ≥1.24.0 | Numerical operations |
| pandas | ≥2.0.0 | Data manipulation |
| scikit-learn | ≥1.3.0 | Machine learning |
| yt-dlp | ≥2024.1.0 | YouTube video download |

## 🧪 Model Performance

```
Overall Accuracy: 98.45%

              precision    recall  f1-score   support
       Hello       0.97      0.99      0.98        96
          No       1.00      0.96      0.98        23
       Sorry       1.00      1.00      1.00        77
    THANKYOU       1.00      0.97      0.98        98
         Yes       0.93      1.00      0.97        28

    accuracy                           0.98       322
```

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

Built with ❤️ using MediaPipe and scikit-learn
