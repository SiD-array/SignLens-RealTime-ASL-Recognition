# 🤟 SignLens - Real-Time ASL Recognition

A proof-of-concept system for real-time American Sign Language (ASL) recognition using computer vision and machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)

## 🎯 Project Goal

Capture webcam video, extract hand landmarks using MediaPipe, and classify them into ASL gestures (Hello, Yes, No, etc.) to display live captions.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SiD-array/SignLens-RealTime-ASL-Recognition.git
cd SignLens-RealTime-ASL-Recognition

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Use webcam (default)
python asl_landmark_detector.py

# Use a local video file
python asl_landmark_detector.py --video path/to/video.mp4

# Use a YouTube video
python asl_landmark_detector.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
```

Press **Q** to quit the video window.

## 📐 Landmark Normalization

The system extracts 21 hand landmarks and normalizes them for consistent gesture recognition:

### Two-Stage Normalization Process

**Stage 1: Translation (Wrist-Relative)**
```
x'ᵢ = xᵢ - x_wrist
y'ᵢ = yᵢ - y_wrist
z'ᵢ = zᵢ - z_wrist
```
Makes the wrist the origin (0,0,0), removing position dependency.

**Stage 2: Scale Normalization**
```
scale = ||landmark₉ - landmark₀||
normalized = translated / scale
```
Divides by wrist-to-middle-MCP distance, making data scale-invariant.

### MediaPipe Hand Landmarks

```
        THUMB_TIP (4)
            │
        THUMB_IP (3)
            │
       THUMB_MCP (2)
            │
       THUMB_CMC (1)
            │
          WRIST (0) ─── INDEX_MCP (5) ─── MIDDLE_MCP (9) ─── RING_MCP (13) ─── PINKY_MCP (17)
                              │                  │                 │                  │
                        INDEX_PIP (6)      MIDDLE_PIP (10)   RING_PIP (14)     PINKY_PIP (18)
                              │                  │                 │                  │
                        INDEX_DIP (7)      MIDDLE_DIP (11)   RING_DIP (15)     PINKY_DIP (19)
                              │                  │                 │                  │
                        INDEX_TIP (8)      MIDDLE_TIP (12)   RING_TIP (16)     PINKY_TIP (20)
```

## 🛠️ Modular Design

The code is designed for easy source swapping:

```python
from asl_landmark_detector import get_video_source, process_video

# Webcam
cap = get_video_source(webcam_index=0)

# Video file
cap = get_video_source(video_path="my_video.mp4")

# YouTube
cap = get_video_source(youtube_url="https://youtube.com/...")

process_video(cap)
```

## 📁 Project Structure

```
SignLens-RealTime-ASL-Recognition/
├── asl_landmark_detector.py   # Main detection script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## 🗺️ Roadmap

- [x] Hand landmark extraction with MediaPipe
- [x] Coordinate normalization (translation + scale)
- [x] Multi-source video input (webcam, file, YouTube)
- [ ] Gesture dataset collection
- [ ] ML model training for ASL classification
- [ ] Real-time caption overlay
- [ ] Multi-hand gesture support

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Video capture and processing |
| mediapipe | ≥0.10.0 | Hand landmark detection |
| numpy | ≥1.24.0 | Numerical operations |
| yt-dlp | ≥2024.1.0 | YouTube video download |

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.
