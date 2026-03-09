"""
Fix Rotated Frames Script
=========================
Rotates incorrectly oriented frames in the dataset.
Run from project root: python scripts/fix_rotated_frames.py
"""

import cv2
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "extracted_dynamic"
ROTATIONS_NEEDED = {"Sorry": cv2.ROTATE_90_COUNTERCLOCKWISE}


def fix_rotated_frames(gesture: str, rotation_code: int):
    gesture_dir = DATA_DIR / gesture
    if not gesture_dir.exists():
        print(f"  ⚠️  Gesture folder not found: {gesture}")
        return 0
    frame_folders = [f for f in gesture_dir.iterdir() if f.is_dir() and "_frames" in f.name]
    total_fixed = 0
    for folder in frame_folders:
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            rotated = cv2.rotate(img, rotation_code)
            cv2.imwrite(str(img_path), rotated)
            total_fixed += 1
        print(f"    ✅ Fixed {len(image_files)} frames in {folder.name}")
    return total_fixed


def main():
    print("\n" + "=" * 60)
    print("   🔧 FIX ROTATED FRAMES")
    print("=" * 60)
    for gesture, rotation_code in ROTATIONS_NEEDED.items():
        print(f"\n📁 Processing '{gesture}'...")
        fixed_count = fix_rotated_frames(gesture, rotation_code)
        print(f"\n   📊 Total frames fixed: {fixed_count}")
    print("\n✅ Done. Next: python extract_landmarks.py → python train_lstm.py → python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
