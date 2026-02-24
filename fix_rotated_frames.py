"""
Fix Rotated Frames Script
=========================
Rotates incorrectly oriented frames in the dataset.

Usage:
    python fix_rotated_frames.py
"""

import cv2
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "extracted_dynamic"

# Gestures that need rotation fix
# Format: {gesture_name: rotation_code}
# cv2.ROTATE_90_COUNTERCLOCKWISE = rotate left (fixes right-rotated images)
# cv2.ROTATE_90_CLOCKWISE = rotate right
# cv2.ROTATE_180 = flip upside down
ROTATIONS_NEEDED = {
    "Sorry": cv2.ROTATE_90_COUNTERCLOCKWISE,  # Fix right-rotated frames
}


# ============================================================================
# FIX FUNCTIONS
# ============================================================================

def fix_rotated_frames(gesture: str, rotation_code: int):
    """
    Rotate all frames in a gesture folder.
    
    Args:
        gesture: Name of gesture folder
        rotation_code: OpenCV rotation code
    """
    gesture_dir = DATA_DIR / gesture
    
    if not gesture_dir.exists():
        print(f"  ⚠️  Gesture folder not found: {gesture}")
        return 0
    
    # Find all frame folders
    frame_folders = [f for f in gesture_dir.iterdir() if f.is_dir() and "_frames" in f.name]
    
    total_fixed = 0
    
    for folder in frame_folders:
        # Get all image files
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        for img_path in image_files:
            # Read image
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"    ⚠️  Could not read: {img_path.name}")
                continue
            
            # Rotate
            rotated = cv2.rotate(img, rotation_code)
            
            # Save back (overwrite)
            cv2.imwrite(str(img_path), rotated)
            total_fixed += 1
        
        print(f"    ✅ Fixed {len(image_files)} frames in {folder.name}")
    
    return total_fixed


def verify_fix(gesture: str):
    """Show a sample frame to verify the fix worked."""
    gesture_dir = DATA_DIR / gesture
    frame_folders = [f for f in gesture_dir.iterdir() if f.is_dir() and "_frames" in f.name]
    
    if frame_folders:
        sample_folder = frame_folders[0]
        sample_images = list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png"))
        
        if sample_images:
            sample_path = sample_images[0]
            img = cv2.imread(str(sample_path))
            
            if img is not None:
                h, w = img.shape[:2]
                print(f"\n  📐 Sample dimensions after fix: {w}x{h}")
                print(f"     (Should be wider than tall for landscape, taller than wide for portrait)")
                
                # Show preview
                preview = cv2.resize(img, (400, 300))
                cv2.imshow(f"Fixed {gesture} Sample - Press any key", preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("   🔧 FIX ROTATED FRAMES")
    print("=" * 60)
    
    for gesture, rotation_code in ROTATIONS_NEEDED.items():
        print(f"\n📁 Processing '{gesture}'...")
        
        rotation_name = {
            cv2.ROTATE_90_COUNTERCLOCKWISE: "90° counter-clockwise",
            cv2.ROTATE_90_CLOCKWISE: "90° clockwise",
            cv2.ROTATE_180: "180°"
        }.get(rotation_code, "unknown")
        
        print(f"   Rotation: {rotation_name}")
        
        fixed_count = fix_rotated_frames(gesture, rotation_code)
        print(f"\n   📊 Total frames fixed: {fixed_count}")
        
        # Verify
        verify_fix(gesture)
    
    print("\n" + "=" * 60)
    print("✅ FRAME ROTATION FIX COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python extract_landmarks.py")
    print("  2. Run: python train_model.py")
    print("  3. Test with: python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
