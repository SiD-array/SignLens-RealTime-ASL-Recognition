"""
ASL Dataset Setup Script
========================
Sets up the project directory structure and extracts gesture datasets from ZIPs.
Run from project root: python scripts/setup_dataset.py
"""

import zipfile
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ZIPS_DIR = DATA_DIR / "raw_zips"
EXTRACTED_DYNAMIC_DIR = DATA_DIR / "extracted_dynamic"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
PROCESSED_LANDMARKS_DIR = DATA_DIR / "processed_landmarks"
SIGN_ALPHA_ZIP = PROJECT_ROOT / "SignAlphaSet.zip"
ASL_DYNAMIC_ZIP = PROJECT_ROOT / "ASL_dynamic.zip"
TARGET_GESTURES = ['Hello', 'THANKYOU', 'Sorry', 'Yes', 'No']


def create_directory_structure():
    print("\n" + "=" * 60)
    print("📁 CREATING DIRECTORY STRUCTURE")
    print("=" * 60)
    for directory in [DATA_DIR, RAW_ZIPS_DIR, EXTRACTED_DYNAMIC_DIR, RAW_IMAGES_DIR, PROCESSED_LANDMARKS_DIR]:
        if directory.exists():
            print(f"  ✓ Already exists: {directory.relative_to(PROJECT_ROOT)}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {directory.relative_to(PROJECT_ROOT)}")
    print()


def extract_dynamic_gestures():
    print("=" * 60)
    print("📦 EXTRACTING DYNAMIC GESTURES")
    print("=" * 60)
    if not ASL_DYNAMIC_ZIP.exists():
        print(f"  ⚠️  ZIP not found: {ASL_DYNAMIC_ZIP.name}\n")
        return False
    print(f"  📂 Source: {ASL_DYNAMIC_ZIP.name}\n")
    extracted_count = 0
    try:
        with zipfile.ZipFile(ASL_DYNAMIC_ZIP, 'r') as zf:
            all_files = zf.namelist()
            for gesture in TARGET_GESTURES:
                gesture_files = [f for f in all_files if f.startswith(f"{gesture}/") or f.startswith(f"ASL_dynamic/{gesture}/") or f"/{gesture}/" in f]
                if not gesture_files:
                    gesture_files = [f for f in all_files if gesture.lower() in f.lower()]
                if gesture_files:
                    gesture_dir = EXTRACTED_DYNAMIC_DIR / gesture
                    if gesture_dir.exists() and any(gesture_dir.iterdir()):
                        print(f"  ✓ '{gesture}' already extracted")
                        extracted_count += 1
                        continue
                    gesture_dir.mkdir(parents=True, exist_ok=True)
                    file_count = 0
                    for file_path in gesture_files:
                        if file_path.endswith('/'):
                            continue
                        parts = Path(file_path).parts
                        for i, part in enumerate(parts):
                            if part.lower() == gesture.lower() or part == gesture:
                                relative_path = Path(*parts[i+1:]) if i+1 < len(parts) else Path(parts[-1])
                                break
                        else:
                            relative_path = Path(parts[-1])
                        target_path = gesture_dir / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(file_path) as src:
                            target_path.write_bytes(src.read())
                        file_count += 1
                    print(f"  ✅ Extracted '{gesture}' ({file_count} files)")
                    extracted_count += 1
                else:
                    print(f"  ⚠️  '{gesture}' not found in ZIP")
        print(f"\n  📊 Extracted {extracted_count}/{len(TARGET_GESTURES)} gesture folders\n")
        return True
    except zipfile.BadZipFile:
        print(f"  ❌ Error: {ASL_DYNAMIC_ZIP.name} is not a valid ZIP file\n")
        return False


def extract_sign_alpha_set():
    print("=" * 60)
    print("📦 EXTRACTING SIGN ALPHA SET")
    print("=" * 60)
    if not SIGN_ALPHA_ZIP.exists():
        print(f"  ⚠️  ZIP not found: {SIGN_ALPHA_ZIP.name}\n")
        return False
    if RAW_IMAGES_DIR.exists() and any(RAW_IMAGES_DIR.iterdir()):
        print(f"  ✓ Already extracted\n")
        return True
    try:
        with zipfile.ZipFile(SIGN_ALPHA_ZIP, 'r') as zf:
            zf.extractall(RAW_IMAGES_DIR)
        print(f"  ✅ Extraction complete!\n")
        return True
    except zipfile.BadZipFile:
        print(f"  ❌ Error: invalid ZIP\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("   🤟 ASL DATASET SETUP")
    print("=" * 60)
    print(f"\n  Project root: {PROJECT_ROOT}\n")
    create_directory_structure()
    extract_dynamic_gestures()
    extract_sign_alpha_set()
    print("✅ Dataset setup complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
