"""
ASL Dataset Setup Script
========================
Sets up the project directory structure and extracts gesture datasets.

Usage:
    python setup_dataset.py
"""

import zipfile
import sys
import io
from pathlib import Path

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Project root (where this script is located)
PROJECT_ROOT = Path(__file__).parent

# Data directory structure
DATA_DIR = PROJECT_ROOT / "data"
RAW_ZIPS_DIR = DATA_DIR / "raw_zips"
EXTRACTED_DYNAMIC_DIR = DATA_DIR / "extracted_dynamic"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
PROCESSED_LANDMARKS_DIR = DATA_DIR / "processed_landmarks"

# Source zip files (expected in project root)
SIGN_ALPHA_ZIP = PROJECT_ROOT / "SignAlphaSet.zip"
ASL_DYNAMIC_ZIP = PROJECT_ROOT / "ASL_dynamic.zip"

# Target gestures to extract from ASL_dynamic.zip
# Note: Folder names in ZIP may differ (e.g., THANKYOU instead of "Thank You")
TARGET_GESTURES = ['Hello', 'THANKYOU', 'Sorry', 'Yes', 'No']


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def create_directory_structure():
    """
    Create the data directory structure.
    Safely creates directories only if they don't exist.
    """
    print("\n" + "=" * 60)
    print("📁 CREATING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    directories = [
        DATA_DIR,
        RAW_ZIPS_DIR,
        EXTRACTED_DYNAMIC_DIR,
        RAW_IMAGES_DIR,
        PROCESSED_LANDMARKS_DIR
    ]
    
    for directory in directories:
        if directory.exists():
            print(f"  ✓ Already exists: {directory.relative_to(PROJECT_ROOT)}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {directory.relative_to(PROJECT_ROOT)}")
    
    print()


# ============================================================================
# ZIP EXTRACTION
# ============================================================================

def extract_dynamic_gestures():
    """
    Extract only the target gesture folders from ASL_dynamic.zip.
    """
    print("=" * 60)
    print("📦 EXTRACTING DYNAMIC GESTURES")
    print("=" * 60)
    
    if not ASL_DYNAMIC_ZIP.exists():
        print(f"  ⚠️  ZIP not found: {ASL_DYNAMIC_ZIP.name}")
        print(f"     Expected location: {ASL_DYNAMIC_ZIP}")
        print(f"     Skipping dynamic gesture extraction.\n")
        return False
    
    print(f"  📂 Source: {ASL_DYNAMIC_ZIP.name}")
    print(f"  🎯 Target gestures: {', '.join(TARGET_GESTURES)}")
    print()
    
    extracted_count = 0
    
    try:
        with zipfile.ZipFile(ASL_DYNAMIC_ZIP, 'r') as zf:
            # Get all file names in the zip
            all_files = zf.namelist()
            
            for gesture in TARGET_GESTURES:
                # Find files that belong to this gesture folder
                # Handle various possible folder structures
                gesture_files = [
                    f for f in all_files 
                    if f.startswith(f"{gesture}/") or 
                       f.startswith(f"ASL_dynamic/{gesture}/") or
                       f"/{gesture}/" in f
                ]
                
                if not gesture_files:
                    # Try case-insensitive match
                    gesture_lower = gesture.lower()
                    gesture_files = [
                        f for f in all_files
                        if gesture_lower in f.lower()
                    ]
                
                if gesture_files:
                    gesture_dir = EXTRACTED_DYNAMIC_DIR / gesture
                    
                    if gesture_dir.exists() and any(gesture_dir.iterdir()):
                        print(f"  ✓ '{gesture}' already extracted ({len(list(gesture_dir.rglob('*')))} files)")
                        extracted_count += 1
                        continue
                    
                    gesture_dir.mkdir(parents=True, exist_ok=True)
                    
                    file_count = 0
                    for file_path in gesture_files:
                        # Skip directory entries
                        if file_path.endswith('/'):
                            continue
                        
                        # Extract to gesture folder, preserving internal structure
                        # Get the relative path after the gesture folder name
                        parts = Path(file_path).parts
                        
                        # Find where the gesture name appears and get everything after
                        for i, part in enumerate(parts):
                            if part.lower() == gesture.lower() or part == gesture:
                                relative_path = Path(*parts[i+1:]) if i+1 < len(parts) else Path(parts[-1])
                                break
                        else:
                            relative_path = Path(parts[-1])
                        
                        target_path = gesture_dir / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Extract the file
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
    """
    Extract the full contents of SignAlphaSet.zip.
    """
    print("=" * 60)
    print("📦 EXTRACTING SIGN ALPHA SET")
    print("=" * 60)
    
    if not SIGN_ALPHA_ZIP.exists():
        print(f"  ⚠️  ZIP not found: {SIGN_ALPHA_ZIP.name}")
        print(f"     Expected location: {SIGN_ALPHA_ZIP}")
        print(f"     Skipping SignAlphaSet extraction.\n")
        return False
    
    print(f"  📂 Source: {SIGN_ALPHA_ZIP.name}")
    print(f"  📂 Destination: {RAW_IMAGES_DIR.relative_to(PROJECT_ROOT)}")
    print()
    
    # Check if already extracted
    if RAW_IMAGES_DIR.exists() and any(RAW_IMAGES_DIR.iterdir()):
        file_count = len(list(RAW_IMAGES_DIR.rglob('*')))
        print(f"  ✓ Already extracted ({file_count} files/folders)")
        print()
        return True
    
    try:
        with zipfile.ZipFile(SIGN_ALPHA_ZIP, 'r') as zf:
            total_files = len([f for f in zf.namelist() if not f.endswith('/')])
            print(f"  📊 Extracting {total_files} files...")
            
            zf.extractall(RAW_IMAGES_DIR)
            
            print(f"  ✅ Extraction complete!")
        
        print()
        return True
        
    except zipfile.BadZipFile:
        print(f"  ❌ Error: {SIGN_ALPHA_ZIP.name} is not a valid ZIP file\n")
        return False


# ============================================================================
# DIRECTORY TREE VISUALIZATION
# ============================================================================

def print_directory_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """
    Print a tree-like representation of a directory structure.
    
    Args:
        directory: Path to directory to visualize
        prefix: Current line prefix for tree structure
        max_depth: Maximum depth to traverse
        current_depth: Current recursion depth
    """
    if current_depth >= max_depth:
        return
    
    # Get sorted list of items
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return
    
    # Count items for proper connector drawing
    items_list = list(items)
    
    for i, item in enumerate(items_list):
        is_last = (i == len(items_list) - 1)
        connector = "└── " if is_last else "├── "
        
        if item.is_dir():
            # Count items in this directory
            try:
                child_count = len(list(item.iterdir()))
                suffix = f" ({child_count} items)" if child_count > 0 else " (empty)"
            except PermissionError:
                suffix = " (no access)"
            
            print(f"{prefix}{connector}📁 {item.name}{suffix}")
            
            # Recurse into directory
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_directory_tree(item, new_prefix, max_depth, current_depth + 1)
        else:
            # File
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            
            print(f"{prefix}{connector}📄 {item.name} ({size_str})")


def print_data_summary():
    """
    Print a summary tree of the data directory.
    """
    print("=" * 60)
    print("🌳 DATA DIRECTORY STRUCTURE")
    print("=" * 60)
    print()
    
    if not DATA_DIR.exists():
        print("  ❌ Data directory does not exist!")
        return
    
    print(f"📁 {DATA_DIR.relative_to(PROJECT_ROOT)}/")
    print_directory_tree(DATA_DIR, "")
    print()
    
    # Print statistics
    print("=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)
    
    for subdir in [RAW_ZIPS_DIR, EXTRACTED_DYNAMIC_DIR, RAW_IMAGES_DIR, PROCESSED_LANDMARKS_DIR]:
        if subdir.exists():
            files = list(subdir.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            dir_count = len([f for f in files if f.is_dir()])
            print(f"  {subdir.name}/: {file_count} files, {dir_count} folders")
        else:
            print(f"  {subdir.name}/: (not created)")
    
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("   🤟 ASL DATASET SETUP")
    print("=" * 60)
    print(f"\n  Project root: {PROJECT_ROOT}\n")
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Extract dynamic gestures
    extract_dynamic_gestures()
    
    # Step 3: Extract SignAlphaSet
    extract_sign_alpha_set()
    
    # Step 4: Print summary
    print_data_summary()
    
    print("✅ Dataset setup complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
