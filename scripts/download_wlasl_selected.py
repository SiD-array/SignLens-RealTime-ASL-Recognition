"""
Download selected WLASL gloss videos into this project's structure.

Target output:
  data/extracted_dynamic/{Label}/{video_id}.mp4

This script:
  - Reads WLASL_v0.3.json from wlasl_source/
  - Filters to a small set of glosses (hello, thank you, sorry, yes, no)
  - Optionally clears existing output folders for those labels (replace mode)
  - Downloads direct mp4 via urllib and YouTube via yt-dlp

Usage (from project root):
  python scripts/download_wlasl_selected.py
"""

import json
import shutil
import subprocess
import sys
import time
import random
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WLASL_JSON = PROJECT_ROOT / "wlasl_source" / "start_kit" / "WLASL_v0.3.json"
OUT_ROOT = PROJECT_ROOT / "data" / "extracted_dynamic"

YT_DLP = "yt-dlp"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Map your project labels -> WLASL gloss strings
TARGET = {
    "Hello": "hello",
    "THANKYOU": "thank you",
    "Sorry": "sorry",
    "Yes": "yes",
    "No": "no",
}

# If True, delete existing label folders before downloading
REPLACE_EXISTING = True


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def check_deps():
    try:
        subprocess.run([YT_DLP, "--version"], capture_output=True, check=True, timeout=5)
    except Exception:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)


def request_video(url: str, referer: str = "") -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def download_direct(url: str, save_path: Path) -> bool:
    try:
        data = request_video(url)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(data)
        time.sleep(random.uniform(0.3, 0.9))
        return True
    except Exception as e:
        print(f"Direct download failed: {url} ({e})")
        return False


def download_youtube(url: str, save_path: Path) -> bool:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [YT_DLP, "-q", "--no-warnings", "-f", "best[ext=mp4]/best", "-o", str(save_path), url],
            check=True,
            timeout=180,
            capture_output=True,
        )
        time.sleep(random.uniform(0.5, 1.2))
        return True
    except Exception as e:
        print(f"yt-dlp failed: {url} ({e})")
        return False


def main():
    if not WLASL_JSON.exists():
        print(f"ERROR: {WLASL_JSON} not found. Clone WLASL into wlasl_source/ first.")
        sys.exit(1)

    check_deps()

    with open(WLASL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    gloss_to_entry = {e["gloss"]: e for e in data}

    missing = [g for g in TARGET.values() if g not in gloss_to_entry]
    if missing:
        print("ERROR: Missing glosses in WLASL_v0.3.json:", missing)
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Replace: clear output dirs
    if REPLACE_EXISTING:
        for label in TARGET.keys():
            d = OUT_ROOT / label
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0

    for label, gloss in TARGET.items():
        entry = gloss_to_entry[gloss]
        instances = entry.get("instances", [])
        out_dir = OUT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{label} <- WLASL gloss '{gloss}' ({len(instances)} instances)")

        for inst in instances:
            url = inst["url"]
            vid = inst["video_id"]
            # Skip aslpro swf entries
            if "aslpro" in url or url.rstrip().lower().endswith(".swf"):
                continue

            save_path = out_dir / f"{vid}.mp4"
            if save_path.exists():
                continue

            total += 1
            if is_youtube(url):
                ok += 1 if download_youtube(url, save_path) else 0
            else:
                ok += 1 if download_direct(url, save_path) else 0

    print(f"\nDone. Downloaded {ok}/{total} videos into {OUT_ROOT}")


if __name__ == "__main__":
    main()

