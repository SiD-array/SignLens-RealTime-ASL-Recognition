"""
Download WLASL100 videos into project structure: data/extracted_dynamic/{gloss}/{video_id}.mp4.
Uses WLASL100_map.json (build with scripts/build_wlasl100_map.py if missing).
Requires: yt-dlp (YouTube + some direct), urllib (direct MP4). No pytube needed.
"""
import json
import logging
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DYNAMIC = DATA_DIR / "extracted_dynamic"
WLASL100_MAP = DATA_DIR / "WLASL100_map.json"
WLASL_SOURCE_JSON = PROJECT_ROOT / "wlasl_source" / "start_kit" / "WLASL_v0.3.json"

YT_DLP = "yt-dlp"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Build WLASL100_map.json if missing
# -----------------------------------------------------------------------------
def ensure_wlasl100_map():
    if WLASL100_MAP.exists():
        log.info("Using existing %s", WLASL100_MAP.name)
        return
    if not WLASL_SOURCE_JSON.exists():
        log.error("Neither %s nor %s found. Run scripts/build_wlasl100_map.py after cloning WLASL.", WLASL100_MAP.name, WLASL_SOURCE_JSON)
        sys.exit(1)
    log.info("Building %s from WLASL_v0.3.json (top 100 glosses)...", WLASL100_MAP.name)
    with open(WLASL_SOURCE_JSON, "r", encoding="utf-8") as f:
        content = json.load(f)
    top100 = sorted(content, key=lambda e: -len(e["instances"]))[:100]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(WLASL100_MAP, "w", encoding="utf-8") as f:
        json.dump(top100, f, indent=2)
    log.info("Wrote %s (%d glosses)", WLASL100_MAP, len(top100))


# -----------------------------------------------------------------------------
# Download helpers
# -----------------------------------------------------------------------------
def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def is_aslpro_swf(url: str) -> bool:
    return "aslpro" in url or url.rstrip().endswith(".swf")


def download_direct(url: str, save_path: Path, referer: str = "") -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if referer:
            req.add_header("Referer", referer)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        log.warning("Direct download failed %s: %s", url[:60], e)
        return False


def download_youtube(url: str, save_path: Path) -> bool:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [YT_DLP, "-q", "--no-warnings", "-f", "best[ext=mp4]/best", "-o", str(save_path), url],
            check=True,
            timeout=120,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("yt-dlp failed for %s: %s", url[:50], e)
        return False


def select_download(url: str, gloss: str, video_id: str) -> bool:
    out_dir = EXTRACTED_DYNAMIC / gloss
    out_path = out_dir / f"{video_id}.mp4"

    if out_path.exists():
        log.debug("Skip (exists): %s/%s", gloss, video_id)
        return True

    if is_aslpro_swf(url):
        log.debug("Skip .swf: %s", video_id)
        return False

    if is_youtube(url):
        ok = download_youtube(url, out_path)
    else:
        referer = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi" if "aslpro" in url else ""
        ok = download_direct(url, out_path, referer=referer)

    if ok:
        time.sleep(random.uniform(0.5, 1.5))
    return ok


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def check_dependencies():
    try:
        subprocess.run([YT_DLP, "--version"], capture_output=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.error("yt-dlp not found or failed. Install with: pip install yt-dlp")
        sys.exit(1)


def main():
    check_dependencies()
    ensure_wlasl100_map()

    with open(WLASL100_MAP, "r", encoding="utf-8") as f:
        content = json.load(f)

    EXTRACTED_DYNAMIC.mkdir(parents=True, exist_ok=True)
    total = 0
    ok_count = 0
    for entry in content:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            url = inst["url"]
            video_id = inst["video_id"]
            total += 1
            if select_download(url, gloss, video_id):
                ok_count += 1
            if total % 50 == 0:
                log.info("Progress: %d processed, %d ok", total, ok_count)

    log.info("Done. Total %d, downloaded/ok %d. Videos under %s", total, ok_count, EXTRACTED_DYNAMIC)
    log.info("For 126-feature extraction: set labels.json to the 100 glosses (keys in %s)", WLASL100_MAP.name)


if __name__ == "__main__":
    main()
