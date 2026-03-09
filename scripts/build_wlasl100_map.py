"""
Build WLASL100_map.json from the official WLASL repo's WLASL_v0.3.json.
Keeps only the top 100 most frequent glosses (by instance count).
Run from project root; expects wlasl_source/ to exist (clone WLASL repo first).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WLASL_JSON = PROJECT_ROOT / "wlasl_source" / "start_kit" / "WLASL_v0.3.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "WLASL100_map.json"
TOP_K = 100


def main():
    if not WLASL_JSON.exists():
        print(f"Error: {WLASL_JSON} not found. Clone WLASL repo into wlasl_source/ first.")
        sys.exit(1)

    print(f"Loading {WLASL_JSON.name}...")
    with open(WLASL_JSON, "r", encoding="utf-8") as f:
        content = json.load(f)

    # Sort by instance count descending, take top 100
    sorted_entries = sorted(content, key=lambda e: -len(e["instances"]))
    top100 = sorted_entries[:TOP_K]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(top100, f, indent=2)

    total_instances = sum(len(e["instances"]) for e in top100)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Glosses: {len(top100)}, Total video instances: {total_instances}")
    print("Top 10:", [e["gloss"] for e in top100[:10]])


if __name__ == "__main__":
    main()
