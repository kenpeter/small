#!/usr/bin/env python3
"""Track download progress."""
import time
from pathlib import Path
import sys

DATA_DIR = Path("/home/kenpeter/work/data/_staging_multi")

DATASETS = {
    "Stack Python": (DATA_DIR / "stack-python", 144),
    "FineMath": (DATA_DIR / "finemath-3plus", 128),
    "Cosmopedia": (DATA_DIR / "cosmopedia", 13),
    "OpenWebMath": (DATA_DIR / "open-web-math", 114),
}

print("Download Tracker Started")
print("=" * 60)
print(f"{'Dataset':<20} {'Files':>10} {'Size':>10} {'Status':>10}")
print("-" * 60)

prev_counts = {name: 0 for name in DATASETS}

while True:
    total_size = 0
    lines = []
    
    for name, (path, target) in DATASETS.items():
        if path.exists():
            files = list(path.glob("*.parquet"))
            count = len(files)
            size = sum(f.stat().st_size for f in files) / (1024**3)  # GB
            total_size += size
            
            status = "✅" if count >= target else f"{count/target*100:.0f}%"
            if count > prev_counts[name]:
                status += f" (+{count - prev_counts[name]})"
            
            lines.append(f"{name:<20} {count:>3}/{target:<3} {size:>6.1f}GB {status:>10}")
            prev_counts[name] = count
        else:
            lines.append(f"{name:<20} {0:>3}/{target:<3} {0:>6.1f}GB {'❌':>10}")
    
    # Clear screen and print
    sys.stdout.write("\033[2J\033[H")  # Clear screen
    print("Download Tracker - Press Ctrl+C to stop")
    print("=" * 60)
    print(f"{'Dataset':<20} {'Files':>10} {'Size':>10} {'Status':>10}")
    print("-" * 60)
    for line in lines:
        print(line)
    print("-" * 60)
    print(f"{'TOTAL':<20} {'':>10} {total_size:>6.1f}GB")
    print(f"\nUpdated: {time.strftime('%H:%M:%S')}")
    
    time.sleep(30)
