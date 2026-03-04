from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "sumitrodatta/nba-aba-baa-stats"
DEST_DIR = Path("data/raw")


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Download latest version to kagglehub cache.
    path = Path(kagglehub.dataset_download(DATASET))
    print(f"Path to dataset files: {path}")

    csv_files = sorted(path.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found in downloaded dataset path.")
        return

    copied = 0
    for src in csv_files:
        dst = DEST_DIR / src.name
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} CSV file(s) to: {DEST_DIR.resolve()}")


if __name__ == "__main__":
    main()
