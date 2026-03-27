"""
Download VIRAT aerial video dataset from Kitware Data Portal.
Source: https://data.kitware.com/#collection/611e77a42fa25629b9daceba

Downloads 25 .mpg files (~4.9 GB total) into data/virat_aerial_videos/
Uses parallel downloads (4 workers by default) for faster throughput.
"""

import os
import sys
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

DOWNLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "virat_aerial_videos",
)

BASE_URL = "https://data.kitware.com/api/v1/item/{}/download"
MAX_WORKERS = 25

FILES = [
    ("56f57fa48d777f753209c714", "09152008flight2tape1_1.mpg", 192100092),
    ("56f57fb48d777f753209c717", "09152008flight2tape1_10.mpg", 198934832),
    ("56f57fcd8d777f753209c71a", "09152008flight2tape1_2.mpg", 198984652),
    ("56f57fde8d777f753209c71d", "09152008flight2tape1_3.mpg", 198224944),
    ("56f57fef8d777f753209c720", "09152008flight2tape1_5.mpg", 197049380),
    ("56f580018d777f753209c723", "09152008flight2tape1_6.mpg", 203474656),
    ("56f580128d777f753209c726", "09152008flight2tape1_7.mpg", 201632820),
    ("56f580258d777f753209c729", "09152008flight2tape2_1.mpg", 193533592),
    ("56f580368d777f753209c72c", "09152008flight2tape2_2.mpg", 199919012),
    ("56f580488d777f753209c72f", "09152008flight2tape2_4.mpg", 203306960),
    ("56f5805a8d777f753209c732", "09152008flight2tape2_5.mpg", 197296788),
    ("56f5806c8d777f753209c735", "09152008flight2tape2_8.mpg", 202941676),
    ("56f580808d777f753209c738", "09152008flight2tape2_9.mpg", 199365540),
    ("56f580928d777f753209c73b", "09152008flight2tape3_1.mpg", 189933392),
    ("56f580a48d777f753209c73e", "09152008flight2tape3_2.mpg", 199210440),
    ("56f580b58d777f753209c741", "09152008flight2tape3_3.mpg", 197221588),
    ("56f580c78d777f753209c744", "09152008flight2tape3_4.mpg", 197300924),
    ("56f580d88d777f753209c747", "09152008flight2tape3_9.mpg", 197182860),
    ("56f580ec8d777f753209c74a", "09162008flight1tape1_1.mpg", 191449424),
    ("56f580fc8d777f753209c74d", "09162008flight1tape1_2.mpg", 197061788),
    ("56f5810e8d777f753209c750", "09162008flight1tape1_3.mpg", 198362936),
    ("56f5811f8d777f753209c753", "09162008flight1tape1_4.mpg", 199009280),
    ("56f581308d777f753209c756", "09162008flight1tape1_5.mpg", 198960964),
    ("56f581418d777f753209c759", "09172008flight1tape1_5.mpg", 197091116),
    ("5ef11b419014a6d84ed53971", "09172008flight1tape3_2.mpg", 198485136),
]

print_lock = threading.Lock()


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def log(msg):
    with print_lock:
        print(msg, flush=True)


def download_file(index, total, item_id, filename, expected_size):
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    # Skip if already downloaded with correct size
    if os.path.exists(filepath) and os.path.getsize(filepath) == expected_size:
        log(f"[{index}/{total}] [skip] {filename} (already downloaded)")
        return True

    url = BASE_URL.format(item_id)
    tmp_path = filepath + ".part"

    try:
        log(f"[{index}/{total}] [downloading] {filename} ({format_bytes(expected_size)})")
        start = time.time()

        urllib.request.urlretrieve(url, tmp_path)
        elapsed = time.time() - start

        # Verify size
        actual = os.path.getsize(tmp_path)
        if actual != expected_size:
            os.remove(tmp_path)
            log(f"[{index}/{total}] [error] {filename}: size mismatch (expected {expected_size}, got {actual})")
            return False

        os.rename(tmp_path, filepath)
        speed = expected_size / elapsed / 1024 / 1024
        log(f"[{index}/{total}] [done] {filename} — {elapsed:.0f}s ({speed:.1f} MB/s)")
        return True

    except Exception as e:
        log(f"[{index}/{total}] [error] {filename}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    total_size = sum(s for _, _, s in FILES)
    total = len(FILES)
    print(f"VIRAT Aerial Dataset Downloader")
    print(f"  Files: {total} videos")
    print(f"  Total: {format_bytes(total_size)}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Destination: {DOWNLOAD_DIR}")
    print()

    failures = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_file, i, total, item_id, name, size): name
            for i, (item_id, name, size) in enumerate(FILES, 1)
        }

        for future in as_completed(futures):
            name = futures[future]
            if not future.result():
                failures.append(name)

    elapsed = time.time() - start
    successes = total - len(failures)
    print(f"\nComplete: {successes}/{total} downloaded in {elapsed:.0f}s")
    if failures:
        print(f"Failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
