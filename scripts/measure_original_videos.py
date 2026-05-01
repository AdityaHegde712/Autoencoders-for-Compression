"""
Measure codec and bitrate/BPP for the source video files.

BPP here is the compressed source-file size in bits divided by decoded pixels:
    file_size_bytes * 8 / (width * height * frame_count)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "virat_aerial_videos"


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def run_ffprobe(video_path: Path, count_frames: bool):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        cmd.append("-count_frames")
    cmd.extend([
        "-show_entries",
        (
            "stream=codec_name,codec_long_name,width,height,avg_frame_rate,"
            "r_frame_rate,nb_frames,nb_read_frames,duration,bit_rate"
        ),
        "-of",
        "json",
        str(video_path),
    ])

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def inspect_video(video_path: Path, count_frames: bool):
    info = run_ffprobe(video_path, count_frames=count_frames)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or 0.0)

    frame_count_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count_raw and frame_count_raw != "N/A":
        frame_count = int(frame_count_raw)
        frame_count_source = "ffprobe"
    elif fps > 0 and duration > 0:
        frame_count = round(fps * duration)
        frame_count_source = "duration*fps"
    else:
        frame_count = 0
        frame_count_source = "unknown"

    file_bytes = video_path.stat().st_size
    source_bpp = (file_bytes * 8 / (width * height * frame_count)) if frame_count > 0 else 0.0
    raw_24bit_bytes = width * height * frame_count * 3
    compression_ratio = (raw_24bit_bytes / file_bytes) if file_bytes > 0 else 0.0

    return {
        "file": video_path.name,
        "codec": stream.get("codec_name", "unknown"),
        "codec_long_name": stream.get("codec_long_name", "unknown"),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_s": duration,
        "frames": frame_count,
        "frame_count_source": frame_count_source,
        "file_bytes": file_bytes,
        "container_bit_rate_bps": int(stream.get("bit_rate") or 0),
        "source_bpp": source_bpp,
        "compression_ratio_vs_raw_24bit": compression_ratio,
    }


def format_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def print_table(rows):
    if not rows:
        print("No videos found.")
        return

    header = (
        f"{'file':34} {'codec':12} {'resolution':11} {'fps':>7} "
        f"{'frames':>8} {'size':>10} {'bpp':>8} {'raw ratio':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        resolution = f"{row['width']}x{row['height']}"
        print(
            f"{row['file'][:34]:34} "
            f"{row['codec'][:12]:12} "
            f"{resolution:11} "
            f"{row['fps']:7.2f} "
            f"{row['frames']:8d} "
            f"{format_bytes(row['file_bytes']):>10} "
            f"{row['source_bpp']:8.4f} "
            f"{row['compression_ratio_vs_raw_24bit']:9.2f}:1"
        )

    total_bits = sum(row["file_bytes"] * 8 for row in rows)
    total_pixels = sum(row["width"] * row["height"] * row["frames"] for row in rows)
    total_raw_bytes = sum(row["width"] * row["height"] * row["frames"] * 3 for row in rows)
    total_file_bytes = sum(row["file_bytes"] for row in rows)

    print("-" * len(header))
    print(f"Videos measured: {len(rows)}")
    if total_pixels > 0:
        print(f"Weighted average source BPP: {total_bits / total_pixels:.4f}")
    if total_file_bytes > 0:
        print(f"Total compression ratio vs raw 24-bit video: {total_raw_bytes / total_file_bytes:.2f}:1")
    codecs = sorted({f"{row['codec']} ({row['codec_long_name']})" for row in rows})
    print(f"Codec(s): {', '.join(codecs)}")


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Measure source video codec, bitrate, and BPP.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Directory containing source videos")
    parser.add_argument("--pattern", default="*.mpg", help="Glob pattern inside --video-dir")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--count-frames",
        action="store_true",
        help="Ask ffprobe to count frames exactly. Slower, but avoids duration*fps estimates.",
    )
    args = parser.parse_args()

    if not args.video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    video_paths = sorted(args.video_dir.glob(args.pattern))
    rows = [inspect_video(path, count_frames=args.count_frames) for path in video_paths]
    print_table(rows)

    if args.csv:
        if not rows:
            print("No CSV written because no videos were found.")
            return
        write_csv(rows, args.csv)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"ffprobe failed: {exc}", file=sys.stderr)
        sys.exit(1)
