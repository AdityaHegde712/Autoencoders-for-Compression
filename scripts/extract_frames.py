import os
import glob
import subprocess
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'virat_aerial_videos')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_frames')

def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Note: Ensure files are .mpg, or change to glob.glob(os.path.join(VIDEO_DIR, "*"))
    video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mpg"))
    print(f"Found {len(video_paths)} videos to process.")

    for v_path in tqdm(video_paths, desc="Videos"):
        v_name = os.path.splitext(os.path.basename(v_path))[0]
        v_out_dir = os.path.join(OUTPUT_DIR, v_name)
        os.makedirs(v_out_dir, exist_ok=True)

        if len(os.listdir(v_out_dir)) > 10:
            continue

        frame_pattern = os.path.join(v_out_dir, "frame_%06d.jpg")
        
        # GPU accelerated command
        subprocess.run(
            [
                "ffmpeg",
                "-hwaccel", "cuda",          # Enable CUDA hardware acceleration
                "-hwaccel_output_format", "cuda",
                "-i", v_path,
                "-vf", "yadif_cuda,hwdownload,format=nv12",         # Use GPU-based deinterlacer
                "-q:v", "2",                 # Quality setting
                "-y",
                frame_pattern,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    print("Extraction complete!")

if __name__ == "__main__":
    extract_all()
