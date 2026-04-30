import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
# Use /content/ for speed; move to /content/drive/ only after extraction is done
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'virat_video')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_frames')

# Ensure directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(v_path):
    """Processes a single video using GPU acceleration."""
    v_name = os.path.splitext(os.path.basename(v_path))[0]
    v_out_dir = os.path.join(OUTPUT_DIR, v_name)
    os.makedirs(v_out_dir, exist_ok=True)
    
    # Skip if already processed
    if len(os.listdir(v_out_dir)) > 10:
        return

    frame_pattern = os.path.join(v_out_dir, "frame_%06d.jpg")

    # FFmpeg GPU Pipeline:
    # 1. -hwaccel cuda: Uses GPU for decoding
    # 2. yadif_cuda: Deinterlaces on the GPU (important for .mpg files)
    # 3. hwdownload: Moves processed frame to system RAM for saving
    # 4. -q:v 2: High quality JPEG setting
    cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", v_path,
        "-vf", "yadif_cuda,hwdownload,format=nv12",
        "-q:v", "2",
        "-pix_fmt", "yuvj420p",
        "-threads", "1", # Prevents CPU thread contention
        "-y", frame_pattern
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print(f"Error processing {v_name}")

def extract_all():
    # Find all .mpg videos
    video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    if not video_paths:
        print(f"No videos found in {VIDEO_DIR}. Please check the path.")
        return

    print(f"Found {len(video_paths)} videos. Starting GPU extraction...")

    # Colab T4 GPUs handle 2-4 concurrent decodes efficiently
    # Max workers = 2 balances GPU load and Disk I/O
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process_video, video_paths), total=len(video_paths), desc="Processing Videos"))

    print("\nExtraction complete!")
    print(f"Frames saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # 1. Verify FFmpeg has CUDA support in this Colab env
    check_gpu = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True)
    if "cuda" not in check_gpu.stdout:
        print("WARNING: CUDA not found in FFmpeg. Ensure you are using a GPU runtime.")
    
    extract_all()
