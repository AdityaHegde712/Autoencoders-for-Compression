import cv2
import os
import glob
from tqdm import tqdm

VIDEO_DIR = r'c:\Users\hifia\Projects\Autoencoders-for-Compression\data\virat_aerial_videos'
OUTPUT_DIR = r'c:\Users\hifia\Projects\Autoencoders-for-Compression\data\processed_frames'

def extract_all():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mpg"))
    print(f"Found {len(video_paths)} videos to process.")
    
    for v_path in tqdm(video_paths, desc="Videos"):
        v_name = os.path.splitext(os.path.basename(v_path))[0]
        v_out_dir = os.path.join(OUTPUT_DIR, v_name)
        
        if not os.path.exists(v_out_dir):
            os.makedirs(v_out_dir)
        else:
            # Skip if already extracted (basic check)
            if len(os.listdir(v_out_dir)) > 10:
                continue
                
        cap = cv2.VideoCapture(v_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame as JPG (reasonable quality/space trade-off)
            frame_path = os.path.join(v_out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx += 1
            
        cap.release()
    print("Extraction complete!")

if __name__ == "__main__":
    extract_all()
