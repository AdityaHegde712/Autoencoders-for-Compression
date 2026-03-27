import cv2
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(PROJECT_ROOT, 'data', 'virat_aerial_videos', '09152008flight2tape1_1.mpg')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
else:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frame Count: {count}")
    cap.release()
