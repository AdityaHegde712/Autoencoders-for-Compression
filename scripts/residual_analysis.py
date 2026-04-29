"""
Capture camera frames and compute residuals against periodically refreshed I-frames.

Usage:
    python scripts/residual_analysis.py
    python scripts/residual_analysis.py --camera 0 --fps 24 --iframe-interval 10

Keys:
    q  Quit
    r  Reset stream and re-select I-frame cadence
"""

import argparse
import time

import cv2
import numpy as np


WINDOW_NAME = "Residual From I-Frame (press q to quit, r to reset)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam residual demo using periodic I-frame refresh.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--fps", type=float, default=24.0, help="Target processing FPS (default: 24).")
    parser.add_argument(
        "--iframe-interval",
        type=int,
        default=10,
        help="Refresh I-frame every N frames (default: 10).",
    )
    parser.add_argument("--width", type=int, default=1920, help="Capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Capture height.")
    return parser.parse_args()


def annotate(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    target_fps = args.fps if args.fps > 0 else 24.0
    frame_interval = 1.0 / target_fps
    iframe_interval = max(1, args.iframe_interval)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    iframe_bgr = None
    stream_index = -1

    print(f"Target FPS: {target_fps:.2f}")
    print(f"I-frame refresh interval: every {iframe_interval} frame(s)")

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera frame grab failed.")
                break

            stream_index += 1

            if stream_index % iframe_interval == 0:
                iframe_bgr = frame_bgr.copy()
                print(f"Set I-frame at stream frame {stream_index}.")

            if iframe_bgr is None:
                wait_frame = frame_bgr.copy()
                annotate(wait_frame, f"Waiting for first I-frame (current {stream_index})")
                cv2.imshow(WINDOW_NAME, wait_frame)
            else:
                residual = cv2.absdiff(frame_bgr, iframe_bgr)

                # Boost visibility while preserving directionless per-pixel difference.
                residual_vis = cv2.convertScaleAbs(residual, alpha=2.0, beta=0)

                current = frame_bgr.copy()
                iframe_view = iframe_bgr.copy()
                annotate(current, f"Current frame: {stream_index}")
                annotate(iframe_view, f"I-frame refresh: every {iframe_interval}f")
                annotate(residual_vis, "Residual |Current - I-frame|")

                canvas = np.hstack((current, iframe_view, residual_vis))
                cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                iframe_bgr = None
                stream_index = -1
                print("Stream reset. Re-selecting I-frame.")

            elapsed = time.perf_counter() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
