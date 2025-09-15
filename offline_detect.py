import os
import sys
import argparse
from ultralytics import YOLO
import time
import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Offline YOLO detection on video/images"
    )
    parser.add_argument(
        '--model',
        type=str,
        default="my_model/my_model.pt",
        help='Path to YOLO model file (default: my_model/my_model.pt)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default="my_model/PitstopVideo.mov",
        help='Input source: image file, folder of images, or video file'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.2,
        help='Confidence threshold for detections (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional name of output folder (default: runs/detect/predict)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",   # ✅ safer default for Mac
        help='Device to run on: "cpu", "mps" (Apple Silicon), or "0" for GPU (Linux/Windows)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'❌ ERROR: Model not found at {args.model}')
        sys.exit(1)

    print(f"✅ Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"▶️ Processing source: {args.source}")
    start_time = time.time()

    # Get total frames if it's a video (for progress bar)
    total_frames = None
    if args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
        cap = cv2.VideoCapture(args.source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    # Run inference with streaming mode
    results = model.predict(
        source=args.source,
        save=True,
        conf=args.conf,
        device=args.device,
        project="runs/detect",
        name=args.output if args.output else "predict",
        verbose=False,
        stream=True   # ✅ avoids RAM buildup
    )

    # Iterate with progress bar if video
    if total_frames:
        for _ in tqdm(results, total=total_frames, desc="Processing frames"):
            pass
    else:
        for _ in results:
            pass

    end_time = time.time()
    elapsed = end_time - start_time

    print("✅ Processing complete!")
    print(f"📂 Results saved to: {results.save_dir}")

    # If source was a video, show FPS
    if total_frames:
        fps_proc = total_frames / elapsed if elapsed > 0 else 0
        print(f"⏱ Processed {total_frames} frames in {elapsed:.2f} sec → {fps_proc:.2f} FPS")
    else:
        print(f"⏱ Elapsed time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
