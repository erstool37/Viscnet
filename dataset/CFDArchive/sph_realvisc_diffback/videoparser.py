import cv2
import os
import re
from pathlib import Path

##### Settings
VIDEO_DIR = "videos"        # folder containing input videos
OUTPUT_DIR = "../../../../Flumesh/images"       # folder to save PNGs
MAX_FRAMES = 33             # extract frames 0–32
TARGET_LEN = 2010 * 20      # final dataset size
os.makedirs(OUTPUT_DIR, exist_ok=True)

##### Collect valid video paths
video_paths = []
for path in Path(VIDEO_DIR).glob("*.mp4"):
    m = re.search(r"(\d+)$", path.stem)
    if m:
        num = int(m.group(1))
        if num < 2010:
            video_paths.append(path)

print(f"✅ Found {len(video_paths)} valid videos (<2010)")

if len(video_paths) == 0:
    raise RuntimeError("No valid videos found!")

##### Expand to exact TARGET_LEN
multiplier = (TARGET_LEN + len(video_paths) - 1) // len(video_paths)
video_paths = (video_paths * multiplier)[:TARGET_LEN]
print(f"✅ Final dataset length: {len(video_paths)}")

##### Parse videos into frames
for video_path in video_paths:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        continue

    base_name = video_path.stem
    frame_count = 0

    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        out_name = f"{base_name}_{frame_count}.png"
        out_path = Path(OUTPUT_DIR) / out_name
        cv2.imwrite(str(out_path), frame)
        frame_count += 1

    cap.release()
    print(f"✅ Saved {frame_count} frames from {video_path.name}")

print("All done ✅")